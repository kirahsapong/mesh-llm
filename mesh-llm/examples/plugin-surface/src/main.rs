use anyhow::{bail, Context, Result};
use prost::Message;
use rmcp::model::{
    CallToolResult, Content, Implementation, ListToolsResult, ServerCapabilities, ServerInfo,
    Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

const PLUGIN_ID: &str = "example";
const PROTOCOL_VERSION: u32 = 1;
const EXAMPLE_CHANNEL: &str = "example.v1";
const DEFAULT_BULK_CHUNK_SIZE: usize = 16 * 1024;
const DEFAULT_SNAPSHOT_LIMIT: usize = 20;
const MAX_RECENT_ITEMS: usize = 128;

#[cfg(unix)]
type LocalStream = tokio::net::UnixStream;

#[cfg(unix)]
async fn connect(endpoint: &str) -> Result<LocalStream> {
    Ok(tokio::net::UnixStream::connect(endpoint).await?)
}

#[cfg(not(unix))]
compile_error!("plugin-surface example currently supports unix-domain socket plugin transport only");

#[derive(Debug, Deserialize, Default)]
struct SnapshotParams {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendMessageArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SendBulkArguments {
    text: String,
    #[serde(default)]
    target_peer_id: Option<String>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    request_ack: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ClearParams {}

#[derive(Clone, Debug, Serialize)]
struct PeerSummary {
    peer_id: String,
    role: String,
    version: String,
    capabilities: Vec<String>,
    models: Vec<String>,
    serving_models: Vec<String>,
    rtt_ms: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedMeshEvent {
    kind: String,
    peer_id: Option<String>,
    local_peer_id: String,
    mesh_id: String,
    detail_json: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedChannelMessage {
    direction: String,
    channel: String,
    source_peer_id: String,
    target_peer_id: String,
    message_kind: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    text_preview: String,
    body_len: usize,
}

#[derive(Clone, Debug, Serialize)]
struct RecordedBulkEvent {
    direction: String,
    kind: String,
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    correlation_id: String,
    content_type: String,
    metadata_json: Option<serde_json::Value>,
    total_bytes: u64,
    offset: u64,
    body_len: usize,
    final_chunk: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CompletedTransfer {
    transfer_id: String,
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    received_bytes: u64,
    preview: String,
}

#[derive(Default)]
struct TransferAccumulator {
    source_peer_id: String,
    target_peer_id: String,
    content_type: String,
    total_bytes: u64,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct ExampleState {
    local_peer_id: String,
    mesh_id: String,
    known_peers: BTreeMap<String, PeerSummary>,
    mesh_events: Vec<RecordedMeshEvent>,
    channel_messages: Vec<RecordedChannelMessage>,
    bulk_events: Vec<RecordedBulkEvent>,
    completed_transfers: BTreeMap<String, CompletedTransfer>,
    transfer_state: HashMap<String, TransferAccumulator>,
    sent_channel_messages: usize,
    sent_bulk_transfers: usize,
    next_id: u64,
}

impl ExampleState {
    fn next_token(&mut self, prefix: &str) -> String {
        self.next_id += 1;
        format!("{prefix}-{}-{}", now_millis(), self.next_id)
    }

    fn snapshot(&self, limit: usize) -> serde_json::Value {
        let limit = limit.max(1);
        let completed_transfers = self
            .completed_transfers
            .values()
            .cloned()
            .collect::<Vec<_>>();
        json!({
            "plugin": PLUGIN_ID,
            "channel": EXAMPLE_CHANNEL,
            "local_peer_id": self.local_peer_id,
            "mesh_id": self.mesh_id,
            "known_peers": self.known_peers.values().cloned().collect::<Vec<_>>(),
            "stats": {
                "known_peer_count": self.known_peers.len(),
                "mesh_event_count": self.mesh_events.len(),
                "channel_message_count": self.channel_messages.len(),
                "bulk_event_count": self.bulk_events.len(),
                "completed_transfer_count": self.completed_transfers.len(),
                "sent_channel_messages": self.sent_channel_messages,
                "sent_bulk_transfers": self.sent_bulk_transfers,
            },
            "recent_mesh_events": recent_items(&self.mesh_events, limit),
            "recent_channel_messages": recent_items(&self.channel_messages, limit),
            "recent_bulk_events": recent_items(&self.bulk_events, limit),
            "completed_transfers": recent_items(&completed_transfers, limit),
        })
    }

    fn clear_history(&mut self) {
        self.mesh_events.clear();
        self.channel_messages.clear();
        self.bulk_events.clear();
        self.completed_transfers.clear();
        self.transfer_state.clear();
        self.sent_channel_messages = 0;
        self.sent_bulk_transfers = 0;
    }

    fn record_mesh_event(&mut self, event: &proto::MeshEvent) {
        if !event.local_peer_id.is_empty() {
            self.local_peer_id = event.local_peer_id.clone();
        }
        if !event.mesh_id.is_empty() {
            self.mesh_id = event.mesh_id.clone();
        }
        if let Some(peer) = &event.peer {
            let peer_id = peer.peer_id.clone();
            match proto::mesh_event::Kind::try_from(event.kind).ok() {
                Some(proto::mesh_event::Kind::PeerDown) => {
                    self.known_peers.remove(&peer_id);
                }
                _ => {
                    self.known_peers.insert(peer_id, peer_summary(peer));
                }
            }
        }

        push_bounded(
            &mut self.mesh_events,
            RecordedMeshEvent {
                kind: mesh_event_kind_name(event.kind).into(),
                peer_id: event.peer.as_ref().map(|peer| peer.peer_id.clone()),
                local_peer_id: event.local_peer_id.clone(),
                mesh_id: event.mesh_id.clone(),
                detail_json: parse_optional_json(&event.detail_json),
            },
        );
    }

    fn record_channel_message(&mut self, direction: &str, message: &proto::ChannelMessage) {
        if direction == "outbound" {
            self.sent_channel_messages += 1;
        }
        push_bounded(
            &mut self.channel_messages,
            RecordedChannelMessage {
                direction: direction.to_string(),
                channel: message.channel.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                message_kind: message.message_kind.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                text_preview: preview_bytes(&message.body),
                body_len: message.body.len(),
            },
        );
    }

    fn record_bulk_message(&mut self, direction: &str, message: &proto::BulkTransferMessage) {
        if direction == "outbound"
            && matches!(
                proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
                Some(proto::bulk_transfer_message::Kind::Offer)
            )
        {
            self.sent_bulk_transfers += 1;
        }
        push_bounded(
            &mut self.bulk_events,
            RecordedBulkEvent {
                direction: direction.to_string(),
                kind: bulk_kind_name(message.kind).into(),
                transfer_id: message.transfer_id.clone(),
                source_peer_id: message.source_peer_id.clone(),
                target_peer_id: message.target_peer_id.clone(),
                correlation_id: message.correlation_id.clone(),
                content_type: message.content_type.clone(),
                metadata_json: parse_optional_json(&message.metadata_json),
                total_bytes: message.total_bytes,
                offset: message.offset,
                body_len: message.body.len(),
                final_chunk: message.final_chunk,
            },
        );
    }

    fn note_bulk_receive(&mut self, message: &proto::BulkTransferMessage) {
        match proto::bulk_transfer_message::Kind::try_from(message.kind).ok() {
            Some(proto::bulk_transfer_message::Kind::Offer) => {
                self.transfer_state.insert(
                    message.transfer_id.clone(),
                    TransferAccumulator {
                        source_peer_id: message.source_peer_id.clone(),
                        target_peer_id: message.target_peer_id.clone(),
                        content_type: message.content_type.clone(),
                        total_bytes: message.total_bytes,
                        bytes: Vec::new(),
                    },
                );
            }
            Some(proto::bulk_transfer_message::Kind::Chunk) => {
                let entry = self
                    .transfer_state
                    .entry(message.transfer_id.clone())
                    .or_default();
                if entry.source_peer_id.is_empty() {
                    entry.source_peer_id = message.source_peer_id.clone();
                }
                if entry.target_peer_id.is_empty() {
                    entry.target_peer_id = message.target_peer_id.clone();
                }
                if entry.content_type.is_empty() {
                    entry.content_type = message.content_type.clone();
                }
                if entry.total_bytes == 0 {
                    entry.total_bytes = message.total_bytes;
                }
                entry.bytes.extend_from_slice(&message.body);
            }
            Some(proto::bulk_transfer_message::Kind::Complete) => {
                if let Some(entry) = self.transfer_state.remove(&message.transfer_id) {
                    self.completed_transfers.insert(
                        message.transfer_id.clone(),
                        CompletedTransfer {
                            transfer_id: message.transfer_id.clone(),
                            source_peer_id: entry.source_peer_id,
                            target_peer_id: entry.target_peer_id,
                            content_type: entry.content_type,
                            total_bytes: entry.total_bytes,
                            received_bytes: entry.bytes.len() as u64,
                            preview: preview_bytes(&entry.bytes),
                        },
                    );
                }
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = std::env::var("MESH_LLM_PLUGIN_ENDPOINT")
        .context("MESH_LLM_PLUGIN_ENDPOINT is not set")?;
    let transport =
        std::env::var("MESH_LLM_PLUGIN_TRANSPORT").unwrap_or_else(|_| "unix".to_string());
    if transport != "unix" {
        bail!("unsupported transport '{transport}'");
    }

    let mut stream = connect(&endpoint).await?;
    let state = Arc::new(Mutex::new(ExampleState::default()));

    loop {
        let envelope = read_envelope(&mut stream).await?;
        match envelope.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                write_envelope(
                    &mut stream,
                    &proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: PLUGIN_ID.into(),
                        request_id: envelope.request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: PLUGIN_ID.into(),
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: env!("CARGO_PKG_VERSION").into(),
                                server_info_json: serde_json::to_string(&server_info())?,
                                capabilities: vec![
                                    format!("channel:{EXAMPLE_CHANNEL}"),
                                    "bulk:example".into(),
                                    "mesh-events".into(),
                                ],
                            },
                        )),
                    },
                )
                .await?;
            }
            Some(proto::envelope::Payload::HealthRequest(_)) => {
                let state = state.lock().await;
                write_envelope(
                    &mut stream,
                    &proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: PLUGIN_ID.into(),
                        request_id: envelope.request_id,
                        payload: Some(proto::envelope::Payload::HealthResponse(
                            proto::HealthResponse {
                                status: proto::health_response::Status::Ok as i32,
                                detail: format!(
                                    "peers={} messages={} bulk={} mesh_events={}",
                                    state.known_peers.len(),
                                    state.channel_messages.len(),
                                    state.bulk_events.len(),
                                    state.mesh_events.len()
                                ),
                            },
                        )),
                    },
                )
                .await?;
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                write_envelope(
                    &mut stream,
                    &proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: PLUGIN_ID.into(),
                        request_id: envelope.request_id,
                        payload: Some(proto::envelope::Payload::ShutdownResponse(
                            proto::ShutdownResponse {},
                        )),
                    },
                )
                .await?;
                break;
            }
            Some(proto::envelope::Payload::RpcRequest(request)) => {
                let payload = handle_rpc_request(&state, &mut stream, request).await?;
                write_envelope(
                    &mut stream,
                    &proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: PLUGIN_ID.into(),
                        request_id: envelope.request_id,
                        payload: Some(payload),
                    },
                )
                .await?;
            }
            Some(proto::envelope::Payload::ChannelMessage(message)) => {
                let mut state = state.lock().await;
                state.record_channel_message("inbound", &message);
                if should_ack_channel(&message) {
                    let ack = proto::ChannelMessage {
                        channel: message.channel.clone(),
                        source_peer_id: String::new(),
                        target_peer_id: message.source_peer_id.clone(),
                        content_type: "application/json".into(),
                        body: serde_json::to_vec(&json!({
                            "acknowledged_kind": message.message_kind,
                            "received_bytes": message.body.len(),
                            "received_by": state.local_peer_id,
                        }))?,
                        message_kind: "example.ack".into(),
                        correlation_id: if message.correlation_id.is_empty() {
                            state.next_token("ack")
                        } else {
                            message.correlation_id.clone()
                        },
                        metadata_json: json!({
                            "reply_to": message.correlation_id,
                        })
                        .to_string(),
                    };
                    state.record_channel_message("outbound", &ack);
                    send_channel(&mut stream, ack).await?;
                }
            }
            Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                let mut state = state.lock().await;
                state.record_bulk_message("inbound", &message);
                state.note_bulk_receive(&message);
                if should_ack_bulk_offer(&message) {
                    let ack = proto::BulkTransferMessage {
                        kind: proto::bulk_transfer_message::Kind::Accept as i32,
                        transfer_id: message.transfer_id.clone(),
                        channel: message.channel.clone(),
                        source_peer_id: String::new(),
                        target_peer_id: message.source_peer_id.clone(),
                        content_type: message.content_type.clone(),
                        correlation_id: message.correlation_id.clone(),
                        metadata_json: json!({
                            "accepted_by": state.local_peer_id,
                        })
                        .to_string(),
                        total_bytes: message.total_bytes,
                        offset: 0,
                        body: Vec::new(),
                        final_chunk: false,
                    };
                    state.record_bulk_message("outbound", &ack);
                    send_bulk(&mut stream, ack).await?;
                }
            }
            Some(proto::envelope::Payload::MeshEvent(event)) => {
                state.lock().await.record_mesh_event(&event);
            }
            Some(proto::envelope::Payload::RpcNotification(_)) => {}
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                bail!("host error: {}", err.message);
            }
            _ => {}
        }
    }

    Ok(())
}

async fn handle_rpc_request(
    state: &Arc<Mutex<ExampleState>>,
    stream: &mut LocalStream,
    request: proto::RpcRequest,
) -> Result<proto::envelope::Payload> {
    match request.method.as_str() {
        "tools/list" => Ok(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
            result_json: serde_json::to_string(&list_tools_result())?,
        })),
        "tools/call" => {
            let params: ToolCallParams = serde_json::from_str(&request.params_json)
                .context("invalid tools/call params_json")?;
            let result = call_tool(state, stream, params).await?;
            Ok(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
                result_json: serde_json::to_string(&result)?,
            }))
        }
        method => Ok(proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
            code: -32601,
            message: format!("unsupported method '{method}'"),
            data_json: String::new(),
        })),
    }
}

#[derive(Debug, Deserialize)]
struct ToolCallParams {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

async fn call_tool(
    state: &Arc<Mutex<ExampleState>>,
    stream: &mut LocalStream,
    params: ToolCallParams,
) -> Result<CallToolResult> {
    let arguments = params.arguments.unwrap_or_else(|| json!({}));
    match params.name.as_str() {
        "snapshot" => {
            let params: SnapshotParams = serde_json::from_value(arguments).unwrap_or_default();
            let snapshot = state
                .lock()
                .await
                .snapshot(params.limit.unwrap_or(DEFAULT_SNAPSHOT_LIMIT));
            Ok(CallToolResult::structured(snapshot))
        }
        "clear" => {
            let _params: ClearParams = serde_json::from_value(arguments).unwrap_or_default();
            state.lock().await.clear_history();
            Ok(CallToolResult::structured(json!({
                "ok": true,
                "cleared": ["mesh_events", "channel_messages", "bulk_events", "completed_transfers"],
            })))
        }
        "send_message" => {
            let params: SendMessageArguments = serde_json::from_value(arguments)?;
            let mut state = state.lock().await;
            let target_peer_id = normalize_target_peer_id(params.target_peer_id);
            let correlation_id = state.next_token("msg");
            let message = proto::ChannelMessage {
                channel: EXAMPLE_CHANNEL.into(),
                source_peer_id: String::new(),
                target_peer_id: target_peer_id.clone(),
                content_type: "text/plain".into(),
                body: params.text.into_bytes(),
                message_kind: "example.message".into(),
                correlation_id: correlation_id.clone(),
                metadata_json: json!({
                    "request_ack": params.request_ack.unwrap_or(true),
                })
                .to_string(),
            };
            state.record_channel_message("outbound", &message);
            send_channel(stream, message).await?;
            Ok(CallToolResult::structured(json!({
                "ok": true,
                "channel": EXAMPLE_CHANNEL,
                "target_peer_id": render_target(&target_peer_id),
                "correlation_id": correlation_id,
            })))
        }
        "send_bulk" => {
            let params: SendBulkArguments = serde_json::from_value(arguments)?;
            let mut state = state.lock().await;
            let target_peer_id = normalize_target_peer_id(params.target_peer_id);
            let correlation_id = state.next_token("bulk-corr");
            let transfer_id = state.next_token("bulk");
            let bytes = params.text.into_bytes();
            let chunk_size = params.chunk_size.unwrap_or(DEFAULT_BULK_CHUNK_SIZE).max(1);
            let metadata_json = json!({
                "request_ack": params.request_ack.unwrap_or(true),
            })
            .to_string();

            let offer = proto::BulkTransferMessage {
                kind: proto::bulk_transfer_message::Kind::Offer as i32,
                transfer_id: transfer_id.clone(),
                channel: EXAMPLE_CHANNEL.into(),
                source_peer_id: String::new(),
                target_peer_id: target_peer_id.clone(),
                content_type: "text/plain".into(),
                correlation_id: correlation_id.clone(),
                metadata_json: metadata_json.clone(),
                total_bytes: bytes.len() as u64,
                offset: 0,
                body: Vec::new(),
                final_chunk: false,
            };
            state.record_bulk_message("outbound", &offer);
            send_bulk(stream, offer).await?;

            let mut offset = 0usize;
            for chunk in bytes.chunks(chunk_size) {
                let message = proto::BulkTransferMessage {
                    kind: proto::bulk_transfer_message::Kind::Chunk as i32,
                    transfer_id: transfer_id.clone(),
                    channel: EXAMPLE_CHANNEL.into(),
                    source_peer_id: String::new(),
                    target_peer_id: target_peer_id.clone(),
                    content_type: "text/plain".into(),
                    correlation_id: correlation_id.clone(),
                    metadata_json: metadata_json.clone(),
                    total_bytes: bytes.len() as u64,
                    offset: offset as u64,
                    body: chunk.to_vec(),
                    final_chunk: false,
                };
                offset += chunk.len();
                state.record_bulk_message("outbound", &message);
                send_bulk(stream, message).await?;
            }

            let complete = proto::BulkTransferMessage {
                kind: proto::bulk_transfer_message::Kind::Complete as i32,
                transfer_id: transfer_id.clone(),
                channel: EXAMPLE_CHANNEL.into(),
                source_peer_id: String::new(),
                target_peer_id: target_peer_id.clone(),
                content_type: "text/plain".into(),
                correlation_id: correlation_id.clone(),
                metadata_json,
                total_bytes: bytes.len() as u64,
                offset: bytes.len() as u64,
                body: Vec::new(),
                final_chunk: true,
            };
            state.record_bulk_message("outbound", &complete);
            send_bulk(stream, complete).await?;

            Ok(CallToolResult::structured(json!({
                "ok": true,
                "transfer_id": transfer_id,
                "target_peer_id": render_target(&target_peer_id),
                "total_bytes": bytes.len(),
                "chunk_size": chunk_size,
            })))
        }
        other => Ok(CallToolResult::error(vec![Content::text(format!(
            "unknown tool '{other}'"
        ))])),
    }
}

fn server_info() -> ServerInfo {
    ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
        Implementation::new(PLUGIN_ID, env!("CARGO_PKG_VERSION"))
            .with_title("Plugin Surface Example")
            .with_description(
                "Standalone example plugin that exercises tools, channel messages, bulk transfers, and mesh events.",
            ),
    )
}

fn list_tools_result() -> ListToolsResult {
    ListToolsResult {
        tools: vec![
            Tool::new(
                "snapshot",
                "Inspect the example plugin state: known peers, mesh events, recent channel messages, recent bulk transfers, and counters.",
                Arc::new(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "limit": { "type": "integer", "minimum": 1 }
                        }
                    })
                    .as_object()
                    .cloned()
                    .unwrap(),
                ),
            ),
            Tool::new(
                "send_message",
                "Send a plugin channel message to one peer or broadcast to all peers. Leave target_peer_id empty or set it to 'all' to broadcast.",
                Arc::new(schema_for::<SendMessageArguments>()),
            ),
            Tool::new(
                "send_bulk",
                "Send a bulk transfer to one peer or broadcast to all peers. This emits OFFER, CHUNK, and COMPLETE frames so the full bulk transport path is exercised.",
                Arc::new(schema_for::<SendBulkArguments>()),
            ),
            Tool::new(
                "clear",
                "Clear recorded example-plugin history while keeping the current peer snapshot.",
                Arc::new(
                    serde_json::json!({
                        "type": "object",
                        "additionalProperties": false
                    })
                    .as_object()
                    .cloned()
                    .unwrap(),
                ),
            ),
        ],
        meta: None,
        next_cursor: None,
    }
}

fn schema_for<T: JsonSchema>() -> serde_json::Map<String, serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T))
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "additionalProperties": true
            })
            .as_object()
            .cloned()
            .unwrap()
        })
}

async fn send_channel(stream: &mut LocalStream, message: proto::ChannelMessage) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: PLUGIN_ID.into(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::ChannelMessage(message)),
        },
    )
    .await
}

async fn send_bulk(stream: &mut LocalStream, message: proto::BulkTransferMessage) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: PLUGIN_ID.into(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::BulkTransferMessage(message)),
        },
    )
    .await
}

async fn write_envelope(stream: &mut LocalStream, envelope: &proto::Envelope) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        bail!("plugin frame too large");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(proto::Envelope::decode(body.as_slice())?)
}

fn should_ack_channel(message: &proto::ChannelMessage) -> bool {
    !message.source_peer_id.is_empty()
        && message.message_kind != "example.ack"
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn should_ack_bulk_offer(message: &proto::BulkTransferMessage) -> bool {
    matches!(
        proto::bulk_transfer_message::Kind::try_from(message.kind).ok(),
        Some(proto::bulk_transfer_message::Kind::Offer)
    ) && !message.source_peer_id.is_empty()
        && parse_optional_json(&message.metadata_json)
            .and_then(|value| value.get("request_ack").and_then(|v| v.as_bool()))
            .unwrap_or(false)
}

fn parse_optional_json(raw: &str) -> Option<serde_json::Value> {
    if raw.trim().is_empty() {
        None
    } else {
        serde_json::from_str(raw).ok()
    }
}

fn preview_bytes(bytes: &[u8]) -> String {
    let mut preview = String::from_utf8_lossy(bytes).to_string();
    if preview.len() > 160 {
        preview.truncate(160);
        preview.push_str("...");
    }
    preview
}

fn normalize_target_peer_id(target_peer_id: Option<String>) -> String {
    let Some(target_peer_id) = target_peer_id else {
        return String::new();
    };
    let trimmed = target_peer_id.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("all")
        || trimmed.eq_ignore_ascii_case("broadcast")
        || trimmed == "*"
    {
        String::new()
    } else {
        trimmed.to_string()
    }
}

fn render_target(target_peer_id: &str) -> String {
    if target_peer_id.is_empty() {
        "all".into()
    } else {
        target_peer_id.to_string()
    }
}

fn push_bounded<T>(items: &mut Vec<T>, item: T) {
    items.push(item);
    if items.len() > MAX_RECENT_ITEMS {
        let overflow = items.len() - MAX_RECENT_ITEMS;
        items.drain(0..overflow);
    }
}

fn recent_items<T: Clone>(items: &[T], limit: usize) -> Vec<T> {
    let len = items.len();
    let start = len.saturating_sub(limit);
    items[start..].to_vec()
}

fn peer_summary(peer: &proto::MeshPeer) -> PeerSummary {
    PeerSummary {
        peer_id: peer.peer_id.clone(),
        role: peer.role.clone(),
        version: peer.version.clone(),
        capabilities: peer.capabilities.clone(),
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        rtt_ms: peer.rtt_ms,
    }
}

fn mesh_event_kind_name(kind: i32) -> &'static str {
    match proto::mesh_event::Kind::try_from(kind).ok() {
        Some(proto::mesh_event::Kind::PeerUp) => "peer_up",
        Some(proto::mesh_event::Kind::PeerDown) => "peer_down",
        Some(proto::mesh_event::Kind::PeerUpdated) => "peer_updated",
        Some(proto::mesh_event::Kind::LocalAccepting) => "local_accepting",
        Some(proto::mesh_event::Kind::LocalStandby) => "local_standby",
        Some(proto::mesh_event::Kind::MeshIdUpdated) => "mesh_id_updated",
        _ => "unknown",
    }
}

fn bulk_kind_name(kind: i32) -> &'static str {
    match proto::bulk_transfer_message::Kind::try_from(kind).ok() {
        Some(proto::bulk_transfer_message::Kind::Offer) => "offer",
        Some(proto::bulk_transfer_message::Kind::Accept) => "accept",
        Some(proto::bulk_transfer_message::Kind::Reject) => "reject",
        Some(proto::bulk_transfer_message::Kind::Chunk) => "chunk",
        Some(proto::bulk_transfer_message::Kind::Complete) => "complete",
        Some(proto::bulk_transfer_message::Kind::Cancel) => "cancel",
        Some(proto::bulk_transfer_message::Kind::Error) => "error",
        _ => "unknown",
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
