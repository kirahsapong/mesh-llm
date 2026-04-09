use crate::crypto::keys::OwnerKeypair;
use crate::runtime::CoreRuntime;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;

type CancelFlagMap =
    Arc<Mutex<HashMap<String, (Arc<AtomicBool>, Arc<dyn crate::events::EventListener>)>>>;

pub const MAX_RECONNECT_ATTEMPTS: u32 = 10;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("runtime error: {0}")]
    Runtime(#[from] crate::runtime::RuntimeError),
    #[error("endpoint error: {0}")]
    Endpoint(String),
    #[error("join error: {0}")]
    Join(String),
}

#[derive(Clone, Debug)]
pub struct InviteToken(pub String);

impl InviteToken {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for InviteToken {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("empty invite token".to_string());
        }
        Ok(Self(s.to_string()))
    }
}

pub struct ClientConfig {
    pub owner_keypair: OwnerKeypair,
    pub invite_token: InviteToken,
    pub user_agent: String,
    pub connect_timeout: Duration,
}

pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    pub fn new(owner_keypair: OwnerKeypair, invite_token: InviteToken) -> Self {
        Self {
            config: ClientConfig {
                owner_keypair,
                invite_token,
                user_agent: format!("mesh-client-core/{}", env!("CARGO_PKG_VERSION")),
                connect_timeout: Duration::from_secs(30),
            },
        }
    }

    pub fn with_user_agent(mut self, ua: String) -> Self {
        self.config.user_agent = ua;
        self
    }

    pub fn with_connect_timeout(mut self, d: Duration) -> Self {
        self.config.connect_timeout = d;
        self
    }

    pub fn build(self) -> Result<MeshClient, ClientError> {
        let runtime = CoreRuntime::new()?;
        Ok(MeshClient {
            runtime,
            config: self.config,
            connected: false,
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
            listeners: Arc::new(Mutex::new(
                Vec::<Arc<dyn crate::events::EventListener>>::new(),
            )),
            reconnect_attempts: 0,
            user_disconnected: false,
        })
    }
}

pub struct MeshClient {
    runtime: CoreRuntime,
    pub(crate) config: ClientConfig,
    pub(crate) connected: bool,
    pub(crate) cancel_flags: CancelFlagMap,
    pub listeners: Arc<Mutex<Vec<Arc<dyn crate::events::EventListener>>>>,
    pub reconnect_attempts: u32,
    pub user_disconnected: bool,
}

impl MeshClient {
    /// Join the mesh using the invite token.
    pub async fn join(&mut self) -> Result<(), ClientError> {
        self.connected = true;
        self.emit_event(crate::events::Event::Connecting);
        self.emit_event(crate::events::Event::Joined {
            node_id: self.config.invite_token.0.clone(),
        });
        Ok(())
    }

    /// List available models on the mesh.
    pub async fn list_models(&self) -> Result<Vec<Model>, ClientError> {
        Ok(vec![])
    }

    /// Start a chat completion request. Sync — returns a `RequestId` immediately.
    /// Streaming tokens are delivered via `listener.on_event()` on the runtime thread.
    pub fn chat(
        &self,
        _request: ChatRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        let id = RequestId::new();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(id.0.clone(), (cancel_flag.clone(), listener.clone()));
        let id_clone = id.0.clone();
        self.runtime.handle().spawn(async move {
            if !cancel_flag.load(Ordering::Relaxed) {
                listener.on_event(crate::events::Event::Completed {
                    request_id: id_clone,
                });
            }
        });
        id
    }

    /// Start a responses request. Sync — returns a `RequestId` immediately.
    pub fn responses(
        &self,
        _request: ResponsesRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        let id = RequestId::new();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(id.0.clone(), (cancel_flag.clone(), listener.clone()));
        let id_clone = id.0.clone();
        self.runtime.handle().spawn(async move {
            if !cancel_flag.load(Ordering::Relaxed) {
                listener.on_event(crate::events::Event::Completed {
                    request_id: id_clone,
                });
            }
        });
        id
    }

    /// Cancel an in-flight request. No-op if the `request_id` is unknown.
    /// Emits `Event::Failed { error: "cancelled" }` to the request's listener when found.
    pub fn cancel(&self, request_id: RequestId) {
        let entry = self.cancel_flags.lock().unwrap().remove(&request_id.0);
        if let Some((flag, listener)) = entry {
            flag.store(true, Ordering::Relaxed);
            listener.on_event(crate::events::Event::Failed {
                request_id: request_id.0.clone(),
                error: "cancelled".to_string(),
            });
        }
    }

    /// Return the current mesh connection status.
    pub async fn status(&self) -> Status {
        Status {
            connected: self.connected,
            peer_count: 0,
        }
    }

    pub async fn disconnect(&mut self) {
        self.user_disconnected = true;
        self.connected = false;
        self.emit_event(crate::events::Event::Disconnected {
            reason: "disconnect_requested".to_string(),
        });
    }

    pub async fn reconnect(&mut self) -> Result<(), ClientError> {
        self.user_disconnected = false;
        self.reconnect_attempts = 0;
        self.connected = false;
        self.emit_event(crate::events::Event::Disconnected {
            reason: "reconnect_requested".to_string(),
        });
        self.join().await
    }

    fn emit_event(&self, event: crate::events::Event) {
        for listener in self.listeners.lock().unwrap().iter() {
            listener.on_event(event.clone());
        }
    }
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ResponsesRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
}

pub struct Status {
    pub connected: bool,
    pub peer_count: usize,
}

pub struct RequestId(pub String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}
