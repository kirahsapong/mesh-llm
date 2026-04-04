use super::super::{
    http::{respond_error, respond_json},
    MeshApi,
};
use crate::plugin::stapler;
use serde_json::{Map, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use url::form_urlencoded;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    match (method, path_only) {
        ("GET", "/api/plugins") => handle_list(stream, state).await,
        ("GET", "/api/plugins/endpoints") => handle_endpoints(stream, state).await,
        ("GET", "/api/plugins/providers") => handle_providers(stream, state).await,
        ("GET", p) if p.starts_with("/api/plugins/providers/") => {
            handle_provider(stream, state, p).await
        }
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/manifest") => {
            handle_manifest(stream, state, p).await
        }
        ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
            handle_tools(stream, state, p).await
        }
        ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
            handle_call(stream, state, p, body).await
        }
        (m, p)
            if p.starts_with("/api/plugins/")
                && matches!(m, "GET" | "POST" | "PUT" | "PATCH" | "DELETE") =>
        {
            handle_stapled_http(stream, state, method, path, path_only, body, raw_request).await
        }
        _ => Ok(()),
    }
}

async fn handle_list(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let plugins = plugin_manager.list().await;
    let json = serde_json::to_string(&plugins)?;
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    Ok(())
}

async fn handle_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.endpoints().await {
        Ok(endpoints) => respond_json(stream, 200, &endpoints).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_providers(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.capability_providers().await {
        Ok(providers) => respond_json(stream, 200, &providers).await?,
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_provider(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let capability = &path["/api/plugins/providers/".len()..];
    let capability = urlencoding::decode(capability)
        .map(|value| value.into_owned())
        .unwrap_or_else(|_| capability.to_string());
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.provider_for_capability(&capability).await {
        Ok(Some(provider)) => respond_json(stream, 200, &provider).await?,
        Ok(None) => {
            respond_error(
                stream,
                404,
                &format!("No provider for capability '{}'", capability),
            )
            .await?
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await?,
    }
    Ok(())
}

async fn handle_tools(stream: &mut TcpStream, state: &MeshApi, path: &str) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/tools");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.tools(plugin_name).await {
        Ok(tools) => {
            let json = serde_json::to_string(&tools)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Err(e) => {
            respond_error(stream, 404, &e.to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_manifest(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    let plugin_name = rest.trim_end_matches("/manifest");
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.manifest_json(plugin_name).await {
        Ok(Some(manifest)) => {
            let json = serde_json::to_string(&manifest)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
        }
        Err(e) => {
            respond_error(stream, 404, &e.to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_call(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let rest = &path["/api/plugins/".len()..];
    if let Some((plugin_name, tool_name)) = rest.split_once("/tools/") {
        let payload = if body.trim().is_empty() { "{}" } else { body };
        let plugin_manager = state.inner.lock().await.plugin_manager.clone();
        match plugin_manager
            .call_tool(plugin_name, tool_name, payload)
            .await
        {
            Ok(result) if !result.is_error => {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    result.content_json.len(),
                    result.content_json
                );
                stream.write_all(resp.as_bytes()).await?;
            }
            Ok(result) => {
                respond_error(stream, 502, &result.content_json).await?;
            }
            Err(e) => {
                respond_error(stream, 502, &e.to_string()).await?;
            }
        }
    } else {
        respond_error(stream, 404, "Not found").await?;
    }
    Ok(())
}

async fn handle_stapled_http(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let Some((plugin_name, route_path)) = parse_stapled_http_path(path_only) else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };

    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let manifest = match plugin_manager.manifest(plugin_name).await {
        Ok(Some(manifest)) => manifest,
        Ok(None) => {
            respond_error(stream, 404, "Plugin did not publish a manifest").await?;
            return Ok(());
        }
        Err(err) => {
            respond_error(stream, 404, &err.to_string()).await?;
            return Ok(());
        }
    };

    let Some(binding) = manifest.http_bindings.iter().find(|binding| {
        stapler::http_binding_route(plugin_name, binding)
            .map(|route| route.method == method && route.route_path == route_path)
            .unwrap_or(false)
    }) else {
        respond_error(stream, 404, "No matching plugin HTTP binding").await?;
        return Ok(());
    };

    let request_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.request_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    let response_streamed = matches!(
        crate::plugin::proto::HttpBodyMode::try_from(binding.response_body_mode)
            .unwrap_or(crate::plugin::proto::HttpBodyMode::Unspecified),
        crate::plugin::proto::HttpBodyMode::Streamed
    );
    if request_streamed || response_streamed {
        return handle_streamed_http_binding(
            stream,
            &plugin_manager,
            plugin_name,
            binding,
            raw_request,
        )
        .await;
    }

    let Some(operation_name) = binding.operation_name.as_deref() else {
        respond_error(
            stream,
            501,
            "HTTP binding does not declare an operation_name yet",
        )
        .await?;
        return Ok(());
    };

    let args = match build_http_arguments(path, body) {
        Ok(args) => args,
        Err(err) => {
            respond_error(stream, 400, &err).await?;
            return Ok(());
        }
    };

    match plugin_manager
        .call_tool(
            plugin_name,
            operation_name,
            &Value::Object(args).to_string(),
        )
        .await
    {
        Ok(result) if !result.is_error => match serde_json::from_str::<Value>(&result.content_json)
        {
            Ok(value) => respond_json(stream, 200, &value).await?,
            Err(_) => {
                respond_error(
                    stream,
                    502,
                    "Plugin returned a non-JSON response for a buffered HTTP binding",
                )
                .await?;
            }
        },
        Ok(result) => {
            respond_error(stream, 502, &result.content_json).await?;
        }
        Err(err) => {
            respond_error(stream, 502, &err.to_string()).await?;
        }
    }

    Ok(())
}

async fn handle_streamed_http_binding(
    client_stream: &mut TcpStream,
    plugin_manager: &crate::plugin::PluginManager,
    plugin_name: &str,
    binding: &crate::plugin::proto::HttpBindingManifest,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let stream_id = format!("http-{}-{}", std::process::id(), rand::random::<u64>());
    let request = crate::plugin::proto::OpenStreamRequest {
        stream_id,
        purpose: crate::plugin::proto::StreamPurpose::Generic as i32,
        mode: crate::plugin::proto::StreamMode::Http1 as i32,
        bidirectional: true,
        content_type: Some("application/http".into()),
        correlation_id: None,
        metadata_json: Some(
            serde_json::json!({
                "binding_id": binding.binding_id,
                "method": method_name(binding.method),
                "path": binding.path,
            })
            .to_string(),
        ),
        expected_bytes: Some(raw_request.len() as u64),
        idle_timeout_ms: Some(30_000),
    };
    let mut plugin_stream = plugin_manager.connect_stream(plugin_name, request).await?;
    let forwarded_request = rewrite_http_request_path(raw_request, &binding.path)?;
    plugin_stream.write_all(&forwarded_request).await?;
    plugin_stream.shutdown().await?;

    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = plugin_stream.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        client_stream.write_all(&buf[..read]).await?;
    }
    Ok(())
}

fn parse_stapled_http_path(path_only: &str) -> Option<(&str, &str)> {
    let rest = path_only.strip_prefix("/api/plugins/")?;
    let (plugin_name, remainder) = rest.split_once("/http")?;
    if plugin_name.is_empty() || remainder.is_empty() {
        return None;
    }
    Some((
        plugin_name,
        &path_only[.."/api/plugins/".len() + plugin_name.len() + "/http".len() + remainder.len()],
    ))
}

fn build_http_arguments(path: &str, body: &str) -> Result<Map<String, Value>, String> {
    let mut args = query_arguments(path);
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(args);
    }
    let body_value: Value =
        serde_json::from_str(trimmed).map_err(|err| format!("Invalid JSON body: {err}"))?;
    let Value::Object(body_map) = body_value else {
        return Err("Buffered plugin HTTP bindings currently require a JSON object body".into());
    };
    args.extend(body_map);
    Ok(args)
}

fn rewrite_http_request_path(raw_request: &[u8], path: &str) -> anyhow::Result<Vec<u8>> {
    let header_end = raw_request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .ok_or_else(|| anyhow::anyhow!("HTTP request is missing a header terminator"))?;
    let mut headers_buf = [httparse::EMPTY_HEADER; 64];
    let mut req = httparse::Request::new(&mut headers_buf);
    req.parse(raw_request)
        .map_err(|err| anyhow::anyhow!("HTTP parse error while rewriting request path: {err}"))?;

    let method = req.method.unwrap_or("GET");
    let version = req.version.unwrap_or(1);
    let mut rebuilt = format!(
        "{method} {} HTTP/1.{version}\r\n",
        normalized_http_path(path)
    );

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(&raw_request[header_end..]);
    Ok(forwarded)
}

fn normalized_http_path(path: &str) -> &str {
    if path.is_empty() {
        "/"
    } else {
        path
    }
}

fn method_name(value: i32) -> &'static str {
    match crate::plugin::proto::HttpMethod::try_from(value)
        .unwrap_or(crate::plugin::proto::HttpMethod::Unspecified)
    {
        crate::plugin::proto::HttpMethod::Get => "GET",
        crate::plugin::proto::HttpMethod::Post => "POST",
        crate::plugin::proto::HttpMethod::Put => "PUT",
        crate::plugin::proto::HttpMethod::Patch => "PATCH",
        crate::plugin::proto::HttpMethod::Delete => "DELETE",
        crate::plugin::proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn query_arguments(path: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let Some((_, query)) = path.split_once('?') else {
        return args;
    };
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        args.insert(key.into_owned(), Value::String(value.into_owned()));
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stapled_http_path() {
        let parsed = parse_stapled_http_path("/api/plugins/demo/http/feed").unwrap();
        assert_eq!(parsed.0, "demo");
        assert_eq!(parsed.1, "/api/plugins/demo/http/feed");
    }

    #[test]
    fn query_arguments_decode_values() {
        let args = query_arguments("/api/plugins/demo/http/feed?name=hello%20world&limit=10");
        assert_eq!(args.get("name"), Some(&Value::String("hello world".into())));
        assert_eq!(args.get("limit"), Some(&Value::String("10".into())));
    }

    #[test]
    fn build_http_arguments_merges_query_and_body() {
        let args = build_http_arguments(
            "/api/plugins/demo/http/feed?from=alice",
            r#"{"limit":10,"from":"bob"}"#,
        )
        .unwrap();
        assert_eq!(args.get("limit"), Some(&Value::Number(10.into())));
        assert_eq!(args.get("from"), Some(&Value::String("bob".into())));
    }

    #[test]
    fn rewrite_http_request_path_updates_request_line_only() {
        let raw = b"POST /api/plugins/demo/http/feed?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\nConnection: keep-alive\r\n\r\n{\"a\":1}";
        let rewritten = rewrite_http_request_path(raw, "/feed").unwrap();
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.starts_with("POST /feed HTTP/1.1\r\n"));
        assert!(text.contains("Host: localhost\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"));
    }
}
