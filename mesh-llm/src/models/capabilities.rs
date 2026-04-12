pub use mesh_client::models::capabilities::*;

use super::build_hf_tokio_api;
use hf_hub::{Repo, RepoType};
use serde_json::Value;

pub async fn infer_remote_hf_capabilities(
    repo: &str,
    revision: Option<&str>,
    file: &str,
    siblings: Option<&[String]>,
) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    caps = merge_name_signals(caps, &[repo, file]);
    if let Some(files) = siblings {
        caps = merge_sibling_signals(caps, files.iter().map(String::as_str));
    }
    for config in fetch_remote_metadata_jsons(repo, revision).await {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

async fn fetch_remote_metadata_jsons(repo: &str, revision: Option<&str>) -> Vec<Value> {
    let mut values = Vec::new();
    for filename in ["config.json", "tokenizer_config.json", "chat_template.json"] {
        if let Some(value) = fetch_remote_json(repo, revision, filename).await {
            values.push(value);
        }
    }
    values
}

async fn fetch_remote_json(repo: &str, revision: Option<&str>, file: &str) -> Option<Value> {
    let api = build_hf_tokio_api(false).ok()?;
    let repo = match revision {
        Some(revision) => {
            Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string())
        }
        None => Repo::new(repo.to_string(), RepoType::Model),
    };
    let path = api.repo(repo).get(file).await.ok()?;
    let text = tokio::fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&text).ok()
}
