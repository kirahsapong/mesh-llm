pub use mesh_client::models::capabilities::{
    merge_config_signals, merge_name_signals, merge_sibling_signals, CapabilityLevel,
    ModelCapabilities,
};

use super::build_hf_tokio_api;
use super::catalog;
use hf_hub::{Repo, RepoType};
use serde_json::Value;
use std::path::Path;

pub fn infer_catalog_capabilities(model: &catalog::CatalogModel) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    if model.mmproj.is_some() {
        caps.vision = CapabilityLevel::Supported;
        caps.multimodal = true;
    }
    caps.moe = model.moe.is_some();
    caps = merge_name_signals(
        caps,
        &[
            model.name.as_str(),
            model.file.as_str(),
            model.description.as_str(),
        ],
    );
    caps.normalize()
}

pub fn infer_local_model_capabilities(
    model_name: &str,
    path: &Path,
    catalog_entry: Option<&catalog::CatalogModel>,
) -> ModelCapabilities {
    let mut caps = catalog_entry
        .map(infer_catalog_capabilities)
        .unwrap_or_default();
    caps = merge_name_signals(
        caps,
        &[
            model_name,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        ],
    );
    for config in read_local_metadata_jsons(path) {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

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

fn read_local_metadata_jsons(path: &Path) -> Vec<Value> {
    let mut values = Vec::new();
    for dir in path.ancestors().skip(1).take(6) {
        for name in ["config.json", "tokenizer_config.json", "chat_template.json"] {
            let candidate = dir.join(name);
            if !candidate.is_file() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            if let Ok(value) = serde_json::from_str(&text) {
                values.push(value);
            }
        }
    }
    values
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
