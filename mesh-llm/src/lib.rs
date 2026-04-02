mod affinity;
mod api;
mod autoupdate;
mod benchmark;
mod cli;
mod hardware;
mod inference;
mod mesh;
mod models;
mod nostr;
mod plugin;
mod plugin_mcp;
mod plugins;
mod protocol;
mod proxy;
mod rewrite;
mod router;
pub(crate) mod runtime;
mod tunnel;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use autoupdate::{latest_release_version, version_newer};
pub(crate) use inference::{election, launch, moe, pipeline};
pub use plugins::blackboard;
pub use plugins::blackboard::mcp as blackboard_mcp;

use anyhow::Result;

pub const VERSION: &str = "0.54.0";

pub async fn run() -> Result<()> {
    runtime::run().await
}
