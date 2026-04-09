mod api;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod protocol; // portable wire types extracted to mesh_client_core::protocol (phase-2/wave-2)
pub(crate) mod runtime;
mod system;

pub use mesh_client_core::proto;

pub(crate) use plugins::blackboard;

use anyhow::Result;

pub const VERSION: &str = "0.58.0";

pub async fn run() -> Result<()> {
    runtime::run().await
}
