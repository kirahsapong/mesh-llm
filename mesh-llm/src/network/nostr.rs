pub use mesh_client::network::nostr::{
    auto_model_pack, default_models_for_vram, discover, score_mesh, smart_auto, AutoDecision,
    DiscoveryClient, MeshFilter, MeshListing, Publisher, DEFAULT_RELAYS,
};

use anyhow::Result;
use nostr_sdk::prelude::*;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Keys — stored in ~/.mesh-llm/nostr.nsec
// ---------------------------------------------------------------------------

fn nostr_key_path() -> Result<std::path::PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".mesh-llm").join("nostr.nsec"))
}

pub fn load_or_create_keys() -> Result<Keys> {
    load_or_create_keys_at(&nostr_key_path()?)
}

fn load_or_create_keys_at(path: &std::path::Path) -> Result<Keys> {
    if let Some(parent) = path.parent() {
        ensure_private_nostr_dir(parent)?;
    }

    if path.exists() {
        ensure_private_nostr_key_file(path)?;
        let nsec = std::fs::read_to_string(&path)?;
        let sk = SecretKey::from_bech32(nsec.trim())?;
        Ok(Keys::new(sk))
    } else {
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32()?;
        crate::crypto::write_keystore_bytes_atomically(path, nsec.as_bytes())?;
        tracing::info!("Generated new Nostr key, saved to {}", path.display());
        Ok(keys)
    }
}

#[cfg(unix)]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(dir)?;
    let metadata = std::fs::metadata(dir)?;
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o700);
        std::fs::set_permissions(dir, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

#[cfg(unix)]
fn ensure_private_nostr_key_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        anyhow::bail!("Nostr key path is not a regular file");
    }
    let mut perms = metadata.permissions();
    if perms.mode() & 0o077 != 0 {
        perms.set_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_nostr_key_file(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

/// Delete the Nostr key and node identity key.  After rotation the
/// node gets a fresh identity on next start.
pub fn rotate_keys() -> Result<()> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    let mesh_dir = home.join(".mesh-llm");

    let nostr_path = nostr_key_path()?;
    if nostr_path.exists() {
        std::fs::remove_file(&nostr_path)?;
        eprintln!("🔑 Deleted {}", nostr_path.display());
    } else {
        eprintln!("No Nostr key to rotate (none exists yet).");
    }

    let node_key_path = mesh_dir.join("key");
    if node_key_path.exists() {
        std::fs::remove_file(&node_key_path)?;
        eprintln!("🔑 Deleted {}", node_key_path.display());
    } else {
        eprintln!("No node key to rotate (none exists yet).");
    }

    eprintln!();
    eprintln!("✅ Keys rotated. New identities will be generated on next start.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Publisher — background task that keeps the listing fresh
// ---------------------------------------------------------------------------

pub async fn publish_loop(
    node: crate::mesh::Node,
    keys: Keys,
    relays: Vec<String>,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
    interval_secs: u64,
) {
    let publisher = match Publisher::new(keys.clone(), &relays).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create Nostr publisher: {e}");
            return;
        }
    };

    let npub = publisher.npub();
    if let Some(cap) = max_clients {
        eprintln!("   Will delist when {} clients connected", cap);
    }

    for _ in 0..120 {
        if node.is_llama_ready().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    eprintln!(
        "📡 Publishing mesh to Nostr (npub: {}...{})",
        &npub[..12],
        &npub[npub.len() - 8..]
    );

    let mut delisted = false;

    let disco = DiscoveryClient::new(Keys::generate(), &relays).await.ok();

    loop {
        let invite_token = node.invite_token();
        let peers = node.peers().await;

        let client_count = peers
            .iter()
            .filter(|p| matches!(p.role, crate::mesh::NodeRole::Client))
            .count();

        if let Some(cap) = max_clients {
            if client_count >= cap && !delisted {
                if let Err(e) = publisher.unpublish().await {
                    tracing::warn!("Failed to unpublish from Nostr: {e}");
                }
                eprintln!(
                    "📡 Delisted from Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = true;
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                continue;
            } else if client_count < cap && delisted {
                eprintln!(
                    "📡 Re-publishing to Nostr ({} clients, cap is {})",
                    client_count, cap
                );
                delisted = false;
            }
        }

        if delisted {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            continue;
        }

        let gpu_peers = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count();
        let my_node_count = gpu_peers + 1;
        if gpu_peers == 0 {
            let filter = MeshFilter::default();
            if let Ok(listings) = discover(&relays, &filter, disco.as_ref()).await {
                let my_npub = publisher.npub();
                let my_mesh_id = node.mesh_id().await;

                let split_target = my_mesh_id.as_ref().and_then(|mid| {
                    listings.iter().find(|m| {
                        m.listing.mesh_id.as_deref() == Some(mid.as_str())
                            && m.publisher_npub != my_npub
                            && m.listing.node_count > my_node_count
                    })
                });

                let merge_target = if split_target.is_none() && name.is_none() {
                    listings.iter().find(|m| {
                        m.publisher_npub != my_npub
                            && m.listing.name.is_none()
                            && m.listing.node_count > my_node_count
                    })
                } else {
                    None
                };

                if let Some(target) = split_target.or(merge_target) {
                    eprintln!(
                        "📡 Found larger mesh '{}' ({} nodes vs our {}) — rejoining",
                        target.listing.name.as_deref().unwrap_or("unnamed"),
                        target.listing.node_count,
                        my_node_count
                    );
                    if let Err(e) = publisher.unpublish().await {
                        tracing::warn!("Failed to unpublish solo listing: {e}");
                    }
                    if let Err(e) = node.join(&target.listing.invite_token).await {
                        tracing::warn!("Merge/rejoin failed: {e}");
                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                    eprintln!("📡 Merged into mesh — resuming publish as member");
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    continue;
                }
            }
        }

        let my_role = node.role().await;
        let mut actually_serving: Vec<String> = Vec::new();
        if matches!(my_role, crate::mesh::NodeRole::Host { .. }) {
            for model in node.hosted_models().await {
                if !actually_serving.contains(&model) {
                    actually_serving.push(model);
                }
            }
        }
        for p in &peers {
            if matches!(p.role, crate::mesh::NodeRole::Host { .. }) {
                for model in p.routable_models() {
                    if !actually_serving.contains(&model) {
                        actually_serving.push(model);
                    }
                }
            }
        }

        let served_set: std::collections::HashSet<&str> =
            actually_serving.iter().map(|s| s.as_str()).collect();

        let active_demand = node.active_demand().await;
        let mut wanted: Vec<String> = Vec::new();
        for m in active_demand.keys() {
            if !served_set.contains(m.as_str()) && !wanted.contains(m) {
                wanted.push(m.clone());
            }
        }

        let mut available: Vec<String> = Vec::new();
        let my_available = node.available_models().await;
        for m in &my_available {
            if !served_set.contains(m.as_str()) && !available.contains(m) {
                available.push(m.clone());
            }
        }
        for p in &peers {
            for m in &p.available_models {
                if !served_set.contains(m.as_str()) && !available.contains(m) {
                    available.push(m.clone());
                }
            }
        }

        let total_vram: u64 = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .map(|p| p.vram_bytes)
            .sum::<u64>()
            + node.vram_bytes();

        let node_count = peers
            .iter()
            .filter(|p| !matches!(p.role, crate::mesh::NodeRole::Client))
            .count()
            + 1;

        let mesh_id = node.mesh_id().await;

        let listing = MeshListing {
            invite_token,
            serving: actually_serving,
            wanted,
            on_disk: available,
            total_vram_bytes: total_vram,
            node_count,
            client_count,
            max_clients: max_clients.unwrap_or(0),
            name: name.clone(),
            region: region.clone(),
            mesh_id,
        };

        let ttl = interval_secs * 2;
        match publisher.publish(&listing, ttl).await {
            Ok(()) => tracing::debug!(
                "Published mesh listing ({} models, {} nodes, {} clients)",
                listing.serving.len(),
                listing.node_count,
                client_count
            ),
            Err(e) => tracing::warn!("Failed to publish to Nostr: {e}"),
        }

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

// ---------------------------------------------------------------------------
// Publish watchdog — take over publishing if the original publisher dies
// ---------------------------------------------------------------------------

pub async fn publish_watchdog(
    node: crate::mesh::Node,
    relays: Vec<String>,
    mesh_name: Option<String>,
    region: Option<String>,
    check_interval_secs: u64,
) {
    let jitter = (rand::random::<u64>() % 20) + 10;
    tokio::time::sleep(Duration::from_secs(jitter)).await;

    let disco = DiscoveryClient::new(Keys::generate(), &relays).await.ok();

    loop {
        let filter = MeshFilter::default();
        match discover(&relays, &filter, disco.as_ref()).await {
            Ok(meshes) => {
                let our_peers = node.peers().await;
                let served = node.models_being_served().await;
                let our_mesh_id = node.mesh_id().await;

                let mesh_listed = if let Some(ref mid) = our_mesh_id {
                    meshes
                        .iter()
                        .any(|m| m.listing.mesh_id.as_deref() == Some(mid.as_str()))
                } else if !served.is_empty() {
                    meshes
                        .iter()
                        .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                } else {
                    false
                };

                if !mesh_listed && (!our_peers.is_empty() || !served.is_empty()) {
                    let backoff = (rand::random::<u64>() % 7) + 3;
                    eprintln!("📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over...");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;

                    if let Ok(recheck) = discover(&relays, &filter, disco.as_ref()).await {
                        let still_missing = if let Some(ref mid) = our_mesh_id {
                            !recheck
                                .iter()
                                .any(|m| m.listing.mesh_id.as_deref() == Some(mid.as_str()))
                        } else if !served.is_empty() {
                            !recheck
                                .iter()
                                .any(|m| served.iter().any(|s| m.listing.serving.contains(s)))
                        } else {
                            true
                        };
                        if !still_missing {
                            eprintln!("📡 Someone else took over publishing — standing down");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    }

                    eprintln!("📡 Taking over Nostr publishing for the mesh");
                    let keys = match load_or_create_keys() {
                        Ok(k) => k,
                        Err(e) => {
                            tracing::warn!("Failed to load Nostr keys for publish takeover: {e}");
                            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                            continue;
                        }
                    };
                    publish_loop(node, keys, relays, mesh_name, region, None, 60).await;
                    return;
                }
            }
            Err(e) => {
                tracing::debug!("Publish watchdog: Nostr check failed: {e}");
            }
        }

        let next_check = (rand::random::<u64>() % 15) + 20;
        tokio::time::sleep(Duration::from_secs(next_check)).await;
    }
}

#[cfg(test)]
mod rotate_key_tests {
    use super::*;
    use serial_test::serial;
    use std::fs;

    #[test]
    #[serial]
    fn rotate_deletes_both_keys_and_handles_missing() {
        let dir = dirs::home_dir().unwrap().join(".mesh-llm");
        fs::create_dir_all(&dir).ok();

        let key_path = dir.join("key");
        let nsec_path = dir.join("nostr.nsec");

        let orig_key = if key_path.exists() {
            Some(fs::read(&key_path).unwrap())
        } else {
            None
        };
        let orig_nsec = if nsec_path.exists() {
            Some(fs::read(&nsec_path).unwrap())
        } else {
            None
        };

        fs::write(&key_path, b"test-node-key").unwrap();
        fs::write(&nsec_path, b"test-nostr-nsec").unwrap();

        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed when keys exist");
        assert!(!key_path.exists(), "node key should be deleted");
        assert!(!nsec_path.exists(), "nostr key should be deleted");

        let result = rotate_keys();
        assert!(result.is_ok(), "rotate should succeed even with no keys");

        if let Some(k) = orig_key {
            fs::write(&key_path, k).ok();
        }
        if let Some(n) = orig_nsec {
            fs::write(&nsec_path, n).ok();
        }
    }
}

#[cfg(test)]
mod key_file_tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_key_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("{prefix}-{unique}"))
            .join("nostr.nsec")
    }

    #[test]
    fn load_or_create_keys_at_round_trips() {
        let path = temp_key_path("mesh-llm-nostr-key");
        let first = load_or_create_keys_at(&path).unwrap();
        let second = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            first.secret_key().to_bech32().unwrap(),
            second.secret_key().to_bech32().unwrap()
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_hardens_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-perms");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&path, &nsec).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        let loaded = load_or_create_keys_at(&path).unwrap();
        assert_eq!(
            loaded.secret_key().to_bech32().unwrap(),
            keys.secret_key().to_bech32().unwrap()
        );
        assert_eq!(
            std::fs::metadata(dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn load_or_create_keys_at_rejects_symlink_key() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_key_path("mesh-llm-nostr-key-symlink");
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let real_file = dir.join("nostr.real");
        let keys = Keys::generate();
        let nsec = keys.secret_key().to_bech32().unwrap();
        std::fs::write(&real_file, &nsec).unwrap();
        std::fs::set_permissions(&real_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&real_file, &path).unwrap();

        let result = load_or_create_keys_at(&path);
        assert!(result.is_err(), "expected error for symlinked nostr key");

        let _ = std::fs::remove_dir_all(dir);
    }
}

// ---------------------------------------------------------------------------
// Integration test — publish/discover against real Nostr relays
// ---------------------------------------------------------------------------
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn publish_discover_round_trip() {
        let relays: Vec<String> = DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect();
        let mesh_name = format!("mesh-llm-test-{}", rand::random::<u32>());
        let mesh_id = format!("test-id-{}", rand::random::<u32>());

        let keys_a = Keys::generate();
        let pub_a = Publisher::new(keys_a.clone(), &relays)
            .await
            .expect("pub_a");
        let listing_a = MeshListing {
            invite_token: "invite-a".into(),
            serving: vec!["Qwen3-8B-Q4_K_M".into()],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 16_000_000_000,
            node_count: 2,
            client_count: 0,
            max_clients: 0,
            name: Some(mesh_name.clone()),
            region: Some("test-region".into()),
            mesh_id: Some(mesh_id.clone()),
        };
        pub_a.publish(&listing_a, 120).await.expect("publish A");

        let keys_b = Keys::generate();
        let pub_b = Publisher::new(keys_b.clone(), &relays)
            .await
            .expect("pub_b");
        let mut listing_b = listing_a.clone();
        listing_b.invite_token = "invite-b".into();
        pub_b.publish(&listing_b, 120).await.expect("publish B");

        tokio::time::sleep(Duration::from_secs(3)).await;

        let dc = DiscoveryClient::new(Keys::generate(), &relays)
            .await
            .expect("dc");
        let meshes = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("discover");

        let found: Vec<_> = meshes
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(
            found.len() >= 2,
            "should find both publishers for mesh_id={mesh_id}, found {}",
            found.len()
        );

        let m = &found[0];
        assert_eq!(m.listing.name.as_deref(), Some(mesh_name.as_str()));
        assert_eq!(m.listing.serving, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(m.listing.node_count, 2);
        assert_eq!(m.listing.total_vram_bytes, 16_000_000_000);

        let tokens: Vec<_> = found
            .iter()
            .map(|m| m.listing.invite_token.as_str())
            .collect();
        assert!(
            tokens.contains(&"invite-a"),
            "missing invite-a in {tokens:?}"
        );
        assert!(
            tokens.contains(&"invite-b"),
            "missing invite-b in {tokens:?}"
        );

        let r2 = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("second discover");
        assert!(
            r2.iter()
                .any(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str())),
            "second discover should still find the mesh"
        );

        pub_a.unpublish().await.ok();
        pub_b.unpublish().await.ok();
    }
}
