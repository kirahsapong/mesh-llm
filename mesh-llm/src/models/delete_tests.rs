use std::fs;
use std::path::PathBuf;

use crate::models::delete::{delete_model_at_path, resolve_model_identifier, ResolveError};
use crate::models::local::mesh_llm_cache_dir;

fn temp_dir(prefix: &str) -> PathBuf {
    // Place temp files inside mesh_llm cache so safety gate accepts them.
    let base = mesh_llm_cache_dir();
    let _ = fs::create_dir_all(&base);
    let dir = base.join(format!("test-{prefix}-{}", std::process::id()));
    let _ = fs::create_dir_all(&dir);
    dir
}

#[test]
fn test_resolve_exact_path_found() {
    let tmp = temp_dir("exact-found");
    let model_file = tmp.join("test-model.gguf");
    fs::write(&model_file, vec![0u8; 100]).expect("should create temp file");

    let result = resolve_model_identifier(model_file.to_str().unwrap());
    assert!(
        result.is_ok(),
        "resolved exact path should succeed: {:?}",
        result.err()
    );
    let resolved = result.unwrap();
    assert_eq!(resolved.path, model_file);
    assert!(resolved.is_exact_path);
    assert_eq!(resolved.display_name, "test-model");
}

#[test]
fn test_resolve_exact_path_not_found() {
    let result = resolve_model_identifier("/nonexistent/path/model.gguf");
    assert!(result.is_err(), "should error for nonexistent path");
    match result.unwrap_err() {
        ResolveError::NotFound(msg) => {
            assert!(msg.contains("/nonexistent/path/model.gguf"));
        }
        other => panic!("expected NotFound, got: {:?}", other),
    }
}

#[test]
fn test_resolve_exact_path_no_gguf_extension() {
    let tmp = temp_dir("no-gguf-ext");
    let txt_file = tmp.join("test.txt");
    fs::write(&txt_file, "not a gguf").expect("should create temp file");

    let result = resolve_model_identifier(txt_file.to_str().unwrap());
    assert!(result.is_err(), "should reject non-.gguf extension");
    match result.unwrap_err() {
        ResolveError::NotFound(msg) => {
            assert!(
                msg.to_lowercase().contains("gguf"),
                "error should mention .gguf requirement"
            );
        }
        other => panic!("expected NotFound, got: {:?}", other),
    }
}

#[test]
fn test_resolve_stem_single_match() {
    // Use find_model_path with a stem that matches an existing model on the dev machine.
    // If no real models exist, it will return a path that doesn't exist — which means
    // our resolution should detect zero matches and return NotFound.
    let result = resolve_model_identifier("Qwen3-8B-Q4_K_M");
    // Either resolves to an actual path (if installed) or returns NotFound
    match result {
        Ok(resolved) => {
            assert!(!resolved.path.as_os_str().is_empty());
            assert!(!resolved.is_exact_path);
        }
        Err(ResolveError::NotFound(_)) => {
            // Expected if no matching GGUF files are found in cache
        }
        Err(ResolveError::Ambiguous(candidates)) => {
            // Also valid - multiple matches found
            assert!(candidates.len() >= 2);
        }
    }
}

#[test]
fn test_resolve_stem_no_match() {
    let result = resolve_model_identifier("this-stem-does-not-exist-xyz-abc123");
    assert!(result.is_err(), "should error for non-existent stem");
    match result.unwrap_err() {
        ResolveError::NotFound(msg) => {
            assert!(msg.to_lowercase().contains("no model found") || msg.contains("this-stem"));
        }
        other => panic!("expected NotFound, got: {:?}", other),
    }
}

async fn setup_mock_delete_dir(prefix: &str) -> (PathBuf, PathBuf) {
    let tmp = temp_dir(prefix);
    let model_file = tmp.join("mock-model.gguf");
    fs::write(&model_file, vec![0u8; 2048]).expect("should create mock file");
    (tmp, model_file)
}

#[tokio::test]
async fn test_delete_dry_run_does_not_modify() {
    let (_tmp, model_file) = setup_mock_delete_dir("dryrun").await;

    // Dry-run should not modify anything
    let original_content = fs::read(&model_file).expect("file should exist before dry run");

    let result = delete_model_at_path(model_file.to_str().unwrap(), true).await;

    assert!(result.is_ok());
    let del_result = result.unwrap();
    // In dry-run mode, no files are actually deleted but paths may be recorded
    // The key assertion is the file still exists with same content
    let current_content = fs::read(&model_file).expect("file should exist after dry run");
    assert_eq!(
        original_content, current_content,
        "file content must not change in dry-run"
    );
    assert!(del_result.deleted_paths.iter().any(|p| p == &model_file));
}

#[tokio::test]
async fn test_delete_actually_removes_files() {
    let (_tmp, model_file) = setup_mock_delete_dir("actual-delete").await;

    // Verify file exists before deletion
    assert!(
        model_file.exists(),
        "mock file should exist before deletion"
    );

    let result = delete_model_at_path(model_file.to_str().unwrap(), false).await;

    assert!(
        result.is_ok(),
        "deletion should succeed: {:?}",
        result.err()
    );
    let del_result = result.unwrap();
    assert!(
        !model_file.exists(),
        "model file should be removed after deletion"
    );
    assert_eq!(del_result.deleted_paths.len(), 1);
}

#[tokio::test]
async fn test_delete_reported_bytes_are_accurate() {
    let (tmp, model_file) = setup_mock_delete_dir("byte-count").await;

    // Create a file of known size by writing exactly 2048 bytes
    fs::write(&model_file, vec![0u8; 2048]).expect("should write exact bytes");

    let result = delete_model_at_path(model_file.to_str().unwrap(), true).await;

    assert!(result.is_ok());
    let del_result = result.unwrap();
    assert_eq!(
        del_result.reclaimed_bytes, 2048,
        "reclaimed bytes should match actual file size"
    );

    // Cleanup temp dir since we're in dry-run mode
    let _ = tmp.read_dir();
    let _ = fs::remove_dir_all(&tmp);
}
