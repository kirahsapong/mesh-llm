#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP_DIR="$(mktemp -d)"
OUT_DIR="$TMP_DIR/out"
RUNNER_DIR="$TMP_DIR/runner"
SWIFT_SOURCE_DIR="$REPO_ROOT/bindings/swift/Sources/MeshLLM/Generated"
FFI_DIR="$REPO_ROOT/bindings/swift/Generated/FFI"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$RUNNER_DIR/src" "$OUT_DIR" "$SWIFT_SOURCE_DIR" "$FFI_DIR"

cat > "$RUNNER_DIR/Cargo.toml" <<'EOF'
[package]
name = "swift_bindgen_runner"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
camino = "1"
uniffi_bindgen = "=0.31.0"
EOF

cat > "$RUNNER_DIR/src/main.rs" <<EOF
use anyhow::Result;
use camino::Utf8PathBuf;
use uniffi_bindgen::bindings::{generate_swift_bindings, SwiftBindingsOptions};

fn main() -> Result<()> {
    let out_dir = Utf8PathBuf::from("$OUT_DIR");
    std::fs::create_dir_all(out_dir.as_std_path())?;
    generate_swift_bindings(SwiftBindingsOptions {
        generate_swift_sources: true,
        generate_headers: true,
        generate_modulemap: true,
        source: Utf8PathBuf::from("$REPO_ROOT/mesh-api-ffi/src/mesh_ffi.udl"),
        out_dir,
        xcframework: false,
        module_name: None,
        modulemap_filename: None,
        metadata_no_deps: true,
        link_frameworks: Vec::new(),
    })?;
    Ok(())
}
EOF

cargo run --manifest-path "$RUNNER_DIR/Cargo.toml"

cp "$OUT_DIR/mesh_ffi.swift" "$SWIFT_SOURCE_DIR/mesh_ffi.swift"
cp "$OUT_DIR/mesh_ffiFFI.h" "$FFI_DIR/mesh_ffiFFI.h"
cat > "$FFI_DIR/mesh_ffiFFI.modulemap" <<'EOF'
framework module mesh_ffiFFI {
  header "mesh_ffiFFI.h"
  export *
  use "Darwin"
  use "_Builtin_stdbool"
  use "_Builtin_stdint"
}
EOF
