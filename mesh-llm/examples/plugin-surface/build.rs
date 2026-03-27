use std::path::Path;

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);

    let proto_dir = Path::new("../../proto");
    let proto_file = proto_dir.join("plugin.proto");
    println!("cargo:rerun-if-changed={}", proto_file.display());

    prost_build::Config::new()
        .compile_protos(&[proto_file], &[proto_dir])
        .expect("compile plugin proto");
}
