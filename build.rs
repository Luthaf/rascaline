fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config {
            language: cbindgen::Language::C,
            cpp_compat: true,
            include_guard: Some("RASCALINE_H".into()),
            include_version: true,
            documentation: true,
            documentation_style: cbindgen::DocumentationStyle::C,
            ..Default::default()
        })
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/rascaline.h");
}
