fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config {
            language: cbindgen::Language::C,
            cpp_compat: true,
            include_guard: Some("RASCALINE_H".into()),
            include_version: false,
            documentation: true,
            documentation_style: cbindgen::DocumentationStyle::C,
            ..Default::default()
        })
        .rename_item("Pair", "rascal_pair_t")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/rascaline.h");

    for entry in glob::glob("src/**/*.rs").unwrap() {
        if let Ok(path) = entry {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
