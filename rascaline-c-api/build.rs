fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let result = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config {
            language: cbindgen::Language::C,
            cpp_compat: true,
            include_guard: Some("RASCALINE_H".into()),
            include_version: false,
            documentation: true,
            documentation_style: cbindgen::DocumentationStyle::Doxy,
            ..Default::default()
        })
        .generate()
        .map(|data| {
            data.write_to_file("rascaline.h");
        });

    // if not ok, rerun the build script unconditionally
    if result.is_ok() {
        for entry in glob::glob("src/**/*.rs").unwrap() {
            if let Ok(path) = entry {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
