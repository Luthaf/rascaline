use std::path::PathBuf;

mod utils;

#[test]
fn check_cxx_install() {
    if cfg!(tarpaulin) {
        // do not run this test when collecting Rust coverage
        return;
    }

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");

    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir).unwrap();
    }

    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // build and install rascaline with cmake
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &build_dir);

    let mut install_dir = build_dir.clone();
    install_dir.push("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()));
    cmake_config.arg("-DRASCALINE_FETCH_EQUISTORE=ON");

    let status = cmake_config.status().expect("cmake configuration failed");
    assert!(status.success());

    let mut cmake_build = utils::cmake_build(&build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("cmake build failed");
    assert!(status.success());

    // try to use the installed rascaline from cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-sample-project");
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir).unwrap();
    }
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", install_dir.display()));

    let status = cmake_config.status().expect("cmake configuration failed");
    assert!(status.success());

    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("cmake build failed");
    assert!(status.success());
}
