use std::path::PathBuf;
use std::process::Command;

mod utils;

#[test]
fn check_c_api() {
    utils::check_command_exists("cmake");
    utils::check_command_exists("ctest");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut build_dir = PathBuf::from(&cargo_manifest_dir);
    build_dir.push("target");
    build_dir.push("rascaline-c-api-build");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = cargo_manifest_dir;
    source_dir.push("tests");
    source_dir.push("c-api");

    // assume that debug assertion means that we are building the code in
    // debug mode, even if that could be not true in some cases
    let build_type = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    let mut shared_lib = "ON";
    if let Ok(value) = std::env::var("RASCALINE_TEST_WITH_STATIC_LIB") {
        if value != "0" {
            shared_lib = "OFF";
        }
    }

    let mut cmake_config = Command::new("cmake");
    cmake_config.current_dir(&build_dir);
    cmake_config.arg(&source_dir);

    // the cargo executable currently running
    let cargo_exe = std::env::var("CARGO").expect("CARGO env var is not set");
    cmake_config.arg(format!("-DCARGO_EXE={}", cargo_exe));

    cmake_config.arg(format!("-DCMAKE_BUILD_TYPE={}", build_type));
    cmake_config.arg(format!("-DBUILD_SHARED_LIBS={}", shared_lib));

    let status = cmake_config.status().expect("failed to configure cmake");
    assert!(status.success());

    let mut cmake_build = Command::new("cmake");
    cmake_build.current_dir(&build_dir);
    cmake_build.arg("--build");
    cmake_build.arg(".");
    cmake_build.arg("--config");
    cmake_build.arg(build_type);
    let status = cmake_build.status().expect("failed to build C++ code");
    assert!(status.success());

    let mut ctest = Command::new("ctest");
    ctest.current_dir(&build_dir);
    ctest.arg("--output-on-failure");
    ctest.arg("--C");
    ctest.arg(build_type);
    let status = ctest.status().expect("failed to run tests");
    assert!(status.success());
}
