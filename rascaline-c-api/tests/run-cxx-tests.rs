use std::path::PathBuf;

mod utils;

#[test]
fn run_cxx_tests() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-tests");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    source_dir.extend(["tests"]);

    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    cmake_config.arg("-DRASCALINE_FETCH_EQUISTORE=ON");
    let mut shared_lib = "ON";
    if let Ok(value) = std::env::var("RASCALINE_TEST_WITH_STATIC_LIB") {
        if value != "0" {
            shared_lib = "OFF";
        }
    }
    cmake_config.arg(format!("-DBUILD_SHARED_LIBS={}", shared_lib));

    // LLVM_PROFILE_FILE is set by cargo tarpaulin, so when it is set we also
    // collect code coverage for the C and C++ API.
    if std::env::var("LLVM_PROFILE_FILE").is_ok() {
        cmake_config.arg("-DRASCAL_ENABLE_COVERAGE=ON");
    }

    let status = cmake_config.status().expect("cmake configuration failed");
    assert!(status.success());

    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("cmake build failed");
    assert!(status.success());

    let mut ctest = utils::ctest(&build_dir);

    let status = ctest.status().expect("failed to run tests");
    assert!(status.success());
}
