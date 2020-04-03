use std::process::Command;
use std::path::PathBuf;

fn build_root() -> PathBuf {
    let mut root = PathBuf::from(file!());
    root.pop();
    root.push("c_api");
    root.push("build");
    return root;
}

#[test]
fn check_c_api() {
    let root = build_root();
    std::fs::create_dir_all(&root).expect("failed to create build dir");

    let mut cmake_config = Command::new("cmake");
    cmake_config.current_dir(&root);
    cmake_config.arg("..");
    let status = cmake_config.status().expect("failed to configure cmake");
    assert!(status.success());

    let mut cmake_build = Command::new("cmake");
    cmake_build.current_dir(&root);
    cmake_build.arg("--build");
    cmake_build.arg(".");
    cmake_build.arg("--config");
    cmake_build.arg("Release");
    let status = cmake_build.status().expect("failed to build C++ code");
    assert!(status.success());

    let mut ctest = Command::new("ctest");
    ctest.current_dir(&root);
    ctest.arg("--output-on-failure");
    ctest.arg("--C");
    ctest.arg("Release");
    let status = ctest.status().expect("failed to run tests");
    assert!(status.success());
}
