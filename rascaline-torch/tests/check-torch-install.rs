use std::path::PathBuf;

mod utils;

#[test]
fn check_torch_install() {
    if cfg!(tarpaulin) {
        // do not run this test when collecting Rust coverage
        return;
    }

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install rascaline-torch with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-install");
    let deps_dir = build_dir.join("deps");

    let rascaline_dep = deps_dir.join("rascaline");
    std::fs::create_dir_all(&rascaline_dep).expect("failed to create rascaline dep dir");
    let rascaline_cmake_prefix = utils::setup_rascaline(rascaline_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let pytorch_cmake_prefix = utils::setup_pytorch(torch_dep);

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // configure cmake for rascaline-torch
    let rascaline_torch_dep = deps_dir.join("rascaline-torch");
    let install_prefix = rascaline_torch_dep.join("usr");
    std::fs::create_dir_all(&rascaline_torch_dep).expect("failed to create rascaline-torch dep dir");
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &rascaline_torch_dep);
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        rascaline_cmake_prefix.display(),
        pytorch_cmake_prefix.display()
    ));

    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));
    cmake_config.arg("-DRASCALINE_TORCH_FETCH_METATENSOR_TORCH=ON");

    // The two properties below handle the RPATH for rascaline_torch, setting it
    // in such a way that we can always load librascaline.so and libtorch.so
    // from the location they are found at when compiling rascaline-torch. See
    // https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
    // for more information on CMake RPATH handling
    cmake_config.arg("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON");
    cmake_config.arg("-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON");

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run rascaline_torch cmake configuration");

    // build and install rascaline-torch
    let mut cmake_build = utils::cmake_build(&rascaline_torch_dep);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run rascaline_torch cmake build");

    // ====================================================================== //
    // try to use the installed rascaline-torch from cmake

    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{};{}",
        rascaline_cmake_prefix.display(),
        pytorch_cmake_prefix.display(),
        install_prefix.display(),
    ));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake configuration");

    // build the code, linking to rascaline-torch
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake build");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run test project tests");
}
