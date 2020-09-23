use std::process::Command;
use std::path::PathBuf;

#[test]
fn check_python() {
    let mut root = PathBuf::from(file!());
    root.pop();
    root.pop();
    root.push("python");

    let mut tox = Command::new("tox");
    tox.current_dir(&root);
    let status = tox.status().expect("failed to run tox");
    assert!(status.success());
}
