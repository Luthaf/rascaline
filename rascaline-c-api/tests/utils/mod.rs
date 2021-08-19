use std::process::Command;
use std::io::ErrorKind;

/// Check that a command is available in PATH by calling `<command> --version`
pub fn check_command_exists(command: &str) {
    match Command::new(command).arg("--version").output() {
        Ok(output) => {
            if !output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!(
                    "failed to run `{} --version`.\n\
                    stdout={}\n\
                    stderr={}\n",
                    command, stdout, stderr
                );
            }
        },
        Err(error) => {
            match error.kind() {
                ErrorKind::NotFound => {
                    panic!("could not find '{}' in PATH, is it installed?", command)
                },
                _ => {
                    panic!("could not execute '{}': {}", command, error)
                }
            }
        }
    }
}
