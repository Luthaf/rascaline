#[test]
fn file_size_below_200k() {
    for entry in glob::glob("tests/data/generated/*").unwrap() {
        let path = entry.unwrap();
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(
            metadata.len() < 200 * 1024,
            "'{}' can not be larger than 200KiB", path.display()
        );
    }
}
