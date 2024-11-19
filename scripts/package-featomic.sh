#!/usr/bin/env bash

# This script creates an archive containing the sources for the featomic
# Rust crate, and copy it to the path given as argument

set -eux

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

rm -rf "$ROOT_DIR/target/package"
cd "$ROOT_DIR/featomic"

# Package featomic using cargo tools
cargo package --allow-dirty --no-verify

TMP_DIR=$(mktemp -d)

cd "$TMP_DIR"
tar xf "$ROOT_DIR"/target/package/featomic-*.crate
ARCHIVE_NAME=$(ls)

# extract the version part of the package from the .crate file name
VERSION=${ARCHIVE_NAME:9}
ARCHIVE_NAME="featomic-cxx-$VERSION"

mv featomic-* "$ARCHIVE_NAME"
cp "$ROOT_DIR/LICENSE" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/AUTHORS" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/README.rst" "$TMP_DIR/$ARCHIVE_NAME"

# Get the git version information, this is used when building the
# code to change the version for development builds
cd "$ROOT_DIR"
./scripts/git-version-info.py "featomic-v" > "$TMP_DIR/$ARCHIVE_NAME/cmake/git_version_info"

cd "$TMP_DIR"
# Compile featomic as it's own Cargo workspace
echo "[workspace]" >> "$ARCHIVE_NAME/Cargo.toml"

cargo generate-lockfile --manifest-path "$ARCHIVE_NAME/Cargo.toml"

# remove tests files from the archive, these are relatively big
rm -rf "$ARCHIVE_NAME/tests"

tar cf "$ARCHIVE_NAME.tar" "$ARCHIVE_NAME"
gzip -9 "$ARCHIVE_NAME.tar"

cp "$TMP_DIR/$ARCHIVE_NAME.tar.gz" "$OUTPUT_DIR/"
