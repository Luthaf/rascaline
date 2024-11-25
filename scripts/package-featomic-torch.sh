#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# featomic-torch, and copy it to the path given as argument

set -eux

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

VERSION=$(cat "$ROOT_DIR/featomic-torch/VERSION")
ARCHIVE_NAME="featomic-torch-cxx-$VERSION"

TMP_DIR=$(mktemp -d)
mkdir "$TMP_DIR/$ARCHIVE_NAME"

cp -r "$ROOT_DIR"/featomic-torch/* "$TMP_DIR/$ARCHIVE_NAME/"
cp "$ROOT_DIR/LICENSE" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/AUTHORS" "$TMP_DIR/$ARCHIVE_NAME"


cd "$TMP_DIR"
tar cf "$ARCHIVE_NAME".tar "$ARCHIVE_NAME"

gzip -9 "$ARCHIVE_NAME".tar

rm -f "$ROOT_DIR"/python/featomic-torch/featomic-torch-cxx-*.tar.gz
cp "$ARCHIVE_NAME".tar.gz "$OUTPUT_DIR/"
