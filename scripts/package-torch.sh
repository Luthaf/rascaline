#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# featomic-torch, and copy it to be included in the featomic-torch python
# package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -eux

cd "$ROOT_DIR"
tar cf featomic-torch.tar featomic-torch
gzip -9 featomic-torch.tar

mv featomic-torch.tar.gz python/featomic-torch/
