#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# rascaline-torch, and copy it to be included in the rascaline-torch python
# package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -eux

cd "$ROOT_DIR"
tar cf rascaline-torch.tar rascaline-torch
gzip -9 rascaline-torch.tar

mv rascaline-torch.tar.gz python/rascaline-torch/
