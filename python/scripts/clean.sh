#/usr/bin/env bash

# This script removes all temporary files created by Python during
# installation and tests running.

set -eux

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/../.. && pwd)
cd $ROOT_DIR

rm -rf build/
rm -rf dist/

find . -name "*.egg-info" -exec rm -rf "{}" +
find . -name "__pycache__" -exec rm -rf "{}" +
