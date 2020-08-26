#!/usr/bin/env bash
set -ux
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"

rm -rf $ROOT/build
rm -rf $ROOT/dist
rm -rf $ROOT/*.egg-info
rm -rf $ROOT/.tox
rm -rf $ROOT/*/__pycache__
