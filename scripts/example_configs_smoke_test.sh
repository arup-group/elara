#!/usr/bin/env bash

set -e

pushd "${0%/*}"/..
python scripts/elara_run_smoke_test.py -d example_configs
popd