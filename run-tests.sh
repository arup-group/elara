#! /bin/bash

# exit script if any step returns a non-0 code
set -e

echo "Executing tests"

python3 -m pytest -n auto -vv tests

echo "Tests complete"

exit 0
