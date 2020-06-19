#! /bin/bash

# exit script if any step returns a non-0 code
set -e

echo "Executing tests"

python3 -m pytest -vv tests

echo "Tests complete"

exit 0
