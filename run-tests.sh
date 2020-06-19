#! /bin/bash

# exit script if any step returns a non-0 code
set -e

echo "Executing tests"

# install development dependencies including test packages
pip3 install -r requirements.txt

# run tests
python3 -m pytest -vv tests

echo "Tests complete"

exit 0
