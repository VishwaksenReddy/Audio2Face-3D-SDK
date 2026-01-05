#!/bin/bash

set -e

BASE_DIR="$(dirname ${BASH_SOURCE})"

# Set up PYTHONPATH
export PYTHONPATH="$BASE_DIR/audio2x-common/scripts:$PYTHONPATH"

if [ -z "$TENSORRT_ROOT_DIR" ]; then
    echo "TENSORRT_ROOT_DIR is not defined"
    exit 1
fi

export PATH="$TENSORRT_ROOT_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_ROOT_DIR/lib:$LD_LIBRARY_PATH"

echo "Generating test data..."
python "$BASE_DIR/audio2x-common/scripts/gen_test_data.py"
python "$BASE_DIR/audio2face-sdk/scripts/gen_test_data.py"

echo "Generating sample data..."
python "$BASE_DIR/audio2face-sdk/scripts/gen_sample_data.py"
