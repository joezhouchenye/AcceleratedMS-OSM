#!/bin/bash
# Manual build without VS Code
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
BUILD_DIR="$SCRIPT_DIR/build"

# Check if the directory exists
if [ ! -d "$BUILD_DIR" ]; then
    # If it does not exist, create the directory
    mkdir -p "$BUILD_DIR"  # -p flag ensures no error if the directory already exists
    echo "Directory $BUILD_DIR created."
else
    echo "Directory $BUILD_DIR already exists."
fi
cmake --build $BUILD_DIR --config Release --target all --