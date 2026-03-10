#!/bin/bash

set -e

RAW_ARCH=$(uname -m)
OS=$(uname | tr '[:upper:]' '[:lower:]')

# Normalize architecture names to match Deno.build.arch
case "$RAW_ARCH" in
  arm64) ARCH="aarch64" ;;  # macOS returns arm64, Deno uses aarch64
  *)     ARCH="$RAW_ARCH" ;;
esac

mkdir -p native/${OS}-${ARCH}

case "$OS" in
  darwin)
    SDK_PATH=$(xcrun --show-sdk-path)
    xcrun clang++ -shared -std=c++17 -stdlib=libc++ \
      -isystem "${SDK_PATH}/usr/include/c++/v1" \
      -o ./native/${OS}-${ARCH}/libfaiss_wrapper.dylib \
      ./src/faiss_wrapper.cpp \
      -I/opt/homebrew/opt/faiss/include \
      -L/opt/homebrew/opt/faiss/lib -lfaiss -fPIC
    ;;
  linux)
    g++ -std=c++11 -shared -fPIC -o native/${OS}-${ARCH}/libfaiss_wrapper.so src/faiss_wrapper.cpp
    ;;
  mingw* | msys* | cygwin* | windows)
    x86_64-w64-mingw32-g++ -shared -o native/windows-${ARCH}/libfaiss_wrapper.dll src/faiss_wrapper.cpp
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

echo "✅ Built for $OS-$ARCH"
