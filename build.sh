#!/bin/bash

set -e

ARCH=$(uname -m)
OS=$(uname | tr '[:upper:]' '[:lower:]')

mkdir -p native/${OS}-${ARCH}

case "$OS" in
  darwin)
    clang++ -shared -std=c++11 -o ./native/${OS}-${ARCH}/libfaiss_wrapper.dylib ./src/faiss_wrapper.cpp -I/opt/homebrew/opt/faiss/include -L/opt/homebrew/opt/faiss/lib -lfaiss -fPIC
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
