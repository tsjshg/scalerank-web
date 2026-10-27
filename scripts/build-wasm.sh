#!/bin/sh
# Builds the WASM bindings into web/src/wasm (single-threaded, SIMD128).
set -e
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown -p scalerank-wasm
wasm-bindgen --target web --out-dir web/src/wasm target/wasm32-unknown-unknown/release/scalerank_wasm.wasm
