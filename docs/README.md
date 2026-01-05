# Audio2Face SDK Documentation

## Overview

The Audio2Face SDK generates 3D facial animation from speech audio using GPU-accelerated models (CUDA + TensorRT).

### Repository components

- `audio2face-sdk/` - Audio2Face public API, core implementation, samples, tests, and benchmarks.
- `audio2x-common/` - Shared utilities used by Audio2Face (CUDA helpers, tensor utilities, accumulators, etc.).
- `audio2x-sdk/` - Builds the `audio2x` shared library which exports the Audio2Face API and bundles required headers.

## Core Concepts

- **Audio accumulator**: Accepts audio samples over time (streaming or offline) and provides executors with the buffered audio they need for inference.
- **Emotion accumulator**: Provides per-timestamp emotion input to Audio2Face. It can be filled from constant values, animation curves, or any external source.
- **Executors**:
  - **Geometry executors** generate animated vertex positions.
  - **Blendshape executors** solve blendshape weights from generated geometry.
  - **Interactive executors** are optimized for fast iteration when tweaking parameters.

## Getting Started

- Build instructions: see `README.md`.
- Optional model/data setup:
  - Download models: `download_models.{bat|sh}`
  - Generate TensorRT engines and sample/test data: `gen_testdata.{bat|sh}`
- Samples:
  - `sample-a2f-executor`
  - `sample-a2f-low-level-api-fullface`
