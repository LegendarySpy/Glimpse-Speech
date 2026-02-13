# glimpse-speech

`glimpse-speech` is a `transcribe-rs` inspired crate built around:
- `whisper-rs` for direct GGML Whisper inference
- FluidAudio (through the bundled Swift bridge) for fast macOS Parakeet-class ASR

The public API is intentionally close to `transcribe-rs`:
- `TranscriptionEngine`
- `TranscriptionResult`
- `TranscriptionSegment`
- `audio::read_wav_samples`
- `engines::*`

## Features

| Feature | Purpose |
|---|---|
| `whisper` | Enable `engines::whisper::WhisperEngine` |
| `parakeet` | Enable `engines::parakeet::ParakeetEngine` (Fluid-backed) |
| `whisperfile` | Enable `engines::whisperfile::WhisperfileEngine` compatibility shim (Fluid-backed) |
| `fluid` | Low-level Fluid engine used by compatibility shims |
| `all` | Enables `whisper`, `parakeet`, and `whisperfile` |

## Installation

```toml
[dependencies]
glimpse-speech = { git = "https://github.com/LegendarySpy/Glimpse-Speech.git", tag = "0.1.0", features = ["whisper", "parakeet"] }
```

For local development in this repository:

```toml
[dependencies]
glimpse-speech = { path = "../Glimpse-Speech", features = ["whisper", "parakeet"] }
```

## Usage

### Whisper (local GGML)

```rust
use glimpse_speech::{engines::whisper::WhisperEngine, TranscriptionEngine};
use std::path::PathBuf;

let mut engine = WhisperEngine::new();
engine.load_model(&PathBuf::from("models/whisper-medium-q4_1.bin"))?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Parakeet-compatible API (Fluid-backed)

```rust
use glimpse_speech::{
    engines::parakeet::{ParakeetEngine, ParakeetModelParams},
    TranscriptionEngine,
};
use std::path::PathBuf;

let mut engine = ParakeetEngine::new();
engine.load_model_with_params(
    &PathBuf::from("models/parakeet-tdt-0.6b-v3-coreml"),
    ParakeetModelParams::int8(),
)?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Fluid Bridge Requirements

Fluid-backed engines require:
- macOS 14+
- A built `GlimpseSpeechFluidBridge` dynamic library (`libGlimpseSpeechFluidBridge.dylib`)
- Model directories compatible with FluidAudio

If the dylib is not auto-discovered, set:

```bash
export GLIMPSE_FLUID_BRIDGE_DYLIB=/absolute/path/to/libGlimpseSpeechFluidBridge.dylib
```

## Releasing With Fluid

`libGlimpseSpeechFluidBridge.dylib` is the native Swift dynamic library that wraps
FluidAudio and exposes a C ABI used by the Rust `FluidEngine`.

Build it from this repo:

```bash
cd swift-bridge
swift build -c release --product GlimpseSpeechFluidBridge
```

When shipping an app that uses Fluid-backed engines, include
`libGlimpseSpeechFluidBridge.dylib` in one of these locations:
- next to the app executable
- `../Frameworks` relative to the executable
- `../Resources` relative to the executable

Or set `GLIMPSE_FLUID_BRIDGE_DYLIB` (or `FluidModelParams::dylib_path`) to an explicit path.

FluidAudio project: [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio)

## Further notes

Example commands:

```bash
cargo run --example whisper --features whisper -- <model.bin> <audio.wav>
cargo run --example parakeet --features parakeet -- <fluid-model-dir> <audio.wav>
cargo run --example whisperfile --features whisperfile -- <fluid-model-dir> <audio.wav>
```


## Acknowledgments


- [transcribe-rs](https://github.com/cjpais/transcribe-rs)(MIT) While not used in this project, GS was heavily inspired by this architecture and mindset towards a backend local rused based API

- [whisper-rs](https://github.com/tazz4843/whisper-rs) (Unlicense) — Rust bindings used by the Whisper engine

- [FluidAudio](https://github.com/FluidInference/FluidAudio) (Apache-2.0) — CoreML inference runtime used by the Parakeet engine on macOS
