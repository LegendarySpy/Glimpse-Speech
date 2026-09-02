#[cfg(nvidia_engines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::{Path, PathBuf};

    use glimpse_speech::{
        TranscriptionEngine, TranscriptionResult,
        engines::nemotron::NemotronEngine,
        engines::parakeet::{ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams},
    };

    let args: Vec<String> = std::env::args().collect();
    let engine = args.get(1).map_or("parakeet", String::as_str);

    let default_model_dir = match engine {
        "parakeet" => "models/parakeet-tdt-0.6b-v3-onnx-int8",
        "nemotron" => "models/nemotron-speech-streaming-en-0.6b",
        other => {
            return Err(format!(
                "Unknown NVIDIA engine `{other}`. Expected `parakeet` or `nemotron`."
            )
            .into());
        }
    };

    let model_dir = PathBuf::from(args.get(2).map_or(default_model_dir, String::as_str));
    let wav_path = PathBuf::from(args.get(3).map_or("samples/dots.wav", String::as_str));

    fn transcribe_with_parakeet(
        model_dir: &Path,
        wav_path: &Path,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let mut engine = ParakeetEngine::new();
        engine.load_model_with_params(model_dir, ParakeetModelParams::int8())?;
        engine.transcribe_file(wav_path, Some(ParakeetInferenceParams::default()))
    }

    fn transcribe_with_nemotron(
        model_dir: &Path,
        wav_path: &Path,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let mut engine = NemotronEngine::new();
        engine.load_model(model_dir)?;
        engine.transcribe_file(wav_path, None)
    }

    let result = match engine {
        "parakeet" => transcribe_with_parakeet(&model_dir, &wav_path)?,
        "nemotron" => transcribe_with_nemotron(&model_dir, &wav_path)?,
        _ => unreachable!(),
    };

    println!("{}", result.text);
    for segment in result.segments.unwrap_or_default() {
        println!(
            "[{:.2}s - {:.2}s] {}",
            segment.start, segment.end, segment.text
        );
    }

    Ok(())
}

#[cfg(not(nvidia_engines))]
fn main() {
    eprintln!("The NVIDIA example is unavailable on Intel macOS builds.");
}
