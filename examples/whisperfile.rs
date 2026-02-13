use std::path::PathBuf;

use glimpse_speech::{
    engines::whisperfile::{WhisperfileEngine, WhisperfileInferenceParams, WhisperfileModelParams},
    TranscriptionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = PathBuf::from(
        args.get(1)
            .map(|value| value.as_str())
            .unwrap_or("models/parakeet-tdt-0.6b-v3-coreml"),
    );
    let wav_path = PathBuf::from(
        args.get(2)
            .map(|value| value.as_str())
            .unwrap_or("samples/dots.wav"),
    );

    let mut engine = WhisperfileEngine::new(PathBuf::from("whisperfile"));
    engine.load_model_with_params(&model_dir, WhisperfileModelParams::default())?;

    let result = engine.transcribe_file(&wav_path, Some(WhisperfileInferenceParams::default()))?;

    println!("{}", result.text);

    if let Some(segments) = result.segments {
        for segment in segments {
            println!(
                "[{:.2}s - {:.2}s] {}",
                segment.start, segment.end, segment.text
            );
        }
    }

    Ok(())
}
