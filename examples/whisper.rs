use std::path::PathBuf;

use glimpse_speech::{
    TranscriptionEngine,
    engines::whisper::{WhisperEngine, WhisperInferenceParams},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(
        args.get(1)
            .map_or("models/whisper-medium-q4_1.bin", String::as_str),
    );
    let wav_path = PathBuf::from(args.get(2).map_or("samples/dots.wav", String::as_str));

    let mut engine = WhisperEngine::new();
    engine.load_model(&model_path)?;

    let result = engine.transcribe_file(
        &wav_path,
        Some(WhisperInferenceParams {
            dictionary: vec!["Glimpse".to_string(), "Parakeet".to_string()],
            ..Default::default()
        }),
    )?;

    println!("{}", result.text);
    for segment in result.segments.unwrap_or_default() {
        println!(
            "[{:.2}s - {:.2}s] {}",
            segment.start, segment.end, segment.text
        );
    }

    Ok(())
}
