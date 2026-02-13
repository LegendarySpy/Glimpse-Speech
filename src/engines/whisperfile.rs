use std::path::{Path, PathBuf};

use crate::{TranscriptionEngine, TranscriptionResult};

use super::fluid::{
    FluidEngine, FluidInferenceParams, FluidModelParams, FluidTimestampGranularity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GPUMode {
    #[default]
    Auto,
    Apple,
    Amd,
    Nvidia,
    Disabled,
}

impl GPUMode {
    pub fn as_arg(&self) -> &'static str {
        match self {
            GPUMode::Auto => "auto",
            GPUMode::Apple => "apple",
            GPUMode::Amd => "amd",
            GPUMode::Nvidia => "nvidia",
            GPUMode::Disabled => "disabled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WhisperfileModelParams {
    pub port: u16,
    pub host: String,
    pub startup_timeout_secs: u64,
    pub gpu: GPUMode,
    pub diarization_model_dir: Option<PathBuf>,
    pub dylib_path: Option<PathBuf>,
    pub runtime_macos_major: Option<u32>,
}

impl Default for WhisperfileModelParams {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            startup_timeout_secs: 30,
            gpu: GPUMode::Auto,
            diarization_model_dir: None,
            dylib_path: None,
            runtime_macos_major: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WhisperfileInferenceParams {
    pub language: Option<String>,
    pub translate: bool,
    pub temperature: Option<f32>,
    pub response_format: Option<String>,
    pub vocabulary: Vec<String>,
}

impl Default for WhisperfileInferenceParams {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            temperature: None,
            response_format: Some("verbose_json".to_string()),
            vocabulary: Vec::new(),
        }
    }
}

/// Compatibility Whisperfile API, but executes through the FluidAudio bridge.
pub struct WhisperfileEngine {
    #[allow(dead_code)]
    binary_path: PathBuf,
    inner: FluidEngine,
}

impl WhisperfileEngine {
    pub fn new(binary_path: impl Into<PathBuf>) -> Self {
        Self {
            binary_path: binary_path.into(),
            inner: FluidEngine::new(),
        }
    }
}

impl Default for WhisperfileEngine {
    fn default() -> Self {
        Self::new(PathBuf::from("whisperfile"))
    }
}

impl TranscriptionEngine for WhisperfileEngine {
    type InferenceParams = WhisperfileInferenceParams;
    type ModelParams = WhisperfileModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _ = (
            &params.port,
            &params.host,
            &params.startup_timeout_secs,
            &params.gpu,
        );
        let effective_model_path = if model_path.is_file() {
            model_path
                .parent()
                .ok_or_else(|| std::io::Error::other("model file path has no parent directory"))?
        } else {
            model_path
        };

        self.inner.load_model_with_params(
            effective_model_path,
            FluidModelParams {
                diarization_model_dir: params.diarization_model_dir,
                dylib_path: params.dylib_path,
                runtime_macos_major: params.runtime_macos_major,
            },
        )
    }

    fn unload_model(&mut self) {
        self.inner.unload_model();
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.inner
            .transcribe_samples(samples, Some(map_inference_params(params)))
    }

    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.inner
            .transcribe_file(wav_path, Some(map_inference_params(params)))
    }
}

fn map_inference_params(params: Option<WhisperfileInferenceParams>) -> FluidInferenceParams {
    let params = params.unwrap_or_default();

    let _ = (params.translate, params.temperature, params.response_format);

    FluidInferenceParams {
        language: params.language,
        vocabulary: params.vocabulary,
        timestamp_granularity: FluidTimestampGranularity::WordPreferred,
    }
}
