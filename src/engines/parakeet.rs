use std::path::{Path, PathBuf};

use crate::{TranscriptionEngine, TranscriptionResult};

use super::fluid::{
    FluidEngine, FluidInferenceParams, FluidModelParams, FluidTimestampGranularity,
};

#[derive(Debug, Clone, Default, PartialEq)]
pub enum TimestampGranularity {
    #[default]
    Token,
    Word,
    Segment,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum QuantizationType {
    #[default]
    FP32,
    Int8,
}

#[derive(Debug, Clone, Default)]
pub struct ParakeetModelParams {
    pub quantization: QuantizationType,
    pub diarization_model_dir: Option<PathBuf>,
    pub dylib_path: Option<PathBuf>,
    pub runtime_macos_major: Option<u32>,
}

impl ParakeetModelParams {
    pub fn fp32() -> Self {
        Self {
            quantization: QuantizationType::FP32,
            ..Self::default()
        }
    }

    pub fn int8() -> Self {
        Self {
            quantization: QuantizationType::Int8,
            ..Self::default()
        }
    }

    pub fn quantized(quantization: QuantizationType) -> Self {
        Self {
            quantization,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetInferenceParams {
    pub timestamp_granularity: TimestampGranularity,
    pub language: Option<String>,
    pub vocabulary: Vec<String>,
}

impl Default for ParakeetInferenceParams {
    fn default() -> Self {
        Self {
            timestamp_granularity: TimestampGranularity::Token,
            language: None,
            vocabulary: Vec::new(),
        }
    }
}

pub struct ParakeetEngine {
    inner: FluidEngine,
}

impl Default for ParakeetEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ParakeetEngine {
    pub fn new() -> Self {
        Self {
            inner: FluidEngine::new(),
        }
    }
}

impl TranscriptionEngine for ParakeetEngine {
    type InferenceParams = ParakeetInferenceParams;
    type ModelParams = ParakeetModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _ = params.quantization;

        self.inner.load_model_with_params(
            model_path,
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

fn map_inference_params(params: Option<ParakeetInferenceParams>) -> FluidInferenceParams {
    let params = params.unwrap_or_default();

    let timestamp_granularity = match params.timestamp_granularity {
        TimestampGranularity::Segment => FluidTimestampGranularity::SegmentsOnly,
        TimestampGranularity::Token | TimestampGranularity::Word => {
            FluidTimestampGranularity::WordPreferred
        }
    };

    FluidInferenceParams {
        language: params.language,
        vocabulary: params.vocabulary,
        timestamp_granularity,
    }
}

#[cfg(test)]
mod tests {
    use super::{ParakeetModelParams, QuantizationType};

    #[test]
    fn int8_constructor_sets_quantized_mode() {
        let params = ParakeetModelParams::int8();
        assert_eq!(params.quantization, QuantizationType::Int8);
    }

    #[test]
    fn fp32_constructor_sets_full_precision_mode() {
        let params = ParakeetModelParams::fp32();
        assert_eq!(params.quantization, QuantizationType::FP32);
    }
}
