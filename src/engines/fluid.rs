use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use libloading::{Library, Symbol};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{TranscriptionEngine, TranscriptionResult, TranscriptionSegment};

const BRIDGE_SCHEMA_VERSION: u32 = 1;
static TEMP_WAV_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FluidTimestampGranularity {
    #[default]
    WordPreferred,
    SegmentsOnly,
}

impl FluidTimestampGranularity {
    fn as_wire_value(self) -> &'static str {
        match self {
            Self::WordPreferred => "word_preferred",
            Self::SegmentsOnly => "segments_only",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FluidModelParams {
    pub diarization_model_dir: Option<PathBuf>,
    pub dylib_path: Option<PathBuf>,
    pub runtime_macos_major: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct FluidInferenceParams {
    pub language: Option<String>,
    pub vocabulary: Vec<String>,
    pub timestamp_granularity: FluidTimestampGranularity,
}

impl Default for FluidInferenceParams {
    fn default() -> Self {
        Self {
            language: None,
            vocabulary: Vec::new(),
            timestamp_granularity: FluidTimestampGranularity::WordPreferred,
        }
    }
}

pub struct FluidEngine {
    loaded_model_path: Option<PathBuf>,
    bridge: Option<FluidBridge>,
}

impl Default for FluidEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FluidEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            bridge: None,
        }
    }
}

impl Drop for FluidEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for FluidEngine {
    type InferenceParams = FluidInferenceParams;
    type ModelParams = FluidModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model_path, params);
            return Err(io_error("Fluid engine is only supported on macOS"));
        }

        #[cfg(target_os = "macos")]
        {
            if !model_path.exists() {
                return Err(io_error(format!(
                    "Model directory not found: {}",
                    model_path.display()
                )));
            }

            let runtime_macos_major = params
                .runtime_macos_major
                .or_else(detect_macos_major)
                .ok_or_else(|| io_error("failed to determine macOS version"))?;

            if runtime_macos_major < 14 {
                return Err(io_error(format!(
                    "Fluid engine requires macOS 14+, found macOS {runtime_macos_major}"
                )));
            }

            let bridge = FluidBridge::new(
                model_path.to_path_buf(),
                params.diarization_model_dir,
                runtime_macos_major,
                params.dylib_path.as_deref(),
            )?;

            self.loaded_model_path = Some(model_path.to_path_buf());
            self.bridge = Some(bridge);
            Ok(())
        }
    }

    fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.bridge = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        if self.bridge.is_none() {
            return Err(io_error("Model not loaded. Call load_model() first."));
        }

        let temp_wav = TempWav::from_f32_samples_16khz(&samples)?;
        self.transcribe_file(temp_wav.path(), params)
    }

    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let bridge = self
            .bridge
            .as_ref()
            .ok_or_else(|| io_error("Model not loaded. Call load_model() first."))?;

        if !wav_path.exists() {
            return Err(io_error(format!(
                "Audio file not found: {}",
                wav_path.display()
            )));
        }

        let params = params.unwrap_or_default();
        bridge.transcribe(wav_path, &params)
    }
}

struct TempWav {
    path: PathBuf,
}

impl TempWav {
    fn from_f32_samples_16khz(samples: &[f32]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut path = std::env::temp_dir();
        path.push(unique_temp_wav_name());

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(&path, spec)?;
        for sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm = (clamped * i16::MAX as f32).round() as i16;
            writer.write_sample(pcm)?;
        }
        writer.finalize()?;

        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWav {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn unique_temp_wav_name() -> String {
    let counter = TEMP_WAV_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("glimpse-speech-fluid-{nanos}-{counter}.wav")
}

struct FluidBridge {
    library: Arc<FluidBridgeLibrary>,
    handle: Mutex<usize>,
}

impl FluidBridge {
    fn new(
        asr_model_dir: PathBuf,
        diarization_model_dir: Option<PathBuf>,
        runtime_macos_major: u32,
        explicit_dylib_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Arc::new(FluidBridgeLibrary::load(explicit_dylib_path)?);

        let payload = BridgeConfigPayload {
            schema_version: BRIDGE_SCHEMA_VERSION,
            asr_model_dir: asr_model_dir.display().to_string(),
            diarization_model_dir: diarization_model_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            runtime_macos_major,
        };

        let payload_bytes = serde_json::to_vec(&payload)?;
        let payload_len = isize::try_from(payload_bytes.len())
            .map_err(|_| io_error("Fluid create payload is too large"))?;

        // SAFETY: function pointer comes from the loaded Fluid bridge dylib.
        let handle = unsafe { (library.create)(payload_bytes.as_ptr(), payload_len) };

        if handle.is_null() {
            return Err(io_error("Fluid bridge failed to initialize"));
        }

        Ok(Self {
            library,
            handle: Mutex::new(handle as usize),
        })
    }

    fn active_handle(&self) -> Result<*mut c_void, Box<dyn std::error::Error>> {
        let guard = self
            .handle
            .lock()
            .map_err(|_| io_error("failed to lock Fluid bridge handle"))?;
        if *guard == 0 {
            return Err(io_error("Fluid bridge handle is unavailable"));
        }

        Ok(*guard as *mut c_void)
    }

    fn transcribe(
        &self,
        wav_path: &Path,
        params: &FluidInferenceParams,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let payload = BridgeTranscribePayload {
            schema_version: BRIDGE_SCHEMA_VERSION,
            language_hint: normalize_language_hint(params.language.as_deref()),
            vocabulary: normalize_vocabulary(&params.vocabulary),
            timestamps: params.timestamp_granularity.as_wire_value(),
        };

        let payload_bytes = serde_json::to_vec(&payload)?;
        let payload_len = isize::try_from(payload_bytes.len())
            .map_err(|_| io_error("Fluid transcribe payload is too large"))?;

        let wav_path_c = CString::new(wav_path.display().to_string())
            .map_err(|_| io_error("wav path contains interior null bytes"))?;
        let handle = self.active_handle()?;

        let mut out_len: isize = 0;
        // SAFETY: all pointers and lengths are valid for the duration of the call.
        let out_ptr = unsafe {
            (self.library.transcribe_wav)(
                handle,
                wav_path_c.as_ptr(),
                payload_bytes.as_ptr(),
                payload_len,
                &mut out_len,
            )
        };

        let bytes = self.library.take_buffer(out_ptr, out_len)?;
        let payload: BridgeTranscriptPayload = parse_bridge_payload(&bytes, "transcribe")?;
        Ok(payload.into_transcription_result())
    }
}

impl Drop for FluidBridge {
    fn drop(&mut self) {
        let Ok(mut guard) = self.handle.lock() else {
            return;
        };
        if *guard == 0 {
            return;
        }

        let handle = *guard as *mut c_void;
        // SAFETY: handle was created by the same bridge dylib.
        unsafe {
            (self.library.destroy)(handle);
        }
        *guard = 0;
    }
}

type GlimpseFluidCreateFn = unsafe extern "C" fn(*const u8, isize) -> *mut c_void;
type GlimpseFluidDestroyFn = unsafe extern "C" fn(*mut c_void);
type GlimpseFluidTranscribeFn =
    unsafe extern "C" fn(*mut c_void, *const i8, *const u8, isize, *mut isize) -> *mut u8;
type GlimpseFluidFreeBufferFn = unsafe extern "C" fn(*mut u8, isize);

struct FluidBridgeLibrary {
    _library: Library,
    create: GlimpseFluidCreateFn,
    destroy: GlimpseFluidDestroyFn,
    transcribe_wav: GlimpseFluidTranscribeFn,
    free_buffer: GlimpseFluidFreeBufferFn,
}

impl FluidBridgeLibrary {
    fn load(explicit_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let dylib_path = resolve_bridge_dylib_path(explicit_path)?;

        // SAFETY: loading shared library by filesystem path.
        let library = unsafe { Library::new(&dylib_path) }.map_err(|error| {
            io_error(format!(
                "failed to load Fluid bridge dylib {}: {error}",
                dylib_path.display()
            ))
        })?;

        let create = load_symbol::<GlimpseFluidCreateFn>(&library, b"glimpse_fluid_create\0")?;
        let destroy = load_symbol::<GlimpseFluidDestroyFn>(&library, b"glimpse_fluid_destroy\0")?;
        let transcribe_wav =
            load_symbol::<GlimpseFluidTranscribeFn>(&library, b"glimpse_fluid_transcribe_wav\0")?;
        let free_buffer =
            load_symbol::<GlimpseFluidFreeBufferFn>(&library, b"glimpse_fluid_free_buffer\0")?;

        Ok(Self {
            _library: library,
            create,
            destroy,
            transcribe_wav,
            free_buffer,
        })
    }

    fn take_buffer(&self, ptr: *mut u8, len: isize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if ptr.is_null() || len <= 0 {
            return Err(io_error("Fluid bridge returned an empty response"));
        }

        let len_isize = len;
        let len =
            usize::try_from(len_isize).map_err(|_| io_error("invalid Fluid response length"))?;

        // SAFETY: `ptr` points to `len` bytes returned by bridge API.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: buffer ownership is returned to bridge API here.
        unsafe {
            (self.free_buffer)(ptr, len_isize);
        }

        Ok(bytes)
    }
}

fn load_symbol<T>(library: &Library, symbol: &[u8]) -> Result<T, Box<dyn std::error::Error>>
where
    T: Copy,
{
    // SAFETY: symbol lookup in a loaded library.
    let value: Symbol<'_, T> = unsafe { library.get(symbol) }.map_err(|error| {
        io_error(format!(
            "missing Fluid bridge symbol {}: {error}",
            String::from_utf8_lossy(symbol)
        ))
    })?;

    Ok(*value)
}

fn resolve_bridge_dylib_path(
    explicit_path: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = explicit_path {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(io_error(format!(
            "Fluid bridge dylib path does not exist: {}",
            path.display()
        )));
    }

    if let Ok(path) = std::env::var("GLIMPSE_FLUID_BRIDGE_DYLIB") {
        let candidate = PathBuf::from(path);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let mut candidates = Vec::new();

    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("libGlimpseSpeechFluidBridge.dylib"));
            candidates.push(parent.join("../Frameworks/libGlimpseSpeechFluidBridge.dylib"));
            candidates.push(parent.join("../Resources/libGlimpseSpeechFluidBridge.dylib"));
        }
    }

    #[cfg(debug_assertions)]
    {
        // Dev-only fallback for local workspace builds.
        let bridge_build = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift-bridge/.build");
        candidates.push(bridge_build.join("debug/libGlimpseSpeechFluidBridge.dylib"));
        candidates.push(bridge_build.join("release/libGlimpseSpeechFluidBridge.dylib"));
        candidates
            .push(bridge_build.join("arm64-apple-macosx/debug/libGlimpseSpeechFluidBridge.dylib"));
        candidates.push(
            bridge_build.join("arm64-apple-macosx/release/libGlimpseSpeechFluidBridge.dylib"),
        );
        candidates
            .push(bridge_build.join("x86_64-apple-macosx/debug/libGlimpseSpeechFluidBridge.dylib"));
        candidates.push(
            bridge_build.join("x86_64-apple-macosx/release/libGlimpseSpeechFluidBridge.dylib"),
        );
    }

    if let Some(found) = candidates.into_iter().find(|candidate| candidate.exists()) {
        return Ok(found);
    }

    Err(io_error(
        "Fluid bridge dylib not found. Bundle libGlimpseSpeechFluidBridge.dylib near your app binary (or in ../Frameworks/../Resources), set GLIMPSE_FLUID_BRIDGE_DYLIB, or provide FluidModelParams::dylib_path",
    ))
}

#[derive(Debug, Serialize)]
struct BridgeConfigPayload {
    schema_version: u32,
    asr_model_dir: String,
    diarization_model_dir: Option<String>,
    runtime_macos_major: u32,
}

#[derive(Debug, Serialize)]
struct BridgeTranscribePayload {
    schema_version: u32,
    language_hint: Option<String>,
    vocabulary: Vec<String>,
    timestamps: &'static str,
}

#[derive(Debug, Deserialize)]
struct BridgeEnvelope<T> {
    schema_version: u32,
    ok: bool,
    data: Option<T>,
    error: Option<BridgeErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct BridgeErrorPayload {
    code: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct BridgeTranscriptPayload {
    text: String,
    segments: Vec<BridgeSegmentPayload>,
}

#[derive(Debug, Deserialize)]
struct BridgeSegmentPayload {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

impl BridgeTranscriptPayload {
    fn into_transcription_result(self) -> TranscriptionResult {
        let mut text = self.text.trim().to_string();

        let segments = self
            .segments
            .into_iter()
            .filter_map(|segment| {
                if segment.end_ms <= segment.start_ms || segment.text.trim().is_empty() {
                    return None;
                }

                Some(TranscriptionSegment {
                    start: segment.start_ms as f32 / 1000.0,
                    end: segment.end_ms as f32 / 1000.0,
                    text: segment.text,
                })
            })
            .collect::<Vec<_>>();

        if text.is_empty() {
            text = segments
                .iter()
                .map(|segment| segment.text.trim())
                .filter(|segment_text| !segment_text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }

        let segments = if segments.is_empty() {
            None
        } else {
            Some(segments)
        };

        TranscriptionResult { text, segments }
    }
}

fn parse_bridge_payload<T>(bytes: &[u8], action: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: DeserializeOwned,
{
    let envelope: BridgeEnvelope<Value> = serde_json::from_slice(bytes).map_err(|error| {
        io_error(format!(
            "failed to decode Fluid {action} envelope: {error}; payload_preview={}",
            preview_payload(bytes)
        ))
    })?;

    if envelope.schema_version != BRIDGE_SCHEMA_VERSION {
        return Err(io_error(format!(
            "Fluid bridge schema mismatch: expected {}, got {}",
            BRIDGE_SCHEMA_VERSION, envelope.schema_version
        )));
    }

    if !envelope.ok {
        let error = envelope
            .error
            .ok_or_else(|| io_error(format!("Fluid {action} failed without error payload")))?;

        return Err(io_error(format!("{}: {}", error.code, error.message)));
    }

    let data = envelope
        .data
        .ok_or_else(|| io_error(format!("Fluid {action} succeeded without payload")))?;

    serde_json::from_value(data)
        .map_err(|error| io_error(format!("failed to decode Fluid {action} payload: {error}")))
}

fn preview_payload(bytes: &[u8]) -> String {
    const MAX_PREVIEW_BYTES: usize = 240;
    let len = bytes.len().min(MAX_PREVIEW_BYTES);
    let snippet = String::from_utf8_lossy(&bytes[..len]);
    if bytes.len() > MAX_PREVIEW_BYTES {
        format!("{snippet}...")
    } else {
        snippet.into_owned()
    }
}

fn normalize_language_hint(value: Option<&str>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }

    let base = raw.split(['-', '_']).next().unwrap_or(raw);
    let normalized = base.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    Some(normalized)
}

fn normalize_vocabulary(values: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(values.len());
    let mut seen = std::collections::HashSet::new();

    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }

        let key = trimmed.to_ascii_lowercase();
        if seen.insert(key) {
            out.push(trimmed.to_string());
        }
    }

    out
}

#[cfg(target_os = "macos")]
fn detect_macos_major() -> Option<u32> {
    use std::process::Command;

    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let version = String::from_utf8(output.stdout).ok()?;
    version.trim().split('.').next()?.parse::<u32>().ok()
}

#[cfg(not(target_os = "macos"))]
fn detect_macos_major() -> Option<u32> {
    None
}

fn io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::{
        parse_bridge_payload, BridgeTranscriptPayload, FluidTimestampGranularity,
        TranscriptionSegment,
    };

    #[test]
    fn timestamp_granularity_wire_values_are_stable() {
        assert_eq!(
            FluidTimestampGranularity::WordPreferred.as_wire_value(),
            "word_preferred"
        );
        assert_eq!(
            FluidTimestampGranularity::SegmentsOnly.as_wire_value(),
            "segments_only"
        );
    }

    #[test]
    fn parses_success_envelope() {
        let json = br#"{"schema_version":1,"ok":true,"data":{"text":"hello","segments":[{"start_ms":0,"end_ms":500,"text":"hello"}]},"error":null}"#;
        let payload: BridgeTranscriptPayload =
            parse_bridge_payload(json, "transcribe").expect("valid envelope should parse");
        let result = payload.into_transcription_result();

        assert_eq!(result.text, "hello");
        assert_eq!(
            result.segments,
            Some(vec![TranscriptionSegment {
                start: 0.0,
                end: 0.5,
                text: "hello".to_string(),
            }])
        );
    }

    #[test]
    fn reports_bridge_error_payload() {
        let json = br#"{"schema_version":1,"ok":false,"data":null,"error":{"code":"unsupported_platform","message":"macOS 13 is unsupported"}}"#;
        let error = parse_bridge_payload::<BridgeTranscriptPayload>(json, "transcribe")
            .expect_err("bridge error should map to error");

        assert!(error.to_string().contains("unsupported_platform"));
    }
}
