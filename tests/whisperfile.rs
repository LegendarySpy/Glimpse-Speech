#![cfg(feature = "whisperfile")]

use glimpse_speech::engines::whisperfile::GPUMode;

#[test]
fn gpu_mode_values_are_stable() {
    assert_eq!(GPUMode::Auto.as_arg(), "auto");
    assert_eq!(GPUMode::Apple.as_arg(), "apple");
    assert_eq!(GPUMode::Amd.as_arg(), "amd");
    assert_eq!(GPUMode::Nvidia.as_arg(), "nvidia");
    assert_eq!(GPUMode::Disabled.as_arg(), "disabled");
}
