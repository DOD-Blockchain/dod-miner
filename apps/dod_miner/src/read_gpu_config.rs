use config_file::FromConfigFile;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]

pub struct GpuConfig {
    pub cuda_cores: u32,
    pub batch_size: u64,
}

pub fn read_config() -> GpuConfig {
    GpuConfig::from_config_file("config/gpu_config.toml").unwrap()
}