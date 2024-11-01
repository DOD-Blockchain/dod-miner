# DOD Miners

## Reminder
This is the build instruction for WSL2, and GPU mining is now only available on Nvidia GPU
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

## Build (For WSL2 Only)
### Dependency
```bash
./install.sh
```

### GPU
```bash
cargo build --release --features=gpu
```

### GPU Config
edit `config/gpu_config.toml` for better performance
```
cuda_cores = $cuda_cores_to_use
batch_size = $batch_size_per_run
```

## Run

#### Bash run with cycles price percent(recommend)
```bash
./target/release/dod_miner miner --cycles_price_percent=$cycles_price_percent --wif=$wif_priv_key

```
eg.
```bash
./target/release/dod_miner miner --cycles_price_percent=0.99 --wif=xxxxxxxxxxxxxxxxxxxxx
```

#### Bash run with fixed cycles price
```bash
./target/release/dod_miner miner --cycles_price=$cycles_price --wif=$wif_priv_key
```
eg.
```bash
./target/release/dod_miner miner --cycles_price=1.5 --wif=xxxxxxxxxxxxxxxxxxxxx
```

## Linux
### install cuda toolkit
https://developer.nvidia.com/cuda-toolkit