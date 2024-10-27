use crate::state::RUNNING;
use crate::types::{MiningResult, ThreadResult};

#[cfg(feature = "gpu")]
use crate::types::{MiningResultGpu, ThreadResultGpu};

use dod_cpu::threads::{get_available_threads, get_multi_progress};
use dod_cpu::tx::{create_dod_tx, CreateDodTxDefault};

use dod_utils::bitwork::Bitwork;

#[cfg(feature = "gpu")]
use dod_utils::mine::easy_bitwork_2;

use flume::Sender;
use log::info;
use std::thread;
use std::time::{Duration, SystemTime};

use std::sync::Arc;
use tokio::sync::Mutex;
use crate::read_gpu_config::read_config;

pub async fn multi_run_v3(
    bitwork: Bitwork,
    remote_hash: Vec<u8>,
    raw_pubkey: Vec<u8>,
    threads: Option<u32>,
    dead_line: u128,
) -> Result<MiningResult, String> {
    let mut thread_available = get_available_threads();
    thread_available = if threads.is_some() {
        threads.unwrap()
    } else {
        thread_available
    };

    let mut running = RUNNING.lock().await;

    info!("Running {} CPU threads", thread_available);

    *running = true;

    let (_, _, tx, rx) = get_multi_progress::<ThreadResult>();

    let _start_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    for i in 0..thread_available {
        let _tx = tx.clone();
        let _bitwork = bitwork.clone();
        let _remote_hash = remote_hash.clone();
        let _raw_pubkey = raw_pubkey.clone();
        let _dead_line = dead_line.clone();

        thread::spawn(move || {
            let res = sub_task_v3(
                _remote_hash,
                _raw_pubkey,
                _bitwork,
                _dead_line,
                _tx.clone(),
                i,
            );

            match _tx.clone().send(res.clone()) {
                Ok(_) => {}
                Err(_) => {}
            }
        });
    }

    let mut ex: Option<MiningResult> = None;
    let mut _expired = false;
    // let mut nonces = vec![0u64; thread_available as usize];

    for v in rx.iter() {
        if v.expired || v.res.is_some() {
            ex = v.res.clone();
            drop(tx);
            break;
        }
        // nonces[v.index as usize] = v.generated_nonce as u64;

        _expired = v.expired;
    }

    // TODO: this is a hack to make sure all threads are done
    thread::sleep(Duration::from_millis(300));
    // tokio::time::sleep(Duration::from_millis(300)).await;

    let mut _used_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        - _start_time;

    if _used_time == 0 {
        _used_time = 1;
    }
    if ex.is_some() {
        *running = false;
        info!("Mining completed in {}ms", _used_time);
        Ok(ex.unwrap())
    } else {
        *running = false;
        info!("Mining exited in {}ms", _used_time);
        Err("Exited on deadline".to_string())
    }

    // Ok("".to_string())
}

pub fn sub_task_v3(
    remote_hash: Vec<u8>,
    raw_pubkey: Vec<u8>,
    bitwork: Bitwork,
    dead_line: u128,
    _tx: Sender<ThreadResult>,
    index: u32,
) -> ThreadResult {
    #[allow(unused_assignments)]
    let mut ret = ThreadResult {
        res: None,
        generated_nonce: 0,
        expired: true,
        index,
    };
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    let (tx, start) = create_dod_tx(
        CreateDodTxDefault {
            nonce: index,
            time,
            remote_hash: remote_hash.clone(),
            raw_pubkey,
        },
        false,
    );

    let split_length = if bitwork.pre % 2 != 0 {
        bitwork.pre + 1
    } else {
        bitwork.pre
    };

    let actual_pre = hex::encode(remote_hash.clone())
        .split_at(split_length as usize)
        .0
        .to_string();

    loop {
        match dod_utils::mine::mine_bitwork_with_deadline(
            tx.clone(),
            start as usize,
            1,
            actual_pre.clone(),
            bitwork.pre,
            Some(bitwork.post_hex.clone()),
            dead_line,
        ) {
            Ok(res) => {
                ret = ThreadResult {
                    res: Some(MiningResult {
                        num_bytes: res,
                        nonce: index,
                        time,
                    }),
                    generated_nonce: res,
                    expired: false,
                    index,
                };
                break;
            }
            Err(e) => {
                if e.contains("exceeded the allowed") {
                    ret = ThreadResult {
                        res: None,
                        generated_nonce: u64::MAX - 1,
                        expired: true,
                        index,
                    };
                    break;
                } else {
                    ThreadResult {
                        res: None,
                        generated_nonce: u64::from_str_radix(e.as_str(), 10).unwrap(),
                        expired: true,
                        index,
                    };
                    break;
                }
            }
        }
    }
    ret.clone()
}

#[cfg(feature = "gpu")]
pub async fn multi_run_v3_gpu(
    bitwork: Bitwork,
    remote_hash: Vec<u8>,
    raw_pubkey: Vec<u8>,
    _threads: Option<u32>,
    dead_line: u128,
) -> Result<MiningResultGpu, String> {
    let running = Arc::new(Mutex::new(true));
    info!("Running GPU mining");

    let (_, _, tx, rx) = get_multi_progress::<ThreadResultGpu>();
    let cancel_flag = Arc::new(Mutex::new(false));

    let start_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    // let mut handles = vec![];

    let tx = tx.clone();
    let bitwork = bitwork.clone();
    let remote_hash = remote_hash.clone();
    let raw_pubkey = raw_pubkey.clone();
    let cancel_flag = cancel_flag.clone();

        loop {
            if *cancel_flag.lock().await {
                break;
            }

            let res = sub_task_v3_gpu(
                remote_hash.clone(),
                raw_pubkey.clone(),
                bitwork.clone(),
                dead_line,
                tx.clone(),
                0
            )
            .await;

            if tx.send(res.clone()).is_err() {
                break;
            }

            if res.expired || res.res.is_some() {
                break;
            }
        };

    // handles.push(handle);

    let result = {
        let mut ex: Option<MiningResultGpu> = None;
        let mut expired = false;

        let timeout = Duration::from_millis(dead_line as u64);
        let start = SystemTime::now();

        while let Ok(v) =
            rx.recv_timeout(timeout.saturating_sub(start.elapsed().unwrap_or(timeout)))
        {
            if v.expired || v.res.is_some() {
                ex = v.res.clone();
                expired = v.expired;
                break;
            }

            if start.elapsed().unwrap_or(timeout) >= timeout {
                expired = true;
                break;
            }
        }

        (ex, expired)
    };

    *cancel_flag.lock().await = true;

    drop(tx);

    let used_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        - start_time;

    let used_time = if used_time == 0 { 1 } else { used_time };

    *running.lock().await = false;

    match result {
        (Some(ex), _) => {
            info!("Mining completed in {}ms", used_time);
            Ok(ex)
        }
        (None, true) => {
            info!("Mining exited in {}ms", used_time);
            Err("Exited on deadline".to_string())
        }
        _ => Err("Unexpected termination".to_string()),
    }
}

#[cfg(feature = "gpu")]
pub async fn sub_task_v3_gpu(
    remote_hash: Vec<u8>,
    raw_pubkey: Vec<u8>,
    bitwork: Bitwork,
    _dead_line: u128,
    _tx: Sender<ThreadResultGpu>,
    index: u32,
) -> ThreadResultGpu {
    #[allow(unused_assignments)]
    let mut ret = ThreadResultGpu {
        res: None,
        generated_nonce: 0,
        expired: true,
        index,
    };
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    let mut nonce = index + 99;

    loop {
        if SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            > _dead_line as u64
        {
            break;
        }

        let (tx, _start) = create_dod_tx(
            CreateDodTxDefault {
                nonce,
                time,
                remote_hash: remote_hash.clone(),
                raw_pubkey: raw_pubkey.clone(),
            },
            true,
        );

        let split_length = if bitwork.pre % 2 != 0 {
            bitwork.pre + 1
        } else {
            bitwork.pre
        };

        let actual_pre = hex::encode(remote_hash.clone())
            .split_at(split_length as usize)
            .0
            .to_string();

        let bw2 = easy_bitwork_2(&actual_pre, bitwork.pre, Some(bitwork.post_hex.clone())).unwrap();

        let gpu_config = read_config();

        match dod_gpu::miner::mine_gpu_single(gpu_config.cuda_cores, 256, index, gpu_config.batch_size, tx.clone(), bw2)
            .await
        {
            Ok(res) => {
                if res.expired {
                    nonce += 1;
                    continue;
                } else {
                    let num_bytes = hex::decode(res.op_bytes).unwrap();
                    let generated_nonce = res.total_hashes;

                    ret = ThreadResultGpu {
                        res: Some(MiningResultGpu {
                            num_bytes,
                            nonce,
                            time,
                        }),
                        generated_nonce,
                        expired: false,
                        index,
                    };
                    break;
                }
            }
            Err(_) => {
                ret = ThreadResultGpu {
                    res: None,
                    generated_nonce: u128::MAX - 1,
                    expired: true,
                    index,
                };
                break;
            }
        }
    }
    ret.clone()
}

#[cfg(test)]
mod test {
    use crate::miner::multi_run_v3;
    use dod_utils::bitwork::Bitwork;
    use std::time::SystemTime;

    #[tokio::test]
    pub async fn multirun() {
        let remote_hash =
            hex::decode("98799b250c911fe0df86cd59066e329d93bfb3d35fa57cdd3b243e2a8eec1b45")
                .unwrap();
        let raw_pubkey =
            hex::decode("02aa7360476d762b5a88df8db5ad2aabdf2656c3f64a5a9d3c0962541575916917")
                .unwrap();
        let _se = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let res = multi_run_v3(
            Bitwork {
                pre: 6,
                post_hex: "0".to_string(),
            },
            remote_hash,
            raw_pubkey,
            Some(10),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                + 3_000_000_000u128,
        )
        .await;
        println!("{:?}", res);
    }
}
