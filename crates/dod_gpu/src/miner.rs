use crate::gpu::run_gpu_single;
use crate::gpu::GpuSingleJob;
use crate::types::BitworkResultPayload;
use crate::types::{GpuJsonStruct, GpuMiningResult};
use dod_utils::mine::BitworkResult2;

pub async fn mine_gpu_single(
    threads: u32,
    blocks: u32,
    gpu: u32,
    per_batch_size: u64,
    tx: Vec<u8>,
    bitwork_result: BitworkResult2,
) -> Result<GpuMiningResult, String> {
    let json_struct = GpuJsonStruct {
        expect_tx_bytes: tx.clone(),
        expect_tx_size: tx.clone().len() as u32, // expect_tx_size,
        op_bytes_start: 101,
        bytes_body_len: bitwork_result.prefix.len() as u32,
        result: BitworkResultPayload {
            prefix: hex::encode(pad_right_zero(bitwork_result.clone().prefix, 32)),
            k: bitwork_result.k,
            len: bitwork_result.len as u32,
        }, // r.bitwork_compare.clone()
    };

    let gpu_single_job = GpuSingleJob {
        json: json_struct.to_json().map_err(|e| e.to_string()).unwrap(),
        threads: threads as u16,
        blocks: blocks as u16,
        gpu: gpu as u16,
        job_size: per_batch_size,
    };

    match run_gpu_single(gpu_single_job).await {
        Ok(r) => GpuMiningResult::from_json(&r).map_err(|_| "Can not decode from json".to_string()),
        Err(e) => Err(e),
    }
}

pub fn pad_right_zero(des: Vec<u8>, final_length: usize) -> Vec<u8> {
    if final_length <= des.len() {
        return des;
    }
    let mut new_vec = vec![0u8; final_length];
    new_vec[0..des.len()].copy_from_slice(&des[..]);
    new_vec
}

#[cfg(test)]
mod test {

    use crate::miner::mine_gpu_single;
    use dod_utils::bitwork::{compare_bitwork_range, Bitwork};
    use dod_utils::mine::BitworkResult2;
    use dod_utils::sha256d;
    
    #[tokio::test]
    pub async fn t() {
        let  tx = hex::decode("010000000159d0e915ea1d5d2e1feb78cb29f0548c0fb7f7c37d72aa6e237f6fb57e0eac5d0000000000ffffffff02b0040000000000002251200695d063e642f5024155eff4afb9d73df90b80fdb0d68deab2d7e7de46c163260000000000000000126a109d4b1212d0c917e668e55bbeb5eda71700000000").unwrap();

        println!("sha256 {:?}", hex::encode(dod_utils::sha256(&tx)));
        println!("sha256d {:?}", hex::encode(sha256d(&tx)));

        // println!("full {:?}", hex::encode(sha256(&tx)));
        let threads = 1800;
        let blocks = 256;
        let gpu = 0;
        let per_batch_size = 960_000_000_000;
        let bitwork = BitworkResult2 {
            prefix: vec![0xc2, 0x99, 0x90, 0x0f],
            len: 7,
            k: 15,
        };
        let rs = mine_gpu_single(
            threads,
            blocks,
            gpu,
            per_batch_size,
            tx.clone(),
            bitwork.clone(),
        )
        .await;

        println!("mine_gpu_single result , {:?}", rs.clone());

        if rs.clone().is_ok() && rs.clone().unwrap().expired == false {
            let op_bytes = hex::decode(rs.clone().unwrap().op_bytes).unwrap();
            let mut tx_clone = tx.clone();
            tx_clone[101..117].copy_from_slice(&op_bytes);
            let mut s = sha256d(&tx_clone);

            let mut ok = false;
            if compare_bitwork_range(&s, &bitwork.prefix, bitwork.len, bitwork.k) {
                ok = true;
            }
            s.reverse();
            println!("tx clone: {:?}", hex::encode(s.clone()));
            println!("ok {:?} \n", ok);
        }
    }
}
