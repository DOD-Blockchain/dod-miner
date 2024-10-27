use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuMiningResult {
    pub hash_count: u128,
    pub op_bytes: String,
    pub expired: bool,
    pub used_time: u32,
    pub total_hashes: u128,
    pub hash_rate: u64,
}

impl GpuMiningResult {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// {
//     "expect_tx_bytes": "0100000001390f7392bb03aae07c26c782213d53edea749952311b6ee075dca30ffe86cccc0100000000fdffffff033404000000000000225120f954e58748b07a201134f664116945b442c409b8501b26048d2cd4cc3245367f0000000000000000126a109d4b1212d0c917e668e55bbeb5eda717d30600000000000022512011b6ce99eab0d8873d787e99e68a351358228893cdf1049ac48aae51391598ab00000000",
//     "expect_tx_size": 164,
//     "op_bytes_start": 101,
//     "baseSeed": 0,
//     "compare": {
//       "bytes_body": "1ba90000",
//       "remain": null,
//       "actual_length": 4,
//       "ext": null
//     }
//   }

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize, Hash)]
pub struct BitworkResultPayload {
    pub prefix: String,
    pub len: u32,
    pub k: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuJsonStruct {
    #[serde(with = "hex::serde")]
    pub expect_tx_bytes: Vec<u8>,
    pub expect_tx_size: u32,
    pub op_bytes_start: u32,
    pub bytes_body_len: u32,
    pub result: BitworkResultPayload,
}

impl GpuJsonStruct {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
