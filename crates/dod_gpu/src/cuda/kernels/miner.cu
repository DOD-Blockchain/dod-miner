#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <boost/algorithm/hex.hpp>

#include "../includes/miner.h"
// #include "../includes/sha256.cuh"
#include "../includes/sha256d.cuh"


// utils
void pre_sha256()
{
    cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
}

__device__ unsigned long deviceRandomGen(unsigned long x)
{
    x ^= (x << 21);
    x ^= (x >> 35);
    x ^= (x << 4);
    return x;
}

void hostRandomGen(unsigned long *x)
{
    *x ^= (*x << 21);
    *x ^= (*x >> 35);
    *x ^= (*x << 4);
}

__global__ void initSolutionMemory(int *blockContainsSolution)
{
    *blockContainsSolution = -1;
}


__device__ void randomBytes16(const unsigned long idx, const unsigned long idy, BYTE *output)
{

    output[0] = (idy >> (8 * 0)) & 0xFF;
    output[1] = (idy >> (8 * 1)) & 0xFF;
    output[2] = (idy >> (8 * 2)) & 0xFF;
    output[3] = (idy >> (8 * 3)) & 0xFF;
    output[4] = (idy >> (8 * 4)) & 0xFF;
    output[5] = (idy >> (8 * 5)) & 0xFF;
    output[6] = (idy >> (8 * 6)) & 0xFF;
    output[7] = (idy >> (8 * 7)) & 0xFF;
    output[8] = (idx >> (8 * 0)) & 0xFF;
    output[9] = (idx >> (8 * 1)) & 0xFF;
    output[10] = (idx >> (8 * 2)) & 0xFF;
    output[11] = (idx >> (8 * 3)) & 0xFF;
    output[12] = (idx >> (8 * 4)) & 0xFF;
    output[13] = (idx >> (8 * 5)) & 0xFF;
    output[14] = (idx >> (8 * 6)) & 0xFF;
    output[15] = (idx >> (8 * 7)) & 0xFF;
}

__device__ bool compare_bitwork_range(BYTE *a, BYTE *b, int n, uint8_t k) {
    for (int i = 0; i < n / 2; i++) {
        if (a[31 - i] != b[i]) {
            return false;
        }
    }

    if (n % 2 != 0) {
        uint8_t last_nibble = (a[31 - n / 2] & 0xF0) >> 4;
        if (last_nibble != (b[n / 2] & 0xF0) >> 4) {
            return false;
        }
    }

    uint8_t next_nibble;
    if (n % 2 == 0) {
        next_nibble = (a[31 - n / 2] & 0xF0) >> 4;
    } else {
        next_nibble = a[31 - n / 2] & 0x0F;
    }

    return next_nibble >= k;
}


__global__ __launch_bounds__(1024) void miner(
    unsigned long baseSeed,
    BYTE *_solution,
    int *_blockContainsSolution,
    uint32_t *c_midstate,
    uint32_t *c_dataEnd,
    BYTE *bytes_body,
    int len, 
    int k)
{
    int tid = threadIdx.x;
    if(tid < BLOCKS){
        BYTE composed_hash[32];
        BYTE random_bytes[16];
    
        uint32_t dat_1[16] = {0};
    
        uint32_t dat_2[16] = {0};
        dat_2[15] = 0x3c8;
        uint32_t dat_3[16] = {0};
        dat_3[8] = 0x80000000;
        dat_3[15] = 0x00000100;
    
        uint32_t buf_3[8];
        AS_UINT2(&buf_3[0]) = __ldg(reinterpret_cast<const uint2*>(c_H256));
        AS_UINT2(&buf_3[2]) = __ldg(reinterpret_cast<const uint2*>(c_H256 + 2));
        AS_UINT2(&buf_3[4]) = __ldg(reinterpret_cast<const uint2*>(c_H256 + 4));
        AS_UINT2(&buf_3[6]) = __ldg(reinterpret_cast<const uint2*>(c_H256 + 6));
    
        uint32_t buf[8];
        AS_UINT2(&buf[0]) = __ldg(reinterpret_cast<const uint2*>(c_midstate));
        AS_UINT2(&buf[2]) = __ldg(reinterpret_cast<const uint2*>(c_midstate + 2));
        AS_UINT2(&buf[4]) = __ldg(reinterpret_cast<const uint2*>(c_midstate + 4));
        AS_UINT2(&buf[6]) = __ldg(reinterpret_cast<const uint2*>(c_midstate + 6));


        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned long seedx = deviceRandomGen(baseSeed) + (unsigned long)idx;
        unsigned long seedy = deviceRandomGen(baseSeed) + (unsigned long)idy;
        randomBytes16(seedx, seedy, random_bytes);

        reinterpret_cast<uint4*>(dat_1)[0] = __ldg(reinterpret_cast<const uint4*>(c_dataEnd));
        reinterpret_cast<uint4*>(dat_1)[1] = __ldg(reinterpret_cast<const uint4*>(c_dataEnd + 4));
        reinterpret_cast<uint4*>(dat_1)[2] = __ldg(reinterpret_cast<const uint4*>(c_dataEnd + 8));
        dat_1[9] = ((c_dataEnd[9] >> 24) & 0xFF) << 24 | ((random_bytes[0] & 0xFF) << 16) | ((random_bytes[1] & 0xFF) << 8) | (random_bytes[2] & 0xFF);
        dat_1[10] = ((random_bytes[3] & 0xFF) << 24) | ((random_bytes[4] & 0xFF) << 16) | ((random_bytes[5] & 0xFF) << 8) | (random_bytes[6] & 0xFF);
        dat_1[11] = ((random_bytes[7] & 0xFF) << 24) | ((random_bytes[8] & 0xFF) << 16) | ((random_bytes[9] & 0xFF) << 8) | (random_bytes[10] & 0xFF);
        dat_1[12] = ((random_bytes[11] & 0xFF) << 24) | ((random_bytes[12] & 0xFF) << 16) | ((random_bytes[13] & 0xFF) << 8) | (random_bytes[14] & 0xFF);
        dat_1[13] = ((random_bytes[15] & 0xFF) << 24) | ((c_dataEnd[13] >> 16) & 0xFF) << 16 | ((c_dataEnd[13] >> 8) & 0xFF) << 8 | (c_dataEnd[13] & 0xFF);
        dat_1[14] = ((c_dataEnd[14] >> 24) & 0xFF) << 24 | (0x80 << 16);
       
        // sha starts
        sha256_round_body(dat_1, buf, c_K);
        sha256_round_body(dat_2, buf, c_K);

        // SHA256 d
        memcpy(dat_3, buf, sizeof(uint32_t) * 8);
        sha256_round_last(dat_3, buf_3, c_K);
        uint32_to_bytes(buf_3, composed_hash, 8);

        int blockContainsSolution = compare_bitwork_range(composed_hash, bytes_body, len, (uint8_t)k);

        if (blockContainsSolution == 1)
        {
            atomicExch(_blockContainsSolution, 1);
            memcpy(_solution, random_bytes, 16);
        }
    }
    
}



std::tuple<bool, BYTE *> launchOneJob(
    void *vargp 
)
{
    
    uint32_t endiandata[31];
    
    MinerArgs *hi = (MinerArgs *)vargp;

    cudaSetDevice(hi->device);

    pre_sha256();

    // compare struct copy
    BYTE *bytes_body = (BYTE *)malloc(sizeof(BYTE) * TX_ID_LEN);
    for (int i = 0; i < TX_ID_LEN; i++)
    {
        bytes_body[i] = hi->bitwork.prefix[i];
    }
    BYTE *d_bytes_body;
    cudaMalloc(&d_bytes_body, sizeof(BYTE) * TX_ID_LEN);
    cudaMemcpy(d_bytes_body, bytes_body, sizeof(BYTE) * TX_ID_LEN, cudaMemcpyHostToDevice);
    
    
     uint8_to_uint32(hi->expect_tx_bytes, endiandata, TX_BYTES_LEN);

     uint32_t in[16];
     uint32_t buf[8];
     uint32_t end[15];
     for (int i=0;i<16;i++) in[i] = endiandata[i]; //cuda_swab32(endiandata[i]);
     for (int i=0;i<8;i++) buf[i] = cpu_H256[i];
     for (int i=0;i<15;i++) end[i] = endiandata[16+i];// cuda_swab32(endiandata[16+i]);
     sha256_round_body_host(in, buf, cpu_K);


     uint32_t *d_midstate;
     cudaMalloc(&d_midstate, sizeof(uint32_t) * 8);
     cudaMemcpy(d_midstate, buf, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);
     
     uint32_t *d_dataEnd;
     cudaMalloc(&d_dataEnd, sizeof(uint32_t) * 15);
     cudaMemcpy(d_dataEnd, end, sizeof(uint32_t) * 15, cudaMemcpyHostToDevice);

     
    int d_len = hi->bitwork.len;
    int d_k = hi->bitwork.k;

    BYTE *_blockSolution = (BYTE *)malloc(sizeof(BYTE)* RANDOM_LEN);
    BYTE *d_solution;
    cudaMalloc(&d_solution, sizeof(BYTE)* RANDOM_LEN);

    int *_blockContainsSolution = (int *)malloc(sizeof(int));
    int *d_blockContainsSolution;
    cudaMalloc(&d_blockContainsSolution, sizeof(int));

    unsigned long rngSeed = timems();

    initSolutionMemory<<<1, 1>>>(d_blockContainsSolution);

    bool _expired = false;

    int blockSize = 0;   // The launch configurator returned block size
    int minGridSize = 0; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize = 0;    // The actual grid size needed, based on input size

   
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, miner, 0, PER_JOB_SIZE);
    // Round up according to array size
    gridSize = (PER_JOB_SIZE + blockSize - 1) / blockSize;

    while (1)
    {
        hostRandomGen(&rngSeed);
        hi->hashesProcessed += gridSize * BLOCKS;
        miner<<<gridSize, blockSize, BLOCKS>>>(
            rngSeed,
            d_solution,
            d_blockContainsSolution,
            d_midstate,
            d_dataEnd,
            d_bytes_body,
            d_len,
            d_k
            );
        cudaDeviceSynchronize();
        cudaMemcpy(_blockContainsSolution, d_blockContainsSolution, sizeof(int), cudaMemcpyDeviceToHost);
        if (*_blockContainsSolution == 1)
        {
            cudaMemcpy(_blockSolution, d_solution, sizeof(BYTE) * RANDOM_LEN, cudaMemcpyDeviceToHost);
            solution_device = hi->device;
           break;
        }
        if (hi->hashesProcessed >= hi->jobSize)
        {
            _expired = true;
           break;
        }
    }
    cudaDeviceReset();
    return std::make_tuple(_expired, _blockSolution);
    
}



// json tools

void hexStringToBytes(const std::string &hex, int *bytes, int maxLen)
{
    std::vector<unsigned char> tempBytes;
    boost::algorithm::unhex(hex.begin(), hex.end(), std::back_inserter(tempBytes));

    int numBytes = std::min(static_cast<int>(tempBytes.size()), maxLen);
    for (int i = 0; i < numBytes; ++i)
    {
        bytes[i] = static_cast<int>(tempBytes[i]);
    }
}


void readJsonStringAndAssign(const std::string &jsonString, MinerArgs &minerArgs, BitWorkResult &bitWorkResult)
{
    nlohmann::json j = nlohmann::json::parse(jsonString);

    minerArgs.expect_tx_size = j["expect_tx_size"];
    minerArgs.op_bytes_start = j["op_bytes_start"];
    minerArgs.device = 0;
    minerArgs.hashesProcessed = 0;
    minerArgs.bytes_body_len = j["bytes_body_len"];
    std::string hexString = j["expect_tx_bytes"];
    hexStringToBytes(hexString, minerArgs.expect_tx_bytes, minerArgs.expect_tx_size);

    bitWorkResult.k = j["result"]["k"].is_null() ? 0 : j["result"]["k"].get<int>();
    bitWorkResult.len = j["result"]["len"].is_null() ? 0 : j["result"]["len"].get<int>();
    std::string prefix = j["result"]["prefix"];
    

    hexStringToBytes(prefix, bitWorkResult.prefix, TX_ID_LEN);
    minerArgs.bitwork = bitWorkResult;
}


std::string bytesToHexString(BYTE *bytes, size_t length)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < length; ++i)
    {
        ss << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return ss.str();
}

void saveJson(std::string filePath, std::string op_bytes, unsigned long hash_count, long long used_time, unsigned long total_hashes)
{
    nlohmann::json j;
    j["op_bytes"] = op_bytes;
    j["hash_count"] = hash_count;
    j["used_time"] = used_time;
    j["total_hashes"] = total_hashes;
    j["hash_rate"] = (unsigned long)((double)total_hashes / (double)used_time) * 1000;

    std::ofstream file(filePath);

    file << j.dump(4);

    file.close();
}

std::string saveJsonString(std::string op_bytes, unsigned long hash_count, long long used_time, unsigned long total_hashes, bool is_expired)
{
    long long _used_time = used_time < 1000 ? 1000 : used_time;
    nlohmann::json j;
    j["expired"] = is_expired;
    j["op_bytes"] = op_bytes;
    j["hash_count"] = hash_count;
    j["used_time"] = _used_time;
    j["total_hashes"] = total_hashes;
    j["hash_rate"] = (unsigned long)((double)total_hashes / (double)_used_time) * 1000;

    return j.dump(4);
}


extern "C" int getGpus()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

extern "C" int whichBusy()
{
    int cpu;
    cudaGetDevice(&cpu);
    return cpu;
}

extern "C" char *singleRun(
    int threads,
    int blocks,
    int gpu,
    unsigned long jobSize,
    // int isSizeCapped,
    const char *inputJson
){
    MinerArgs minerArgs;
    BitWorkResult bitWorkResult;
    std::string jsonString(inputJson);
    readJsonStringAndAssign(jsonString, minerArgs, bitWorkResult);
    
    setlocale(LC_NUMERIC, "");

    // pthread_mutex_init(&solutionLock, NULL);
    // pthread_mutex_lock(&solutionLock);

    unsigned long **processedPtrs = (unsigned long **)malloc(sizeof(unsigned long *) * 1);
    pthread_t *tids = (pthread_t *)malloc(sizeof(pthread_t) * 1);

    long long start = timems();

    MinerArgs *hi = (MinerArgs *)malloc(sizeof(MinerArgs));
    hi->device = gpu;
    hi->hashesProcessed = 0;
    hi->expect_tx_size = minerArgs.expect_tx_size;
    hi->op_bytes_start = minerArgs.op_bytes_start;
    hi->bytes_body_len = minerArgs.bytes_body_len;
    hi->jobSize = jobSize;
    memcpy(hi->expect_tx_bytes, minerArgs.expect_tx_bytes, sizeof(int) * minerArgs.expect_tx_size);
    hi->bitwork = bitWorkResult;
    hi->blocks = blocks;
    hi->threads = threads;
    processedPtrs[0] = &hi->hashesProcessed;
    // pthread_create(tids + gpu, NULL, launchOneJob, hi);
    usleep(10);

    std::tuple<bool,  BYTE *> result = launchOneJob(hi);

    bool _expired = std::get<0>(result);
    BYTE *_solution = std::get<1>(result);

    // pthread_mutex_lock(&solutionLock);
    long long end = timems();

    long long elapsed = end - start;

    if (elapsed < 1000)
    {
        elapsed = 1000;
    }

    unsigned long totalProcessed = 0;
    for (int i = 0; i < 1; i++)
    {
        totalProcessed += *(processedPtrs[i]);
    }
    // printf("Hashes (%'lu) Seconds (%'f) Hashes/sec (%'lu)\r", totalProcessed, ((float)elapsed) / 1000.0, (unsigned long)((double)totalProcessed / (double)elapsed) * 1000);
    // printf("\n");
    // pthread_join(tids[gpu], NULL);
    std::string solutionString = _expired ? "" :  bytesToHexString(_solution, 16);
    // if (_expired == true)
    // {
    //     printf("expired \n");
    // }
    // else
    // {
    //     printf("hit \n");
    // }
    std::string resultJson = saveJsonString(solutionString, totalProcessed, elapsed, totalProcessed, _expired);

    char *cstr = new char[resultJson.length() + 1];
    std::strcpy(cstr, resultJson.c_str());
    return cstr;
}



