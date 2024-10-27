#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "/usr/local/cuda-12.6/include/cuda.h"
#include "/usr/local/cuda-12.6/include/curand_kernel.h"
#include <sys/time.h>
#include <pthread.h>
#include <locale.h>

// #include "sha256.cuh"

#define THREADS 1500
#define BLOCKS 256
#define GPUS 4
// #define STREAMS 1
#define RANDOM_LEN 16
#define TX_ID_LEN 32
#define TX_BYTES_LEN 121
// #define TX_BYTES_LEN_2 153
// #define TX_BYTES_LEN_3 152
// #define TX_BYTES_LEN_4 155
// #define TX_BYTES_REVEAL_LEN 121
#define PER_JOB_SIZE 2000000000
// #define DEFAULT_STREAM_SIZE 1000000000

#ifdef __cplusplus
extern "C" {
#endif 

typedef unsigned char BYTE; // 8-bit byte

struct BitWorkResult
{
    int prefix[TX_ID_LEN]; 
    int k;     
    int len;
};
typedef struct BitWorkResult BitWorkResult;

long long timems()
{
    struct timeval end;
    gettimeofday(&end, NULL);
    return end.tv_sec * 1000LL + end.tv_usec / 1000;
}


pthread_mutex_t solutionLock;
BYTE *solution;
bool expired = false;
int solution_device;

typedef struct
{
    int expect_tx_bytes[TX_BYTES_LEN];
    int expect_tx_size;
    int op_bytes_start;
    int bytes_body_len;
    BitWorkResult bitwork;
    int device;
    unsigned long hashesProcessed;
    int blocks;
    int threads;
    unsigned long jobSize;
} MinerArgs;




// __cplusplus

char* singleRun(
    int threads,
    int blocks,
    int gpu,
    unsigned long jobSize,
    // int isSizeCapped,
    const char* inputJson
);

int getGpus();

int whichBusy();

void freeString(char* str) {
    delete[] str;  // C++delete[]
};

#ifdef __cplusplus
}
#endif