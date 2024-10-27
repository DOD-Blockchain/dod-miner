#ifndef SHA256_H
#define SHA256_H

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32 // SHA256 outputs a 32 byte digest

#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))

#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

#define checkCudaErrors(x)                                                    \
    {                                                                         \
        cudaGetLastError();                                                   \
        x;                                                                    \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess)                                               \
            printf("GPU: cudaError %d (%s)\n", err, cudaGetErrorString(err)); \
    }
/**************************** DATA TYPES ****************************/
// typedef unsigned char BYTE; // 8-bit byte
typedef uint32_t WORD;      // 32-bit word, change to "long" for 16-bit machines

typedef struct JOB
{
    BYTE *data;
    unsigned long long size;
    BYTE digest[64];
    char fname[128];
} JOB;

typedef struct
{
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} SHA256_CTX;

__constant__ WORD dev_k[64];

static const WORD host_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

/*********************** FUNCTION DECLARATIONS **********************/
char *print_sha(BYTE *buff);
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);

char *hash_to_string(BYTE *buff)
{
    char *string = (char *)malloc(70);
    int k, i;
    for (i = 0, k = 0; i < 32; i++, k += 2)
    {
        sprintf(string + k, "%.2x", buff[i]);
        // printf("%02x", buff[i]);
    }
    string[64] = 0;
    return string;
}

void print_job(JOB *j)
{
    printf("%s  %s\n", hash_to_string(j->digest), j->fname);
}

void print_jobs(JOB **jobs, int n)
{
    for (int i = 0; i < n; i++)
    {
        print_job(jobs[i]);
        // printf("@ %p JOB[%i] \n", jobs[i], i);
        // printf("\t @ 0x%p data = %x \n", jobs[i]->data, (jobs[i]->data == 0)? 0 : jobs[i]->data[0]);
        // printf("\t @ 0x%p size = %llu \n", &(jobs[i]->size), jobs[i]->size);
        // printf("\t @ 0x%p fname = %s \n", &(jobs[i]->fname), jobs[i]->fname);
        // printf("\t @ 0x%p digest = %s \n------\n", jobs[i]->digest, hash_to_string(jobs[i]->digest));
    }
}

__device__ void mycpy12(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
}

__device__ void mycpy16(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
    d[3]=s[3];
}

__device__ void mycpy32(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
    d[3]=s[3];
    d[4]=s[4]; 
    d[5]=s[5];
    d[6]=s[6];
    d[7]=s[7];
}

__device__ void mycpy44(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
    d[3]=s[3];
    d[4]=s[4]; 
    d[5]=s[5];
    d[6]=s[6];
    d[7]=s[7];
    d[8]=s[8]; 
    d[9]=s[9];
    d[10]=s[10];
}

__device__ void mycpy48(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
    d[3]=s[3];
    d[4]=s[4]; 
    d[5]=s[5];
    d[6]=s[6];
    d[7]=s[7];
    d[8]=s[8]; 
    d[9]=s[9];
    d[10]=s[10];
    d[11]=s[11];
}

__device__ void mycpy64(uint32_t *d, const uint32_t *s)
{
    d[0]=s[0]; 
    d[1]=s[1];
    d[2]=s[2];
    d[3]=s[3];
    d[4]=s[4]; 
    d[5]=s[5];
    d[6]=s[6];
    d[7]=s[7];
    d[8]=s[8]; 
    d[9]=s[9];
    d[10]=s[10];
    d[11]=s[11];
    d[12]=s[12]; 
    d[13]=s[13];
    d[14]=s[14];
    d[15]=s[15];
}

__device__ void copyRandom16(BYTE *d, const BYTE *s, const int op_bytes_start)
{
    d[0]=s[0+op_bytes_start]; 
    d[1]=s[1+op_bytes_start];
    d[2]=s[2+op_bytes_start];
    d[3]=s[3+op_bytes_start];
    d[4]=s[4+op_bytes_start]; 
    d[5]=s[5+op_bytes_start];
    d[6]=s[6+op_bytes_start];
    d[7]=s[7+op_bytes_start];
    d[8]=s[8+op_bytes_start]; 
    d[9]=s[9+op_bytes_start];
    d[10]=s[10+op_bytes_start];
    d[11]=s[11+op_bytes_start];
    d[12]=s[12+op_bytes_start]; 
    d[13]=s[13+op_bytes_start];
    d[14]=s[14+op_bytes_start];
    d[15]=s[15+op_bytes_start];
}



__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
    WORD a, b, c, d, e, f, g, h, i, t1, t2, m[64];
    // WORD S[8];

    // mycpy32(S, ctx->state);

        m[0] = (data[0] << 24) | (data[0 + 1] << 16) | (data[0 + 2] << 8) | (data[0 + 3]);
        m[1] = (data[4] << 24) | (data[4 + 1] << 16) | (data[4 + 2] << 8) | (data[4 + 3]);
        m[2] = (data[8] << 24) | (data[8 + 1] << 16) | (data[8 + 2] << 8) | (data[8 + 3]);
        m[3] = (data[12] << 24) | (data[12 + 1] << 16) | (data[12 + 2] << 8) | (data[12 + 3]);
        m[4] = (data[16] << 24) | (data[16 + 1] << 16) | (data[16 + 2] << 8) | (data[16 + 3]);
        m[5] = (data[20] << 24) | (data[20 + 1] << 16) | (data[20 + 2] << 8) | (data[20 + 3]);
        m[6] = (data[24] << 24) | (data[24 + 1] << 16) | (data[24 + 2] << 8) | (data[24 + 3]);
        m[7] = (data[28] << 24) | (data[28 + 1] << 16) | (data[28 + 2] << 8) | (data[28 + 3]);
        m[8] = (data[32] << 24) | (data[32 + 1] << 16) | (data[32 + 2] << 8) | (data[32 + 3]);
        m[9] = (data[36] << 24) | (data[36 + 1] << 16) | (data[36 + 2] << 8) | (data[36 + 3]);
        m[10] = (data[40] << 24) | (data[40 + 1] << 16) | (data[40 + 2] << 8) | (data[40 + 3]);
        m[11] = (data[44] << 24) | (data[44 + 1] << 16) | (data[44 + 2] << 8) | (data[44 + 3]);
        m[12] = (data[48] << 24) | (data[48 + 1] << 16) | (data[48 + 2] << 8) | (data[48 + 3]);
        m[13] = (data[52] << 24) | (data[52 + 1] << 16) | (data[52 + 2] << 8) | (data[52 + 3]);
        m[14] = (data[56] << 24) | (data[56 + 1] << 16) | (data[56 + 2] << 8) | (data[56 + 3]);
        m[15] = (data[60] << 24) | (data[60 + 1] << 16) | (data[60 + 2] << 8) | (data[60 + 3]);
        m[16] = SIG1(m[16 - 2]) + m[16 - 7] + SIG0(m[16 - 15]) + m[16 - 16];
        m[17] = SIG1(m[17 - 2]) + m[17 - 7] + SIG0(m[17 - 15]) + m[17 - 16];
        m[18] = SIG1(m[18 - 2]) + m[18 - 7] + SIG0(m[18 - 15]) + m[18 - 16];
        m[19] = SIG1(m[19 - 2]) + m[19 - 7] + SIG0(m[19 - 15]) + m[19 - 16];
        m[20] = SIG1(m[20 - 2]) + m[20 - 7] + SIG0(m[20 - 15]) + m[20 - 16];
        m[21] = SIG1(m[21 - 2]) + m[21 - 7] + SIG0(m[21 - 15]) + m[21 - 16];
        m[22] = SIG1(m[22 - 2]) + m[22 - 7] + SIG0(m[22 - 15]) + m[22 - 16];
        m[23] = SIG1(m[23 - 2]) + m[23 - 7] + SIG0(m[23 - 15]) + m[23 - 16];
        m[24] = SIG1(m[24 - 2]) + m[24 - 7] + SIG0(m[24 - 15]) + m[24 - 16];
        m[25] = SIG1(m[25 - 2]) + m[25 - 7] + SIG0(m[25 - 15]) + m[25 - 16];
        m[26] = SIG1(m[26 - 2]) + m[26 - 7] + SIG0(m[26 - 15]) + m[26 - 16];
        m[27] = SIG1(m[27 - 2]) + m[27 - 7] + SIG0(m[27 - 15]) + m[27 - 16];
        m[28] = SIG1(m[28 - 2]) + m[28 - 7] + SIG0(m[28 - 15]) + m[28 - 16];
        m[29] = SIG1(m[29 - 2]) + m[29 - 7] + SIG0(m[29 - 15]) + m[29 - 16];
        m[30] = SIG1(m[30 - 2]) + m[30 - 7] + SIG0(m[30 - 15]) + m[30 - 16];
        m[31] = SIG1(m[31 - 2]) + m[31 - 7] + SIG0(m[31 - 15]) + m[31 - 16];
        m[32] = SIG1(m[32 - 2]) + m[32 - 7] + SIG0(m[32 - 15]) + m[32 - 16];
        m[33] = SIG1(m[33 - 2]) + m[33 - 7] + SIG0(m[33 - 15]) + m[33 - 16];
        m[34] = SIG1(m[34 - 2]) + m[34 - 7] + SIG0(m[34 - 15]) + m[34 - 16];
        m[35] = SIG1(m[35 - 2]) + m[35 - 7] + SIG0(m[35 - 15]) + m[35 - 16];
        m[36] = SIG1(m[36 - 2]) + m[36 - 7] + SIG0(m[36 - 15]) + m[36 - 16];
        m[37] = SIG1(m[37 - 2]) + m[37 - 7] + SIG0(m[37 - 15]) + m[37 - 16];
        m[38] = SIG1(m[38 - 2]) + m[38 - 7] + SIG0(m[38 - 15]) + m[38 - 16];
        m[39] = SIG1(m[39 - 2]) + m[39 - 7] + SIG0(m[39 - 15]) + m[39 - 16];
        m[40] = SIG1(m[40 - 2]) + m[40 - 7] + SIG0(m[40 - 15]) + m[40 - 16];
        m[41] = SIG1(m[41 - 2]) + m[41 - 7] + SIG0(m[41 - 15]) + m[41 - 16];
        m[42] = SIG1(m[42 - 2]) + m[42 - 7] + SIG0(m[42 - 15]) + m[42 - 16];
        m[43] = SIG1(m[43 - 2]) + m[43 - 7] + SIG0(m[43 - 15]) + m[43 - 16];
        m[44] = SIG1(m[44 - 2]) + m[44 - 7] + SIG0(m[44 - 15]) + m[44 - 16];
        m[45] = SIG1(m[45 - 2]) + m[45 - 7] + SIG0(m[45 - 15]) + m[45 - 16];
        m[46] = SIG1(m[46 - 2]) + m[46 - 7] + SIG0(m[46 - 15]) + m[46 - 16];
        m[47] = SIG1(m[47 - 2]) + m[47 - 7] + SIG0(m[47 - 15]) + m[47 - 16];
        m[48] = SIG1(m[48 - 2]) + m[48 - 7] + SIG0(m[48 - 15]) + m[48 - 16];
        m[49] = SIG1(m[49 - 2]) + m[49 - 7] + SIG0(m[49 - 15]) + m[49 - 16];
        m[50] = SIG1(m[50 - 2]) + m[50 - 7] + SIG0(m[50 - 15]) + m[50 - 16];
        m[51] = SIG1(m[51 - 2]) + m[51 - 7] + SIG0(m[51 - 15]) + m[51 - 16];
        m[52] = SIG1(m[52 - 2]) + m[52 - 7] + SIG0(m[52 - 15]) + m[52 - 16];
        m[53] = SIG1(m[53 - 2]) + m[53 - 7] + SIG0(m[53 - 15]) + m[53 - 16];
        m[54] = SIG1(m[54 - 2]) + m[54 - 7] + SIG0(m[54 - 15]) + m[54 - 16];
        m[55] = SIG1(m[55 - 2]) + m[55 - 7] + SIG0(m[55 - 15]) + m[55 - 16];
        m[56] = SIG1(m[56 - 2]) + m[56 - 7] + SIG0(m[56 - 15]) + m[56 - 16];
        m[57] = SIG1(m[57 - 2]) + m[57 - 7] + SIG0(m[57 - 15]) + m[57 - 16];
        m[58] = SIG1(m[58 - 2]) + m[58 - 7] + SIG0(m[58 - 15]) + m[58 - 16];
        m[59] = SIG1(m[59 - 2]) + m[59 - 7] + SIG0(m[59 - 15]) + m[59 - 16];
        m[60] = SIG1(m[60 - 2]) + m[60 - 7] + SIG0(m[60 - 15]) + m[60 - 16];
        m[61] = SIG1(m[61 - 2]) + m[61 - 7] + SIG0(m[61 - 15]) + m[61 - 16];
        m[62] = SIG1(m[62 - 2]) + m[62 - 7] + SIG0(m[62 - 15]) + m[62 - 16];
        m[63] = SIG1(m[63 - 2]) + m[63 - 7] + SIG0(m[63 - 15]) + m[63 - 16];


    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    #pragma unroll 64
    for (i = 0; i < 64; ++i)
    {
        t1 = h + EP1(e) + CH(e, f, g) + dev_k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void sha256_init(SHA256_CTX *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
    WORD i;

    // for each byte in message
    for (i = 0; i < len; ++i)
    {
        // ctx->data == message 512 bit chunk
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64)
        {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
    WORD i;

    i = ctx->datalen;

    // Pad whatever data is left in the buffer.
    if (ctx->datalen < 56)
    {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    }
    else
    {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    sha256_transform(ctx, ctx->data);

    // Since this implementation uses little endian byte ordering and SHA uses big endian,
    // reverse all the bytes when copying the final state to the output hash.
    hash[0] = (ctx->state[0] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 4] = (ctx->state[1] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 8] = (ctx->state[2] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 12] = (ctx->state[3] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 16] = (ctx->state[4] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 20] = (ctx->state[5] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 24] = (ctx->state[6] >> (24 - 0 * 8)) & 0x000000ff;
    hash[0 + 28] = (ctx->state[7] >> (24 - 0 * 8)) & 0x000000ff;
    hash[1] = (ctx->state[0] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 4] = (ctx->state[1] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 8] = (ctx->state[2] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 12] = (ctx->state[3] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 16] = (ctx->state[4] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 20] = (ctx->state[5] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 24] = (ctx->state[6] >> (24 - 1 * 8)) & 0x000000ff;
    hash[1 + 28] = (ctx->state[7] >> (24 - 1 * 8)) & 0x000000ff;
    hash[2] = (ctx->state[0] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 4] = (ctx->state[1] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 8] = (ctx->state[2] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 12] = (ctx->state[3] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 16] = (ctx->state[4] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 20] = (ctx->state[5] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 24] = (ctx->state[6] >> (24 - 2 * 8)) & 0x000000ff;
    hash[2 + 28] = (ctx->state[7] >> (24 - 2 * 8)) & 0x000000ff;
    hash[3] = (ctx->state[0] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 4] = (ctx->state[1] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 8] = (ctx->state[2] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 12] = (ctx->state[3] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 16] = (ctx->state[4] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 20] = (ctx->state[5] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 24] = (ctx->state[6] >> (24 - 3 * 8)) & 0x000000ff;
    hash[3 + 28] = (ctx->state[7] >> (24 - 3 * 8)) & 0x000000ff;    
}

#endif // SHA256_H