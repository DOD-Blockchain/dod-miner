/*
 * sha256d CUDA implementation.
 * tpruvot 2017
 */

 #include <stdio.h>
 #include <stdint.h>
 #include <memory.h>
 
 #include "cuda_helper.h"

 
const __constant__  uint32_t __align__(8) c_H256[8] = {
     0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
     0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
 };
 __constant__ static uint32_t __align__(8) c_K[64];
 __constant__ static uint32_t __align__(8) c_target[2];
 __device__ uint64_t d_target[1];
 
 static uint32_t* d_resNonces[MAX_GPUS] = { 0 };
 
 // ------------------------------------------------------------------------------------------------
 
 static const uint32_t cpu_H256[8] = {
     0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
     0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
 };
 
 static const uint32_t cpu_K[64] = {
     0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
     0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
     0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
     0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
     0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
     0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
     0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
     0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
 };
 
 #define ROTR ROTR32
 
 __host__
 static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
     uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
     uint32_t in, const uint32_t Kshared)
 {
     uint32_t t1;
     uint32_t t2;
     uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
     uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
     uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
     uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
 
     t1 = h + bsg21 + vxandx + Kshared + in;
     t2 = bsg20 + andorv;
     d = d + t1;
     h = t1 + t2;
 }
 
 __host__
 static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
     uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
     uint32_t* in, uint32_t pc, const uint32_t Kshared)
 {
     uint32_t t1;
     uint32_t t2;
 
     int pcidx1 = (pc-2)  & 0xF;
     int pcidx2 = (pc-7)  & 0xF;
     int pcidx3 = (pc-15) & 0xF;
 
     uint32_t inx0 = in[pc];
     uint32_t inx1 = in[pcidx1];
     uint32_t inx2 = in[pcidx2];
     uint32_t inx3 = in[pcidx3];
 
     uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
     uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
     uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
     uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
     uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
     uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
 
     in[pc] = ssg21 + inx2 + ssg20 + inx0;
 
     t1 = h + bsg21 + vxandx + Kshared + in[pc];
     t2 = bsg20 + andorv;
     d =  d + t1;
     h = t1 + t2;
 }
 

 __host__
 static void sha256_round_body_host(uint32_t* in, uint32_t* state, const uint32_t* Kshared)
 {
     uint32_t a = state[0];
     uint32_t b = state[1];
     uint32_t c = state[2];
     uint32_t d = state[3];
     uint32_t e = state[4];
     uint32_t f = state[5];
     uint32_t g = state[6];
     uint32_t h = state[7];
 
     sha256_step1_host(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
     sha256_step1_host(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
     sha256_step1_host(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
     sha256_step1_host(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
     sha256_step1_host(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
     sha256_step1_host(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
     sha256_step1_host(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
     sha256_step1_host(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
     sha256_step1_host(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
     sha256_step1_host(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
     sha256_step1_host(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
     sha256_step1_host(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
     sha256_step1_host(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
     sha256_step1_host(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
     sha256_step1_host(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
     sha256_step1_host(b,c,d,e,f,g,h,a,in[15], Kshared[15]);
 
     for (int i=0; i<3; i++)
     {
         sha256_step2_host(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
         sha256_step2_host(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
         sha256_step2_host(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
         sha256_step2_host(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
         sha256_step2_host(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
         sha256_step2_host(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
         sha256_step2_host(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
         sha256_step2_host(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
         sha256_step2_host(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
         sha256_step2_host(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
         sha256_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
         sha256_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
         sha256_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
         sha256_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
         sha256_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
         sha256_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
     }
 
     state[0] += a;
     state[1] += b;
     state[2] += c;
     state[3] += d;
     state[4] += e;
     state[5] += f;
     state[6] += g;
     state[7] += h;
 }
 
 #if  (__CUDA_ARCH__ >= 500)
 // device asm for whirpool
 __device__ __forceinline__
 uint64_t xor3b(uint32_t a, uint32_t b, uint32_t c)
 {
        uint32_t d;
        asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
 }
 #else
 #define xor3b(a,b,c) (a ^ b ^ c)
 #endif


 #if  (__CUDA_ARCH__ >= 500)
 // device asm for whirpool
 __device__ __forceinline__
 uint64_t xandx32(uint32_t a, uint32_t b, uint32_t c)
 {
        uint32_t d;
        asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
 }
 #else
 #define xandx32(g,e,f) (g ^ (e & (f ^ g)))
 #endif


 // #define xor3b(a,b,c) (a ^ b ^ c)
 
 __device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
 {
     return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
 }
 
 __device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
 {
     return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
 }
 
 __device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
 {
     return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
 }
 
 __device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
 {
     return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
 }
 
 __device__ __forceinline__ uint32_t andor32(const uint32_t a, const uint32_t b, const uint32_t c)
 {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
 }

__device__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	uint32_t t1,t2;
	uint32_t vxandx = xandx32(g, e, f);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__device__
static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1,t2;

	int pcidx1 = (pc-2) & 0xF;
	int pcidx2 = (pc-7) & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ssg2_1(inx1);
	uint32_t ssg20 = ssg2_0(inx3);
	uint32_t vxandx = xandx32(g, e, f);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__device__
static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
	sha2_step1(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
	sha2_step1(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
	sha2_step1(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
	sha2_step1(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
	sha2_step1(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
	sha2_step1(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
	sha2_step1(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
	sha2_step1(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha2_step1(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha2_step1(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha2_step1(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha2_step1(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha2_step1(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	#pragma unroll 3
	for (int i=0; i<3; i++)
	{
		sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

__device__
static void sha256_round_last(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a,b,c,d, e,f,g,h, in[ 0], Kshared[ 0]);
	sha2_step1(h,a,b,c, d,e,f,g, in[ 1], Kshared[ 1]);
	sha2_step1(g,h,a,b, c,d,e,f, in[ 2], Kshared[ 2]);
	sha2_step1(f,g,h,a, b,c,d,e, in[ 3], Kshared[ 3]);
	sha2_step1(e,f,g,h, a,b,c,d, in[ 4], Kshared[ 4]);
	sha2_step1(d,e,f,g, h,a,b,c, in[ 5], Kshared[ 5]);
	sha2_step1(c,d,e,f, g,h,a,b, in[ 6], Kshared[ 6]);
	sha2_step1(b,c,d,e, f,g,h,a, in[ 7], Kshared[ 7]);
	sha2_step1(a,b,c,d, e,f,g,h, in[ 8], Kshared[ 8]);
	sha2_step1(h,a,b,c, d,e,f,g, in[ 9], Kshared[ 9]);
	sha2_step1(g,h,a,b, c,d,e,f, in[10], Kshared[10]);
	sha2_step1(f,g,h,a, b,c,d,e, in[11], Kshared[11]);
	sha2_step1(e,f,g,h, a,b,c,d, in[12], Kshared[12]);
	sha2_step1(d,e,f,g, h,a,b,c, in[13], Kshared[13]);
	sha2_step1(c,d,e,f, g,h,a,b, in[14], Kshared[14]);
	sha2_step1(b,c,d,e, f,g,h,a, in[15], Kshared[15]);

	#pragma unroll 2
	for (int i=0; i<2; i++)
	{
		sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*i]);
		sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*i]);
		sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*i]);
		sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*i]);
		sha2_step2(c,d,e,f, g,h,a,b, in,14, Kshared[30+16*i]);
		sha2_step2(b,c,d,e, f,g,h,a, in,15, Kshared[31+16*i]);
	}

	sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*2]);
	sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*2]);
	sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*2]);
	sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*2]);
	sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*2]);
	sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*2]);
	sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*2]);
	sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*2]);
	sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*2]);
	sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*2]);
	sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*2]);
	sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*2]);
	sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*2]);
	sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*2]);

	state[6] += g;
	state[7] += h;
}
 
 __device__ __forceinline__
 uint64_t cuda_swab32ll(uint64_t x) {
     return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
 }
 

 uint32_t byte_swap_32(uint32_t x) {
    return ((x >> 24) & 0x000000FF) | 
           ((x >> 8) & 0x0000FF00) | 
           ((x << 8) & 0x00FF0000) | 
           ((x << 24) & 0xFF000000);
}

__device__ uint32_t byte_swap_32_device(uint32_t x) {
    return ((x >> 24) & 0x000000FF) | 
           ((x >> 8) & 0x0000FF00) | 
           ((x << 8) & 0x00FF0000) | 
           ((x << 24) & 0xFF000000);
}



void uint8_to_uint32(int *input, uint32_t* output, size_t len) {
    size_t i;

    for (i = 0; i < len / 4; i++) {
        output[i] = ((uint32_t)input[i * 4 + 0] << 24) |
                    ((uint32_t)input[i * 4 + 1] << 16) |
                    ((uint32_t)input[i * 4 + 2] << 8)  |
                    ((uint32_t)input[i * 4 + 3] << 0);
    }

    size_t remaining_bytes = len % 4;
    if (remaining_bytes > 0) {
        uint32_t last_value = 0;  

        if (remaining_bytes == 1) {
            last_value |= ((uint32_t)input[i * 4 + 0] << 24);  
        } else if (remaining_bytes == 2) {
            last_value |= ((uint32_t)input[i * 4 + 0] << 24);  
            last_value |= ((uint32_t)input[i * 4 + 1] << 16);  
        } else if (remaining_bytes == 3) {
            last_value |= ((uint32_t)input[i * 4 + 0] << 24);  
            last_value |= ((uint32_t)input[i * 4 + 1] << 16);  
            last_value |= ((uint32_t)input[i * 4 + 2] << 8);   
        }

        output[i] = last_value;
    }
}

__device__ void uint8_to_uint32_device(uint8_t *input, uint32_t* output, size_t len) {
    for (size_t i = 0; i < len / 4; i++) {
        output[i] = ((uint32_t)input[i * 4 + 0] << 24) |
                    ((uint32_t)input[i * 4 + 1] << 16) |
                    ((uint32_t)input[i * 4 + 2] << 8)  |
                    ((uint32_t)input[i * 4 + 3] << 0);
        // output[i] = byte_swap_32_device(output[i]);
    }
}


__device__ void uint32_to_uint8(uint32_t* input, uint8_t* output, size_t len, size_t full_len) {
    for (size_t i = 0; i < full_len; i++) {
        if(i<len){
            output[i * 4 + 0] = (input[i] >> 24) & 0xFF;
            output[i * 4 + 1] = (input[i] >> 16) & 0xFF;
            output[i * 4 + 2] = (input[i] >> 8) & 0xFF;
            output[i * 4 + 3] = (input[i] >> 0) & 0xFF;
        }
        else{
            output[i * 4 + 0] = 0;
            output[i * 4 + 1] = 0;
            output[i * 4 + 2] = 0;
            output[i * 4 + 3] = 0;
        }
    }
}

__device__ void uint32_to_bytes(uint32_t* input, BYTE *output, size_t len) {
    for (size_t i = 0; i < len; i++) {
        output[i * 4 + 0] = (BYTE)(input[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (BYTE)(input[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (BYTE)(input[i] >> 8) & 0xFF;
        output[i * 4 + 3] = (BYTE)(input[i] >> 0) & 0xFF;
    }
}

__device__ void sha256_padding(uint8_t* data, size_t data_length_bytes, size_t total_length_bits) {
    for (size_t i = data_length_bytes + 1; i < 64 - 8; i++) {
        data[i] = 0x00;
    }

    size_t bit_length = total_length_bits;
    for (int i = 0; i < 8; i++) {
        data[64 - 1 - i] = bit_length & 0xFF;
        bit_length >>= 8;
    }

    data[data_length_bytes] = 0x80;
}

