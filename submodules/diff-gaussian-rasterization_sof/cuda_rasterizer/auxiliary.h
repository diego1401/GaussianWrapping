/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define DEPTH_OFFSET 6
#define ALPHA_OFFSET 7
#define DISTORTION_OFFSET 8
#define OPACITY_LOSS_OFFSET 9
#define OUTPUT_CHANNELS 10

// distortion loss
#define NEAR_PLANE 0.2f
#define OPT_CULL_POINTS
// 100.0 for GOF/2DGS
// #define FAR_PLANE 100.0

// sh debug
// #define SH_DEBUG

// correct bounding

#define MAX_NUM_PROJECTED 256
#define DEBUG 0
//#define CORRECT_EXACT_DEPTH_GRAD
//#define DEBUG_INTEGRATE
//#define DEBUG_OPACITY_FIELD
//#define DEBUG_TILE_LAUNCHES
#define OPT_TILE_LAUNCHES
//#define ENABLE_NAN_CHECKS
//#define DEBUG_MIN_Z_BOUNDING
#define POINT_TO_DEBUG (1226113)

#define ALPHA_THRESHOLD (1.0f / 255.0f)
#define ALPHA_THRESHOLD_PADDED (1.0f / 255.0f)
#define T_THRESHOLD (0.0001f)

constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t WARP_MASK = 0xFFFFFFFFU;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float square(float x) 
{ 
	return x * x;
}

template<typename T>
__device__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

__device__ inline float3 make_float3(const float4& f4)
{
	return { f4.x, f4.y, f4.z };
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float NDC(float z, float far_plane)
{
	return (far_plane * z - far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * z);
}

__forceinline__ __device__ glm::vec3 pix2world(const glm::vec2 pix, const int W, const int H, glm::vec4 inverse_vp0, glm::vec4 inverse_vp1, glm::vec4 inverse_vp3)
{
	const glm::vec2 pix_ndc = pix * glm::vec2(2.0f / W, 2.0f / H) - 1.0f;
	glm::vec4 p_world = inverse_vp0 * pix_ndc.x + inverse_vp1 * pix_ndc.y + inverse_vp3;
	float rcp_w = __frcp_rn(p_world.w);
	return glm::vec3(p_world) * rcp_w;
}
__forceinline__ __device__ glm::vec3 pix2world(const glm::vec2 pix, const int W, const int H, const glm::mat4 inverse_vp)
{
	return pix2world(pix, W, H, inverse_vp[0], inverse_vp[1], inverse_vp[3]);
}

__forceinline__ __device__ glm::vec3 world2ndc(const glm::vec3 p_world, const glm::mat4 viewproj_matrix)
{
	glm::vec4 p_hom = viewproj_matrix * glm::vec4(p_world, 1.0f);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	glm::vec3 p_ndc = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	return p_ndc;
}
// GOF Version of GetRect
__forceinline__ __device__ void getRect_GOF(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, const float2 rect_extent, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int) floorf((p.x - rect_extent.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int) floorf((p.y - rect_extent.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int) ceilf((p.x + rect_extent.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int) ceilf((p.y + rect_extent.y) / BLOCK_Y)))
	};
}

__forceinline__ __device__ glm::mat4x4 loadMatrix4x4(const float* matrix)
{
	glm::mat4x4 mat;
	for (int i = 0; i < 4; i++)
	{
		float4 tmp = *((float4*) (matrix + i * 4));
		mat[i][0] = tmp.x;
		mat[i][1] = tmp.y;
		mat[i][2] = tmp.z;
		mat[i][3] = tmp.w;
	}
	return mat;
}

__forceinline__ __device__ glm::mat4x3 loadMatrix4x3(const float* matrix)
{
	glm::mat4x3 mat;
	for (int i = 0; i < 4; i++)
	{
		float4 tmp = *((float4*) (matrix + i * 4));
		mat[i][0] = tmp.x;
		mat[i][1] = tmp.y;
		mat[i][2] = tmp.z;
	}
	return mat;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum_GOF(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);
	//printf("Z:%f", p_view.z);
	if (p_view.z <= NEAR_PLANE)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}


__forceinline__ __device__ bool in_frustum(int idx,
	const glm::vec3 mean3D,
	const glm::mat4x3 viewmatrix,
	bool prefiltered,
	glm::vec3& p_view)
{
	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(mean3D, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// p_view = transformPoint4x3(p_orig, viewmatrix);

	glm::vec4 p_world(mean3D, 1.0f);
	p_view = viewmatrix * p_world;

	if (p_view.z <= NEAR_PLANE)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__device__ __inline__ uint64_t constructSortKey(uint32_t tile_id, float depth)
{
	uint64_t key = tile_id;
	key <<= 32;
	key |= *((uint32_t*)&depth);
	return key;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif