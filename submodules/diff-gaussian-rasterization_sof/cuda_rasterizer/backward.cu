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

#include "backward.h"
#include "auxiliary.h"
#include "stopthepop/resorted_render.cuh"
#include "stopthepop/hierarchical_render.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

#ifdef ENABLE_NAN_CHECKS
	if(isnan(dL_dmeans[idx].x) || isnan(dL_dmeans[idx].y) || isnan(dL_dmeans[idx].z))
   	{
       printf("ComputeColorFromSH: (%f, %f, %f)\n", dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z);
   	}
#endif
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const bool proper_ewa_scaling,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	const float det_cov_orig = c_xx * c_yy - c_xy * c_xy;

	constexpr float h_var = 0.3f;
	c_xx += h_var;
	c_yy += h_var;

	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if (proper_ewa_scaling)
	{
		// As employed by Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting"
		// https://github.com/autonomousvision/mip-splatting
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov_orig / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		const float d_inside_root = (det_cov_orig / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);

		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / square(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}

	float denom = c_xx * c_yy - c_xy * c_xy;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian_backward(
	int idx, 
	const glm::vec3 scale, 
	const float3& mean, 
	const glm::vec4 rot, 
	const float* viewmatrix,  
	const float* view2gaussian, 
	const float* dL_dview2gaussian,
	glm::vec3* dL_dmeans, 
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots
	)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);

	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;


	double3 S_inv_square = {1.0f / ((double)scale.x * scale.x + 1e-7), 1.0f / ((double)scale.y * scale.y+ 1e-7), 1.0f / ((double)scale.z * scale.z+ 1e-7)};
	double C = t2.x * t2.x * S_inv_square.x + t2.y * t2.y * S_inv_square.y + t2.z * t2.z * S_inv_square.z;
	glm::mat3 S_inv_square_R = glm::mat3(
		S_inv_square.x * R_transpose[0][0], S_inv_square.y * R_transpose[0][1], S_inv_square.z * R_transpose[0][2],
		S_inv_square.x * R_transpose[1][0], S_inv_square.y * R_transpose[1][1], S_inv_square.z * R_transpose[1][2],
		S_inv_square.x * R_transpose[2][0], S_inv_square.y * R_transpose[2][1], S_inv_square.z * R_transpose[2][2]
	); 

	// compute the gradient here
	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dview2gaussian[0], 0.5f * dL_dview2gaussian[1], 0.5f * dL_dview2gaussian[2],
		0.5f * dL_dview2gaussian[1], dL_dview2gaussian[3], 0.5f * dL_dview2gaussian[4],
		0.5f * dL_dview2gaussian[2], 0.5f * dL_dview2gaussian[4], dL_dview2gaussian[5]
	);
	glm::vec3 dL_dB = glm::vec3(dL_dview2gaussian[6], dL_dview2gaussian[7], dL_dview2gaussian[8]);
	float dL_dC = dL_dview2gaussian[9];

	// glm::vec3 B = t2 * S_inv_square_R;
	// glm::mat3 Sigma = glm::transpose(R_transpose) * S_inv_square_R;
	glm::mat3 dL_dS_inv_square_R = R_transpose * dL_dSigma + glm::outerProduct(t2, dL_dB); //TODO: check if this is correct
	glm::mat3 dL_dR_transpose = glm::transpose(dL_dSigma * glm::transpose(S_inv_square_R));
	
	// glm::mat3 S_inv_square_R = glm::mat3(
	// 	S_inv_square.x * R_transpose[0][0], S_inv_square.y * R_transpose[0][1], S_inv_square.z * R_transpose[0][2],
	// 	S_inv_square.x * R_transpose[1][0], S_inv_square.y * R_transpose[1][1], S_inv_square.z * R_transpose[1][2],
	// 	S_inv_square.x * R_transpose[2][0], S_inv_square.y * R_transpose[2][1], S_inv_square.z * R_transpose[2][2]
	// ); 
	dL_dR_transpose += glm::mat3(
		S_inv_square.x * dL_dS_inv_square_R[0][0], S_inv_square.y * dL_dS_inv_square_R[0][1], S_inv_square.z * dL_dS_inv_square_R[0][2],
		S_inv_square.x * dL_dS_inv_square_R[1][0], S_inv_square.y * dL_dS_inv_square_R[1][1], S_inv_square.z * dL_dS_inv_square_R[1][2],
		S_inv_square.x * dL_dS_inv_square_R[2][0], S_inv_square.y * dL_dS_inv_square_R[2][1], S_inv_square.z * dL_dS_inv_square_R[2][2]
	); 
	float3 dL_dS_inv_square = {
		dL_dS_inv_square_R[0][0] * R_transpose[0][0] + dL_dS_inv_square_R[1][0] * R_transpose[1][0] + dL_dS_inv_square_R[2][0] * R_transpose[2][0],
		dL_dS_inv_square_R[0][1] * R_transpose[0][1] + dL_dS_inv_square_R[1][1] * R_transpose[1][1] + dL_dS_inv_square_R[2][1] * R_transpose[2][1],
		dL_dS_inv_square_R[0][2] * R_transpose[0][2] + dL_dS_inv_square_R[1][2] * R_transpose[1][2] + dL_dS_inv_square_R[2][2] * R_transpose[2][2]
	};
	// float C = t2.x * t2.x * S_inv_square.x + t2.y * t2.y * S_inv_square.y + t2.z * t2.z * S_inv_square.z;
	float3 dL_dt2 = {
		2 * t2.x * S_inv_square.x * dL_dC + dL_dB.x * S_inv_square_R[0][0] + dL_dB.y * S_inv_square_R[1][0] + dL_dB.z * S_inv_square_R[2][0],
		2 * t2.y * S_inv_square.y * dL_dC + dL_dB.x * S_inv_square_R[0][1] + dL_dB.y * S_inv_square_R[1][1] + dL_dB.z * S_inv_square_R[2][1],
		2 * t2.z * S_inv_square.z * dL_dC + dL_dB.x * S_inv_square_R[0][2] + dL_dB.y * S_inv_square_R[1][2] + dL_dB.z * S_inv_square_R[2][2]
	};
	dL_dS_inv_square.x += dL_dC * t2.x * t2.x;
	dL_dS_inv_square.y += dL_dC * t2.y * t2.y;
	dL_dS_inv_square.z += dL_dC * t2.z * t2.z;
	// float3 S_inv_square = {1.0f / (scale.x * scale.x), 1.0f / (scale.y * scale.y), 1.0f / (scale.z * scale.z)};
	glm::vec3 dL_dscale_idx = {
		-2 / scale.x * S_inv_square.x * dL_dS_inv_square.x,
		-2 / scale.y * S_inv_square.y * dL_dS_inv_square.y,
		-2 / scale.z * S_inv_square.z * dL_dS_inv_square.z
	};
	
	// write to memory 
	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = dL_dscale_idx.x;
	dL_dscale->y = dL_dscale_idx.y;
	dL_dscale->z = dL_dscale_idx.z;

	// glm::mat4 V2G = glm::inverse(G2V);
	// G2V = [R, t], V2G = inverse(G2V) = [R^T, -R^T * t]
	// V2G_R = G2V_R^T
	// V2G_t = -G2V_R^T * G2V_t
	glm::mat3 G2V_R_t = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);
	glm::mat3 G2V_R = glm::transpose(G2V_R_t);
	glm::vec3 G2V_t = glm::vec3(
		G2V[3][0], G2V[3][1], G2V[3][2]
	);
	// printf("backward: in cuda\n");
	// dL_dG2V_R = dL_dV2G_R^T
	// dL_dG2V_t = -dL_dV2G_t * G2V_R^T
	// glm::mat3 dL_dV2G_R_t = glm::mat3(
	// 	dL_dview2gaussian[0], dL_dview2gaussian[4], dL_dview2gaussian[8],
	// 	dL_dview2gaussian[1], dL_dview2gaussian[5], dL_dview2gaussian[9],
	// 	dL_dview2gaussian[2], dL_dview2gaussian[6], dL_dview2gaussian[10]
	// );
	// glm::vec3 dL_dV2G_t = glm::vec3(
	// 	dL_dview2gaussian[12], dL_dview2gaussian[13], dL_dview2gaussian[14]
	// );
	glm::vec3 dL_dV2G_t = glm::vec3(dL_dt2.x, dL_dt2.y, dL_dt2.z);
	glm::mat3 dL_dV2G_R_t = glm::transpose(dL_dR_transpose);

	// also gradient from -R^T * t
	glm::mat3 dL_dG2V_R_from_t = glm::mat3(
		-dL_dV2G_t.x * G2V_t.x, -dL_dV2G_t.x * G2V_t.y, -dL_dV2G_t.x * G2V_t.z,
		-dL_dV2G_t.y * G2V_t.x, -dL_dV2G_t.y * G2V_t.y, -dL_dV2G_t.y * G2V_t.z,
		-dL_dV2G_t.z * G2V_t.x, -dL_dV2G_t.z * G2V_t.y, -dL_dV2G_t.z * G2V_t.z
	);

	// TODO:
	glm::mat3 dL_dG2V_R = dL_dV2G_R_t + dL_dG2V_R_from_t;
	glm::vec3 dL_dG2V_t = -dL_dV2G_t * G2V_R_t;

	// dL_dG2V = [dL_dG2V_R, dL_dG2V_t]
	glm::mat4 dL_dG2V = glm::mat4(
		dL_dG2V_R[0][0], dL_dG2V_R[0][1], dL_dG2V_R[0][2], 0.0f,
		dL_dG2V_R[1][0], dL_dG2V_R[1][1], dL_dG2V_R[1][2], 0.0f,
		dL_dG2V_R[2][0], dL_dG2V_R[2][1], dL_dG2V_R[2][2], 0.0f,
		dL_dG2V_t.x, dL_dG2V_t.y, dL_dG2V_t.z, 0.0f
	);

	// Gaussian to view transform
	// glm::mat4 G2V = W2V * G2W;
	glm::mat4 dL_dG2W = glm::transpose(W2V) * dL_dG2V;

	
	// Gaussian to world transform
	// glm::mat4 G2W = glm::mat4(
	// 	R[0][0], R[1][0], R[2][0], 0.0f,
	// 	R[0][1], R[1][1], R[2][1], 0.0f,
	// 	R[0][2], R[1][2], R[2][2], 0.0f,
	// 	mean.x, mean.y, mean.z, 1.0f
	// );
	// dL_dG2W_R = dL_dG2W_R^T
	// dL_dG2W_t = dL_dG2W_t
	glm::mat3 dL_dG2W_R = glm::mat3(
		dL_dG2W[0][0], dL_dG2W[0][1], dL_dG2W[0][2],
		dL_dG2W[1][0], dL_dG2W[1][1], dL_dG2W[1][2],
		dL_dG2W[2][0], dL_dG2W[2][1], dL_dG2W[2][2]
	);
	glm::vec3 dL_dG2W_t = glm::vec3(
		dL_dG2W[3][0], dL_dG2W[3][1], dL_dG2W[3][2]
	);
	glm::mat3 dL_dR = dL_dG2W_R;

	// Gradients of loss w.r.t. means
	glm::vec3* dL_dmean = dL_dmeans + idx;
	dL_dmean->x = dL_dG2W_t.x;
	dL_dmean->y = dL_dG2W_t.y;
	dL_dmean->z = dL_dG2W_t.z;

	glm::mat3 dL_dMt = dL_dR;

	// // Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
#ifdef ENABLE_NAN_CHECKS
	if(isnan(dL_dmean->x) || isnan(dL_dmean->y) || isnan(dL_dmean->z))
   	{
       printf("ComputeView2Gaussian: (%f, %f, %f)\n",dL_dmean->x, dL_dmean->y, dL_dmean->z);
   	}
#endif
}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* view2gaussian,
	const float* filter3D,
	const float* viewmatrix,
	const float* dL_dview2gaussian,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	float* dL_dopacity,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;
	
	glm::vec3 scale =  scales[idx];

	// Loss w.r.t. filter3D part of opacity and scales
	// intermediate gradients
	if (filter3D != nullptr) {
		float _dL_dopacity = dL_dopacity[idx];	

		float opacity = opacities[idx];

		float filter_3d_sq = square(filter3D[idx]);

		glm::vec3 scale_sq = glm::vec3(
			square(scale.x),
			square(scale.y),
			square(scale.z)
		);

		// 3D filter for scaling
		glm::vec3 scale_hat_sq = glm::vec3{
			scale_sq.x + filter_3d_sq,
			scale_sq.y + filter_3d_sq,
			scale_sq.z + filter_3d_sq
		};

		float det1 = scale_sq.x * scale_sq.y * scale_sq.z;
		float det2 = scale_hat_sq.x * scale_hat_sq.y * scale_hat_sq.z;

		scale = glm::vec3{
			sqrtf(scale_hat_sq.x),
			sqrtf(scale_hat_sq.y),
			sqrtf(scale_hat_sq.z)
		};

		computeView2Gaussian_backward(idx, scale, means[idx], rotations[idx], viewmatrix, view2gaussian + 10 * idx, dL_dview2gaussian + 10 * idx, dL_dmeans, dL_dscale, dL_drot);
		
		// dL_do = dL_dohat * dohat_do
		float coef = sqrtf(det1 / det2);
		dL_dopacity[idx] = _dL_dopacity * coef;

		// dL_ds = dL_dshat * dshat_ds
		dL_dscale[idx].x *= (scales[idx].x / (scale.x));
		dL_dscale[idx].y *= (scales[idx].y / (scale.y));
		dL_dscale[idx].z *= (scales[idx].z / (scale.z));

		// dL_ds = dL_dohat * dohat_ds
		float o_hat = opacity * coef;
		dL_dscale[idx].x += filter_3d_sq * _dL_dopacity * o_hat / scale_hat_sq.x / scales[idx].x;
		dL_dscale[idx].y += filter_3d_sq * _dL_dopacity * o_hat / scale_hat_sq.y / scales[idx].y;
		dL_dscale[idx].z += filter_3d_sq * _dL_dopacity * o_hat / scale_hat_sq.z / scales[idx].z;
	}
	else {
		computeView2Gaussian_backward(idx, scale, means[idx], rotations[idx], viewmatrix, view2gaussian + 10 * idx, dL_dview2gaussian + 10 * idx, dL_dmeans, dL_dscale, dL_drot);
	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float far_plane,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* view2gaussian,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* dL_dview2gaussian)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	float2 ray = { (pixf.x - W/2.) / focal_x, (pixf.y - H/2.) / focal_y };

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	const float dL_dreg = inside ? dL_dpixels[DISTORTION_OFFSET * H * W + pix_id] : 0;
	// gradient from normalization
	// distortion /= (1 - T) * (1 - T) + 1e-7;
	const float distortion_before_normalized = inside ? final_Ts[pix_id + 3 * H * W] : 0;
	
	const float ddist_done_minus_T = -2.0f / ((1.f - T) * (1.f - T) * (1.f - T) + 1e-7);
	float dL_done_minus_T = distortion_before_normalized * ddist_done_minus_T * dL_dreg;
	const float dL_dT_final = -1.f * dL_done_minus_T;

	float last_dL_dT = 0;


	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int max_contributor = inside ? n_contrib[pix_id + H * W] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C]; // RGB
	float dL_dnormal2D[3]; // Normal
	float dL_dmax_depth = 0;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		for (int i = 0; i < 3; i++)
			dL_dnormal2D[i] = dL_dpixels[(C+i) * H * W + pix_id];
		dL_dmax_depth = dL_dpixels[DEPTH_OFFSET * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			for (int ii = 0; ii < 10; ii++)
				collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - (pixf.x - 0.5f), xy.y - (pixf.y - 0.5f) };
			const float4 con_o = collected_conic_opacity[j];
			float* view2gaussian_j = collected_view2gaussian + j * 10;
			
			float3 ray_point = { ray.x , ray.y, 1.0 };

			const float normal[3] = { 
				view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2], 
				view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
				view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
			};

			// use AA, BB, CC so that the name is unique
			double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
			double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			float CC = view2gaussian_j[9];
			
			// t is the depth of the gaussian
			float t = -BB/(2*AA);
			// depth must be positive otherwise it is not valid and we skip it
			if (t <= NEAR_PLANE)
				continue;

			double min_value = -(BB/AA) * (BB/4.) + CC;

			float power = -0.5f * min_value;
			if (power > 0.0f){
				power = 0.0f;
			}

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);

			if (alpha < 1.0f / 255.0f)
				continue;

			// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
			const float max_t = t;
			const float mapped_max_t = (far_plane * max_t - far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t);

			float dmax_t_dd = (far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t * max_t);

			// normalize normal
			float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length};

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// gradient for the distoration loss is taken from 2DGS paper, please check https://arxiv.org/pdf/2403.17888.pdf
			float dL_dt = 0.0f;
			float dL_dmax_t = 0.0f;
			float dL_dweight = 0.0f;

			// one_div_square_one_minus_T is from the normalization of distoration_2
			const float one_div_square_one_minus_T = 1.f / ((1.f - T_final) * (1.f - T_final));
			dL_dweight += (final_D2 + mapped_max_t * mapped_max_t * final_A - 2 * mapped_max_t * final_D) * dL_dreg * one_div_square_one_minus_T;			
			//TODO normalization of one_div_square_one_minus_T is missing
			dL_dmax_t += 2.0f * (T * alpha) * (mapped_max_t * final_A - final_D) * dL_dreg * dmax_t_dd;
			// from dL_done_minus_T since 1-T is  sum over weight;
			dL_dweight += dL_done_minus_T;
			// detach weight
			dL_dweight = 0.f;

			// only positive alpha gradient is considered
			// dL_dalpha += max(0.0f, dL_dweight - last_dL_dT);
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			
			float dL_dnormal_normalized[3] = {0};
			// // Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal_normalized[ch];
				dL_dalpha += (normal_normalized[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				dL_dnormal_normalized[ch] = alpha * T * dL_dnormal2D[ch];
			}

			// float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			// const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length};
			float dL_dlength = (dL_dnormal_normalized[0] * normal[0] + dL_dnormal_normalized[1] * normal[1] + dL_dnormal_normalized[2] * normal[2]);
			dL_dlength *= 1.f / (length * length);
			float dL_dnormal[3] = {
				(-dL_dnormal_normalized[0] + dL_dlength * normal[0]) / length,
				(-dL_dnormal_normalized[1] + dL_dlength * normal[1]) / length,
				(-dL_dnormal_normalized[2] + dL_dlength * normal[2]) / length
			};
			
			dL_dt = dL_dmax_t;
			if (contributor == max_contributor-1) {
				dL_dt += dL_dmax_depth;
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			// new metric for densification, please see Densification section in our paper (https://arxiv.org/pdf/2404.10772.pdf) for more details.
			const float abs_dL_dmean2D = abs(dL_dG * dG_ddelx * ddelx_dx) + abs(dL_dG * dG_ddely * ddely_dy);
            atomicAdd(&dL_dmean2D[global_id].z, abs_dL_dmean2D);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			// other gradients
			// G = exp(power);
			const float dG_dpower = G;
			const float dL_dpower = dL_dG * dG_dpower;

			// // float power = -0.5f * min_value;
			const float dL_dmin_value = dL_dpower * -0.5f;
			// float min_value = -(BB*BB)/(4*AA) + CC;
			// const float dL_dA = dL_dmin_value * (BB*BB)/4 *  1. / (AA*AA);
			double dL_dA = dL_dmin_value * (BB / AA) * (BB / AA) / 4.f;
			double dL_dB = dL_dmin_value * -BB / (2 *AA);
			double dL_dC = dL_dmin_value * 1.0f;

			dL_dA += dL_dt * BB / (2 * AA * AA);
			dL_dB += dL_dt * -1.f / (2 * AA);

			// const float normal[3] = { view2gaussian_j[0] * ray.x + view2gaussian_j[1] * ray.y + view2gaussian_j[2], 
			// 						view2gaussian_j[1] * ray.x + view2gaussian_j[3] * ray.y + view2gaussian_j[4],
			// 						view2gaussian_j[2] * ray.x + view2gaussian_j[4] * ray.y + view2gaussian_j[5]};

			// use AA, BB, CC so that the name is unique
			// float AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
			// float BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			// float CC = view2gaussian_j[9];
			dL_dnormal[0] += dL_dA * ray.x;
			dL_dnormal[1] += dL_dA * ray.y;
			dL_dnormal[2] += dL_dA;
			
			// write the gradients to global memory directly
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 0]), dL_dnormal[0] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 1]), dL_dnormal[0] * ray.y + dL_dnormal[1] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 2]), dL_dnormal[0] + dL_dnormal[2] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 3]), dL_dnormal[1] * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 4]), dL_dnormal[1] + dL_dnormal[2] * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 5]), dL_dnormal[2]);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 6]), dL_dB * 2 * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 7]), dL_dB * 2 * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 8]), dL_dB * 2);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 9]), dL_dC);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const bool proper_ewa_scaling,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* view2gaussian,
	const float* filter_3d,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dview2gaussian,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	//computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
	//	P,
	//	proper_ewa_scaling,
	//	means3D,
	//	radii,
	//	cov3Ds,
	//	focal_x,
	//	focal_y,
	//	tan_fovx,
	//	tan_fovy,
	//	viewmatrix,
	//	opacities,
	//	dL_dconic,
	//	dL_dopacity,
	//	(float3*)dL_dmean3D,
	//	dL_dcov3D);



	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		opacities,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		view2gaussian,
		filter_3d,
		viewmatrix,
		dL_dview2gaussian,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_dopacity,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	SplattingSettings splatting_settings,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* cov3D_inv,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	const float4* conic_opacity,
	const float* colors,
	const float* view2gaussian,
	const float* viewmatrix,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* pixel_colors,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dview2gaussian)
{
	#define CALL_KBUFFER(WINDOW) renderkBufferBackwardCUDA<NUM_CHANNELS, WINDOW> << <grid, block >> > (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*)cam_pos, colors, conic_opacity, final_Ts, n_contrib, bg_color, pixel_colors, dL_dpixels, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors)

	if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL)
	{
		renderCUDA<NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			splatting_settings.far_plane,
			bg_color,
			means2D,
			conic_opacity,
			colors,
			view2gaussian,
			final_Ts,
			n_contrib,
			dL_dpixels,
			dL_dmean2D,
			dL_dconic2D,
			dL_dopacity,
			dL_dcolors,
			dL_dview2gaussian);
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_KBUFFER)
	{
		int window_size = splatting_settings.sort_settings.queue_sizes.per_pixel;
		if (window_size <= 1) 
			CALL_KBUFFER(1); 
		else if (window_size <= 2) 
			CALL_KBUFFER(2); 
		else if (window_size <= 4) 
			CALL_KBUFFER(4); 
		else if (window_size <= 8) 
			CALL_KBUFFER(8); 
		else if (window_size <= 12) 
			CALL_KBUFFER(12); 
		else if (window_size <= 16) 
			CALL_KBUFFER(16); 
		else if (window_size <= 20) 
			CALL_KBUFFER(20); 
		else 
			CALL_KBUFFER(24);
		return;
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_FULL)
	{
		throw std::runtime_error("Backward not supported for full per-pixel sort");
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::HIERARCHICAL)
	{
#define CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, DETACH_ALPHA) \
	sortGaussiansRayHierarchicalCUDA_backward<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, DETACH_ALPHA><<<grid, {16, 4, 4}>>>( \
		ranges, point_list, W, H, focal_x, focal_y, splatting_settings.far_plane, splatting_settings.detach_alpha_extent, splatting_settings.include_alpha, view2gaussian, bg_color, means2D, cov3D_inv, projmatrix_inv, (float3*) cam_pos, conic_opacity, \
		colors, final_Ts, n_contrib, pixel_colors, dL_dpixels, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_dview2gaussian)

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE) if (splatting_settings.detach_alpha) { CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, true); } else { CALL_HIER_DETACHALPHA(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, false); }

#ifndef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		case 8: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8); break; } \
		case 12: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 12); break; } \
		case 16: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16); break; } \
		default: { throw std::runtime_error("Not supported head queue size " + std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel)); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		case 12: { CALL_HIER_HEAD(HIER_CULLING, 12); break; } \
		case 20: { CALL_HIER_HEAD(HIER_CULLING, 20); break; } \
		default: { throw std::runtime_error("Not supported mid queue size " + std::to_string(splatting_settings.sort_settings.queue_sizes.tile_2x2)); } \
	}
#else
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		default: { throw std::runtime_error("Not supported head queue size " + std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel)); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		default: { throw std::runtime_error("Not supported mid queue size " + std::to_string(splatting_settings.sort_settings.queue_sizes.tile_2x2)); } \
	}
#endif // STOPTHEPOP_FASTBUILD

	if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
		CALL_HIER_MID(true);
	} else {
		CALL_HIER_MID(false);
	}

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_DETACHALPHA
	}
}