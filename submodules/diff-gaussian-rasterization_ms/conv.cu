#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>

namespace cg = cooperative_groups;

#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f




#define BX 32
#define BY 32
#define BLOCK_DIM 16


template <int C>
__device__ float get_pix_value(const float* img, const int c, const int y, const int x, const int H, const int W) {
    if (x >= W || y >= H || x < 0 || y < 0) {
        return 0.0f;
    } else {
        return img[c * H * W + y * W + x];
    }
}

__device__ inline float do_sq(float val) {
    return val * val;
}




template <int C>
__device__ void load_into_shared(float pixels[BY + 10][BX + 10], float *input1, float *input2, int H, int W, int i, int subtract = 0) {
	auto block = cg::this_thread_block();
    const int start_y = block.group_index().y * (BY - subtract) - subtract / 2;
    const int start_x = block.group_index().x * (BX - subtract) - subtract / 2;

    const int cnt = (BY + 10) * (BX + 10);
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        int tid = b * (BX * BY) + block.thread_rank();
        if (tid < cnt) {
            int local_y = tid / (BX + 10);
            int local_x = tid % (BX + 10);
            int y = start_y + local_y;
            int x = start_x + local_x;
            if (input2 == nullptr) {
                float one = get_pix_value<C>(input1, i, y - 5, x - 5, H, W);
                pixels[local_y][local_x] = one;
            } else {
                float one = get_pix_value<C>(input1, i, y - 5, x - 5, H, W);
                float two = get_pix_value<C>(input2, i, y - 5, x - 5, H, W);
                pixels[local_y][local_x] = one * two;
            }
        }
    }
}



__device__ void write_to_shared(float pixels[BY + 10][BX + 10], float val) {
	auto block = cg::this_thread_block();

    // flush with 0s
    const int cnt = (BY + 10) * (BX + 10);
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        int tid = b * (BX * BY) + block.thread_rank();
        if (tid < cnt) {
            int local_y = tid / (BX + 10);
            int local_x = tid % (BX + 10);
            pixels[local_y][local_x] = 0.0f;
        }
    }
    block.sync();

    // write the values in the central BXxBY zone
    pixels[block.thread_index().y + 5][block.thread_index().x + 5] = val;
}

__device__ void multiply_shared_mem(float pix1[BY + 10][BX + 10], float pix2[BY + 10][BX + 10]) {
	auto block = cg::this_thread_block();
    const int cnt = (BY + 10) * (BX + 10);
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        int tid = b * (BX * BY) + block.thread_rank();
        if (tid < cnt) {
            int local_y = tid / (BX + 10);
            int local_x = tid % (BX + 10);
            float one = pix1[local_y][local_x];
            float two = pix2[local_y][local_x];
            pix1[local_y][local_x] = one * two;
        }
    }
}



__device__ float do_conv_1D2(float pixels_out[BY][BX + 10], float pixels_in[BY + 10][BX + 10], bool sq = false) {
	auto block = cg::this_thread_block();

    const int cnt = (BY) * (BX + 10);
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        int tid = b * (BX * BY) + block.thread_rank();
        if (tid < cnt) {
            int local_y = tid / (BX + 10);
            int local_x = tid % (BX + 10);

            int y = local_y+5;         

            float val=0;

            if (sq) {
                val += G_00 * do_sq(pixels_in[y -5][local_x]);
                val += G_01 * do_sq(pixels_in[y -4][local_x]);
                val += G_02 * do_sq(pixels_in[y -3][local_x]);
                val += G_03 * do_sq(pixels_in[y -2][local_x]);
                val += G_04 * do_sq(pixels_in[y -1][local_x]);
                val += G_05 * do_sq(pixels_in[y +0][local_x]);
                val += G_06 * do_sq(pixels_in[y +1][local_x]);
                val += G_07 * do_sq(pixels_in[y +2][local_x]);
                val += G_08 * do_sq(pixels_in[y +3][local_x]);
                val += G_09 * do_sq(pixels_in[y +4][local_x]);
                val += G_10 * do_sq(pixels_in[y +5][local_x]);
            } else {
                val += G_00 * (pixels_in[y -5][local_x]);
                val += G_01 * (pixels_in[y -4][local_x]);
                val += G_02 * (pixels_in[y -3][local_x]);
                val += G_03 * (pixels_in[y -2][local_x]);
                val += G_04 * (pixels_in[y -1][local_x]);
                val += G_05 * (pixels_in[y +0][local_x]);
                val += G_06 * (pixels_in[y +1][local_x]);
                val += G_07 * (pixels_in[y +2][local_x]);
                val += G_08 * (pixels_in[y +3][local_x]);
                val += G_09 * (pixels_in[y +4][local_x]);
                val += G_10 * (pixels_in[y +5][local_x]);
            }
            

            pixels_out[local_y][local_x] = val;
        }
        
    }

    block.sync();

    int local_y = block.thread_index().y;
    int local_x = block.thread_index().x+5;
    float val = 0.0f;
    
    val += G_00 * pixels_out[local_y][local_x - 5];
    val += G_01 * pixels_out[local_y][local_x - 4];
    val += G_02 * pixels_out[local_y][local_x - 3];
    val += G_03 * pixels_out[local_y][local_x - 2];
    val += G_04 * pixels_out[local_y][local_x - 1];
    val += G_05 * pixels_out[local_y][local_x    ];
    val += G_06 * pixels_out[local_y][local_x + 1];
    val += G_07 * pixels_out[local_y][local_x + 2];
    val += G_08 * pixels_out[local_y][local_x + 3];
    val += G_09 * pixels_out[local_y][local_x + 4];
    val += G_10 * pixels_out[local_y][local_x + 5];  
    return val;
}












template <int CH>
__global__ void fusedssimCUDA(
    int H,
    int W,
    float C1,
    float C2,
    float* img1,
    float* img2,
    float* MU1,
    float* MU2,
    float* SIGMA1_sq,
    float* SIGMA2_sq,
    float* SIGMA12,
    float* ssim_map
)
{
	auto block = cg::this_thread_block();
    const int pix_y = block.group_index().y * BY + block.thread_index().y;
    const int pix_x = block.group_index().x * BX + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // stats for ssim
    float mu1 = 0.0f;
    float mu2 = 0.0f;
    float sigma1_sq = 0.0f;
    float sigma2_sq = 0.0f;
    float sigma12 = 0.0f;

    // shared memory that will be used to load pixels temporarily
    __shared__ float buf1[BY + 10][BX + 10];
    __shared__ float buf2[BY + 10][BX + 10];
    __shared__ float buf3[BY][BX + 10];

    // mu1 <- Conv(img1)
    // sigma1_sq = Conv(img1 * img1) - mu1_sq
    for (int i = 0; i < CH; ++i) {
        // load into shared
        load_into_shared<CH>(buf1, img1, nullptr, H, W, i);
        block.sync();

        // conv
        mu1 = do_conv_1D2(buf3, buf1);
        block.sync();

        sigma1_sq = do_conv_1D2(buf3, buf1, true) - mu1 * mu1;
        block.sync();

    // mu2 <- Conv(img2)
    // sigma2_sq = Conv(img2 * img2) - mu2_sq
        // load into shared
        load_into_shared<CH>(buf2, img2, nullptr, H, W, i);
        block.sync();
        // conv
        mu2 = do_conv_1D2(buf3, buf2);
        block.sync();

        sigma2_sq = do_conv_1D2(buf3, buf2, true) - mu2 * mu2;
        block.sync();

    // sigma12 = Conv(img1 * img2) - mu1_mu2
        // load into shared
        multiply_shared_mem(buf1, buf2);
        block.sync();
        // conv
        sigma12 = do_conv_1D2(buf3, buf1) - mu1 * mu2;
        block.sync();

        float mu1_sq = mu1 * mu1;
        float mu2_sq = mu2 * mu2;
        float mu1_mu2 = mu1 * mu2;
        float C = (2.0f * mu1_mu2 + C1);
        float D = (2.0f * sigma12 + C2);
        float A = (mu1_sq + mu2_sq + C1);
        float B = (sigma1_sq + sigma2_sq + C2);
        float m = (C * D) / (A * B);
        if (pix_x < W && pix_y < H) {
            ssim_map[i * num_pix + pix_id] = m;
            MU1[i * num_pix + pix_id] = mu1;
            MU2[i * num_pix + pix_id] = mu2;
            SIGMA1_sq[i * num_pix + pix_id] = sigma1_sq;
            SIGMA2_sq[i * num_pix + pix_id] = sigma2_sq;
            SIGMA12[i * num_pix + pix_id] = sigma12;
        }
    }
}




__device__ bool in_inner_window() {
	auto block = cg::this_thread_block();
    return 5 <= block.thread_index().y && block.thread_index().y < BY - 5 && 5 <= block.thread_index().x && block.thread_index().x < BX - 5;
}

template <int CH>
__global__ void fusedssim_backwardCUDA(
    int H,
    int W,
    float C1,
    float C2,
    float* img1,
    float* img2,
    float* MU1,
    float* MU2,
    float* SIGMA1_sq,
    float* SIGMA2_sq,
    float* SIGMA12,    
    float *dL_dmap,
    float *dL_dimg1)
{
	auto block = cg::this_thread_block();
    const int pix_y = block.group_index().y * (BY - 10) + block.thread_index().y - 5;
    const int pix_x = block.group_index().x * (BX - 10) + block.thread_index().x - 5;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // stats for ssim
    float mu1 = 0.0f;
    float mu2 = 0.0f;
    float sigma1_sq = 0.0f;
    float sigma2_sq = 0.0f;
    float sigma12 = 0.0f;

    // shared memory that will be used to load pixels temporarily
    __shared__ float buf2[BY + 10][BX + 10];
    __shared__ float buf3[BY][BX + 10];

    // mu1 <- Conv(img1)
    // sigma1_sq = Conv(img1 * img1) - mu1_sq
    for (int i = 0; i < CH; ++i) {
        
        mu1 = get_pix_value<CH>(MU1, i, pix_y, pix_x, H, W);
        mu2 = get_pix_value<CH>(MU2, i, pix_y, pix_x, H, W);
        sigma1_sq = get_pix_value<CH>(SIGMA1_sq, i, pix_y, pix_x, H, W);
        sigma2_sq = get_pix_value<CH>(SIGMA2_sq, i, pix_y, pix_x, H, W);
        sigma12 = get_pix_value<CH>(SIGMA12, i, pix_y, pix_x, H, W);

        float mu1_sq = mu1 * mu1;
        float mu2_sq = mu2 * mu2;
        float mu1_mu2 = mu1 * mu2;
        float C = (2.0f * mu1_mu2 + C1);
        float D = (2.0f * sigma12 + C2);
        float A = (mu1_sq + mu2_sq + C1);
        float B = (sigma1_sq + sigma2_sq + C2);


        float dL_dm = 0.0f;
        // if (in_inner_window() && pix_x < W && pix_y < H)
        if (pix_x < W && pix_y < H && pix_x >=0 && pix_y >=0)
            dL_dm = dL_dmap[i * num_pix + pix_id];
        float dL_dmu1 = dL_dm * (
            (mu2 * 2.0f * D) / (A * B)
            -(mu2 * 2.0f * C) / (A * B)
            -(mu1 * 2.0f * C * D) / ( A * A * B)
            +(mu1 * 2.0f * C * D) / (A * B * B)
            );
        float dL_dsigma1_sq = dL_dm * ((-C * D) / (A * B * B));
        float dL_dsigma12 = dL_dm * ((2 * C) / (A * B));

        float dL_dpix = 0.0f;
        float tmp = 0.0f;

        // gradient from mu1
        write_to_shared(buf2, dL_dmu1);
        block.sync();
        // tmp = do_conv(buf2, H, W);
        tmp = do_conv_1D2(buf3, buf2);
        block.sync();
        dL_dpix += tmp;

        // gradient from sigma1_sq
        write_to_shared(buf2, dL_dsigma1_sq);
        block.sync();
        tmp = get_pix_value<CH>(img1, i, pix_y, pix_x, H, W);
        tmp *= 2.0f * do_conv_1D2(buf3, buf2);

        block.sync();
        dL_dpix += tmp;

        // gradient from sigma12
        write_to_shared(buf2, dL_dsigma12);
        block.sync();
        tmp = get_pix_value<CH>(img2, i, pix_y, pix_x, H, W);
        tmp *= do_conv_1D2(buf3, buf2);
        
        block.sync();
        dL_dpix += tmp;

        if (in_inner_window() && pix_x < W && pix_y < H)
            dL_dimg1[i * num_pix + pix_id] = dL_dpix;
    }
}










std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
)
{
    int H = img1.size(1);
    int W = img1.size(2);
	dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, 1);
	dim3 block(BX, BY, 1);
	// dim3 grid((W + (BX - 10) - 1) / (BX - 10), (H + (BY - 10) - 1) / (BY - 10), 1);
	// dim3 block(BX, BY, 1);

    torch::Tensor MU1 = torch::zeros_like(img1).contiguous();
    torch::Tensor MU2 = torch::zeros_like(img1).contiguous();
    torch::Tensor SIGMA1_sq = torch::zeros_like(img1).contiguous();
    torch::Tensor SIGMA2_sq = torch::zeros_like(img1).contiguous();
    torch::Tensor SIGMA12 = torch::zeros_like(img1).contiguous();
    torch::Tensor target = torch::zeros_like(img1).contiguous();

    fusedssimCUDA<3><<<grid,block>>>(
        H,
        W,
        C1,
        C2,
        img1.contiguous().data<float>(),
        img2.contiguous().data<float>(),
        MU1.contiguous().data<float>(),
        MU2.contiguous().data<float>(),
        SIGMA1_sq.contiguous().data<float>(),
        SIGMA2_sq.contiguous().data<float>(),
        SIGMA12.contiguous().data<float>(),
        target.contiguous().data<float>()
    );

    return std::make_tuple(target, MU1, MU2, SIGMA1_sq, SIGMA2_sq, SIGMA12);
}





torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &MU1,
    torch::Tensor &MU2,    
    torch::Tensor &SIGMA1_sq,
    torch::Tensor &SIGMA2_sq,
    torch::Tensor &SIGMA12,
    torch::Tensor &dL_dmap
)
{
    int H = img1.size(1);
    int W = img1.size(2);
	dim3 grid((W + (BX - 10) - 1) / (BX - 10), (H + (BY - 10) - 1) / (BY - 10), 1);
	dim3 block(BX, BY, 1);

    torch::Tensor dL_dimg1 = torch::zeros_like(img1).contiguous();

    fusedssim_backwardCUDA<3><<<grid,block>>>(
        H,
        W,
        C1,
        C2,
        img1.contiguous().data<float>(),
        img2.contiguous().data<float>(),
        MU1.contiguous().data<float>(),
        MU2.contiguous().data<float>(),
        SIGMA1_sq.contiguous().data<float>(),
        SIGMA2_sq.contiguous().data<float>(),
        SIGMA12.contiguous().data<float>(),
        dL_dmap.contiguous().data<float>(),
        dL_dimg1.contiguous().data<float>()
    );

    return dL_dimg1;
}