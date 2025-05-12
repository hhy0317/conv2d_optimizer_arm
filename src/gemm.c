#include "gemm.h"
#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>

void gemm_conv2d(const float* col_matrix,        // [patch_num][patch_size]
                 const float* weight_matrix,     // [out_channels][patch_size]
                 float* output,                  // [patch_num][out_channels]
                 int patch_num,                  // = N * out_h * out_w
                 int patch_size,                 // = C * kH * kW
                 int out_channels)               // = num_output_channels
{
    for (int i = 0; i < patch_num; ++i) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.f;
            for (int j = 0; j < patch_size; ++j) {
                sum += col_matrix[i * patch_size + j] * weight_matrix[oc * patch_size + j];
            }
            output[i * out_channels + oc] = sum;
        }
    }
}

void gemm_conv2d_neon(const float *col_matrix,    // [patch_num][patch_size]
                      const float *weight_matrix, // [out_channels][patch_size]
                      float *output,              // [patch_num][out_channels]
                      int patch_num,              // = N * out_h * out_w
                      int patch_size,             // = C * kH * kW
                      int out_channels)          // = num_output_channels
{
    int neon_loop = patch_size >> 2;
    int neon_remain = patch_size & 3;
    const float *col_matrix_ptr = NULL;
    const float *weight_matrix_ptr = NULL;

    for (int i = 0; i < patch_num; ++i) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float32x4_t sum_vec = vdupq_n_f32(0.f);
            float sum = 0.f;
            col_matrix_ptr = &col_matrix[i * patch_size];
            weight_matrix_ptr = &weight_matrix[oc * patch_size];

            for (int n = neon_loop; n > 0; --n) {
                float32x4_t col = vld1q_f32(col_matrix_ptr);
                float32x4_t weight = vld1q_f32(weight_matrix_ptr);
                sum_vec = vmlaq_f32(sum_vec, col, weight);

                col_matrix_ptr += 4;
                weight_matrix_ptr += 4;
            }
            sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                  vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
            for (int n = neon_remain; n > 0; --n)
            {
                sum += *col_matrix_ptr * *weight_matrix_ptr;
                col_matrix_ptr++;
                weight_matrix_ptr++;
            }

            output[i * out_channels + oc] = sum;
        }
    }
}

void gemm_conv2d_neon_v2(const float *col_matrix,    // [patch_num][patch_size]
                      const float *weight_matrix, // [out_channels][patch_size]
                      float *output,              // [patch_num][out_channels]
                      int patch_num,              // = N * out_h * out_w
                      int patch_size,             // = C * kH * kW
                      int out_channels)          // = num_output_channels
{
    int neon_loop = patch_size >> 2;
    // int neon_remain = patch_size & 3;
    const float *col_matrix_ptr = NULL;
    const float *weight_matrix_ptr = NULL;

    for (int i = 0; i < patch_num; ++i) {
        for (int oc = 0; oc < out_channels; oc += 4) {
            float32x4_t sum_vec0 = vdupq_n_f32(0.f);
            float32x4_t sum_vec1 = vdupq_n_f32(0.f);
            float32x4_t sum_vec2 = vdupq_n_f32(0.f);
            float32x4_t sum_vec3 = vdupq_n_f32(0.f);

            float sum[4] = {0.f};
            col_matrix_ptr = &col_matrix[i * patch_size];
            weight_matrix_ptr = &weight_matrix[oc * patch_size];

            for (int n = neon_loop; n > 0; --n) {
                float32x4_t col = vld1q_f32(col_matrix_ptr);
                float32x4_t weight0 = vld1q_f32(weight_matrix_ptr);
                float32x4_t weight1 = vld1q_f32(weight_matrix_ptr + patch_size);
                float32x4_t weight2 = vld1q_f32(weight_matrix_ptr + patch_size * 2);
                float32x4_t weight3 = vld1q_f32(weight_matrix_ptr + patch_size * 3);

                sum_vec0 = vmlaq_f32(sum_vec0, col, weight0);
                sum_vec1 = vmlaq_f32(sum_vec1, col, weight1);
                sum_vec2 = vmlaq_f32(sum_vec2, col, weight2);
                sum_vec3 = vmlaq_f32(sum_vec3, col, weight3);

                col_matrix_ptr += 4;
                weight_matrix_ptr += 4;
            }

            sum[0] = vgetq_lane_f32(sum_vec0, 0) + vgetq_lane_f32(sum_vec0, 1) +
                     vgetq_lane_f32(sum_vec0, 2) + vgetq_lane_f32(sum_vec0, 3);
            sum[1] = vgetq_lane_f32(sum_vec1, 0) + vgetq_lane_f32(sum_vec1, 1) +
                     vgetq_lane_f32(sum_vec1, 2) + vgetq_lane_f32(sum_vec1, 3);
            sum[2] = vgetq_lane_f32(sum_vec2, 0) + vgetq_lane_f32(sum_vec2, 1) +
                     vgetq_lane_f32(sum_vec2, 2) + vgetq_lane_f32(sum_vec2, 3);
            sum[3] = vgetq_lane_f32(sum_vec3, 0) + vgetq_lane_f32(sum_vec3, 1) +
                     vgetq_lane_f32(sum_vec3, 2) + vgetq_lane_f32(sum_vec3, 3);

            vst1q_f32(&output[i * out_channels + oc], vld1q_f32(sum));
        }
    }
}
