#ifndef GRMM_H
#define GEMM_H

#define GEMM_MODE_BASE      0   // 朴素实现
#define GEMM_MODE_NEON      1   // NEON并行实现
#define GEMM_MODE_NEON_V2   2   // 块实现

#define GEMM_MODE           GEMM_MODE_NEON_V2

void gemm_conv2d(const float *col_matrix,    // [patch_num][patch_size]
                 const float *weight_matrix, // [out_channels][patch_size]
                 float *output,              // [patch_num][out_channels]
                 int patch_num,              // = N * out_h * out_w
                 int patch_size,             // = C * kH * kW
                 int out_channels);          // = num_output_channels

void gemm_conv2d_neon(const float *col_matrix,    // [patch_num][patch_size]
                      const float *weight_matrix, // [out_channels][patch_size]
                      float *output,              // [patch_num][out_channels]
                      int patch_num,              // = N * out_h * out_w
                      int patch_size,             // = C * kH * kW
                      int out_channels);          // = num_output_channels

void gemm_conv2d_neon_v2(const float *col_matrix,    // [patch_num][patch_size]
                         const float *weight_matrix, // [out_channels][patch_size]
                         float *output,              // [patch_num][out_channels]
                         int patch_num,              // = N * out_h * out_w
                         int patch_size,             // = C * kH * kW
                         int out_channels);          // = num_output_channels
#endif
