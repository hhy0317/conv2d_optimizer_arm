#include "conv2d_process.h"
#include "gemm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <arm_neon.h>

#define DEBUG_LOG_ENABLE    0

#ifndef ALIGNED_ADDR_BYTE
#define ALIGNED_ADDR_BYTE   16
#endif

static inline void pld(const void *ptr)
{
    __builtin_prefetch(ptr, 0, 3);  // 0 = read, 3 = high locality
}

void nchw_to_nhwc(const float *input, float *output, int N, int C, int H, int W)
{
    for (int n = 0; n < N; n++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                for (int c = 0; c < C; c++)
                {
                    output[n * H * W * C + h * W * C + w * C + c] =
                        input[n * C * H * W + c * H * W + h * W + w];
                }
            }
        }
    }
}


void nhwc_to_nchw(const float *input, float *output, int N, int C, int H, int W)
{
    for (int n = 0; n < N; n++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                for (int c = 0; c < C; c++)
                {
                    output[n * C * H * W + c * H * W + h * W + w] =
                        input[n * H * W * C + h * W * C + w * C + c];
                }
            }
        }
    }
}

void reorder_conv_weight_oihw_to_ohwi(const float *input, float *output, int c_in, int c_out, int H, int W)
{
    for (int c_out_idx = 0; c_out_idx < c_out; c_out_idx++)
    {
        for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    output[c_out_idx * H * W * c_in + h * W * c_in + w * c_in + c_in_idx] = \
                    input[c_out_idx * c_in * H * W + c_in_idx * H * W + h * W + w];
                }
            }
        }
    }
}

void reorder_conv_weight_oihw_to_hwio(const float *input, float *output, int c_in, int c_out, int H, int W)
{
    for (int c_out_idx = 0; c_out_idx < c_out; c_out_idx++)
     {
         for (int c_in_idx = 0;   c_in_idx < c_in; c_in_idx++)
          {
              for (int h = 0; h  < H; h++)
               {
                   for (int w = 0; w < W; w++)
                {
                    output[h * W * c_in * c_out + w * c_in * c_out + c_in_idx * c_out + c_out_idx] = \
                    input[c_out_idx * c_in * H * W + c_in_idx * H * W + h * W + w];
                }
            }
        }
    }
}

void reorder_conv_weight_oihw_to_howi(const float *input, float *output, int c_in, int c_out, int H, int W)
{
    for (int c_out_idx = 0; c_out_idx < c_out; c_out_idx++)
    {
        for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    output[h * c_out * W * c_in + c_out_idx * W * c_in + w * c_in + c_in_idx] = \
                    input[c_out_idx * c_in * H * W + c_in_idx * H * W + h * W + w];
                }
            }
        }
    }
}

// NCHW版本,最原始的计算方法
void conv2d_process_basic(const float *input,
                          const float *weight,
                          const float *bias,
                          const int output_channel,
                          const int *input_size,
                          const int *kernel_size,
                          const int *stride,
                          const int *padding,
                          float *output,
                          int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    for (int n = 0; n < shape_n; n++)
    {
        for (int c = 0; c < output_channel; c++)
        {
            for (int h = 0; h < output_size[2]; h++)
            {
                for (int w = 0; w < output_size[3]; w++)
                {
                    float sum = 0;

                    for (int ic = 0; ic < shape_c; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {

                                int in_h = h * stride_h + kh - pad_h;
                                int in_w = w * stride_w + kw - pad_w;

                                // padding 的数据为0，所以不做计算
                                if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                    sum += input[n * shape_c * shape_h * shape_w + ic * shape_h * shape_w + in_h * shape_w + in_w] *
                                           weight[c * shape_c * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw];
                                }
                            }
                        }
                    }

                    output[n * output_channel * output_size[2] * output_size[3] + c * output_size[2] * output_size[3] + h * output_size[3] + w] = sum + bias[c];
                }
            }
        }
    }
}

// NHWC版本
void conv2d_process_basic_nhwc(const float *input,
                               const float *weight,
                               const float *bias,
                               const int output_channel,
                               const int *input_size,
                               const int *kernel_size,
                               const int *stride,
                               const int *padding,
                               float *output,
                               int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    // 输入转换为NHWC格式
    float *input_nhwc_ptr = NULL;
    posix_memalign((void **)&input_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * shape_h * shape_w * shape_c * sizeof(float));

    nchw_to_nhwc(input, input_nhwc_ptr, shape_n, shape_c, shape_h, shape_w);

    // 权重转换为HOWI格式
    float *weight_howi_ptr = NULL;
    posix_memalign((void **)&weight_howi_ptr, ALIGNED_ADDR_BYTE, output_channel * shape_c * kernel_h * kernel_w * sizeof(float));

    reorder_conv_weight_oihw_to_howi(weight, weight_howi_ptr, shape_c, output_channel, kernel_h, kernel_w);

    float *output_nhwc_ptr = NULL;
    posix_memalign((void **)&output_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * output_channel * output_size[2] * output_size[3] * sizeof(float));

    // NHWC版本计算卷积
    for (int n = 0; n < shape_n; n++)
    {
        for (int h = 0; h < output_size[2]; h++)
        {
            for (int w = 0; w < output_size[3]; w++)
            {
                for (int c = 0; c < output_channel; c++)
                {
                    float sum = 0;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            for (int ic = 0; ic < shape_c; ++ic) {
                                int in_h = h * stride_h + kh - pad_h;
                                int in_w = w * stride_w + kw - pad_w;

                                // padding 的数据为0，所以不做计算
                                if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                    sum += input_nhwc_ptr[n * shape_h * shape_w * shape_c + in_h * shape_w * shape_c + in_w * shape_c + ic] *
                                           weight_howi_ptr[kh * output_channel * kernel_w * shape_c + c * kernel_w * shape_c + kw * shape_c + ic];
                                }
                            }
                        }
                    }

                    // 输出赋值:NHWC版本
                    output_nhwc_ptr[n * output_size[2] * output_size[3] * output_channel + h * output_size[3] * output_channel + w * output_channel + c] = sum + bias[c];
                }
            }
        }
    }

    nhwc_to_nchw(output_nhwc_ptr, output, shape_n, output_channel, output_size[2], output_size[3]);

    free(input_nhwc_ptr);
    free(weight_howi_ptr);
    free(output_nhwc_ptr);
}


// NHWC版本，数据级并行
void conv2d_process_neon_nhwc(const float *input,
                              const float *weight,
                              const float *bias,
                              const int output_channel,
                              const int *input_size,
                              const int *kernel_size,
                              const int *stride,
                              const int *padding,
                              float *output,
                              int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    int neno_loop = shape_c >> 2;
    int neno_loop_remain = shape_c & 3;
    float *tmp_input_ic_ptr = NULL;
    float *tmp_weight_ic_ptr = NULL;

    // 输入转换为NHWC格式
    float *input_nhwc_ptr = NULL;
    posix_memalign((void **)&input_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * shape_h * shape_w * shape_c * sizeof(float));

    nchw_to_nhwc(input, input_nhwc_ptr, shape_n, shape_c, shape_h, shape_w);

    // 权重转换为HOWI格式
    float *weight_howi_ptr = NULL;
    posix_memalign((void **)&weight_howi_ptr, ALIGNED_ADDR_BYTE, output_channel * shape_c * kernel_h * kernel_w * sizeof(float));

    reorder_conv_weight_oihw_to_howi(weight, weight_howi_ptr, shape_c, output_channel, kernel_h, kernel_w);

    float *output_nhwc_ptr = NULL;
    posix_memalign((void **)&output_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * output_channel * output_size[2] * output_size[3] * sizeof(float));

    // NHWC版本计算卷积
    for (int n = 0; n < shape_n; n++)
    {
        for (int h = 0; h < output_size[2]; h++)
        {
            for (int w = 0; w < output_size[3]; w++)
            {
                for (int c = 0; c < output_channel; c++)
                {
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    float sum = 0;
                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            int in_h = h * stride_h + kh - pad_h;
                            int in_w = w * stride_w + kw - pad_w;
                            if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                tmp_input_ic_ptr = &input_nhwc_ptr[n * shape_h * shape_w * shape_c + in_h * shape_w * shape_c + in_w * shape_c];
                                tmp_weight_ic_ptr = &weight_howi_ptr[kh * output_channel * kernel_w * shape_c + c * kernel_w * shape_c + kw * shape_c];

                                for (int n = neno_loop; n > 0; n--)
                                {
                                    float32x4_t input_v = vld1q_f32(tmp_input_ic_ptr);
                                    float32x4_t weight_v = vld1q_f32(tmp_weight_ic_ptr);

                                    sum_vec = vmlaq_f32(sum_vec, input_v, weight_v);
                                    tmp_input_ic_ptr += 4;
                                    tmp_weight_ic_ptr += 4;
                                }

                                // 横向累加
                                sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                                      vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);

                                for (int re = neno_loop_remain; re > 0; re--)
                                {
                                    sum += *tmp_input_ic_ptr * *tmp_weight_ic_ptr;
                                    tmp_input_ic_ptr++;
                                    tmp_weight_ic_ptr++;
                                }
                            }
                        }
                    }

                    // 输出赋值:NHWC版本
                    output_nhwc_ptr[n * output_size[2] * output_size[3] * output_channel + h * output_size[3] * output_channel + w * output_channel + c] = sum + bias[c];
                }
            }
        }
    }

    nhwc_to_nchw(output_nhwc_ptr, output, shape_n, output_channel, output_size[2], output_size[3]);

    free(input_nhwc_ptr);
    free(weight_howi_ptr);
    free(output_nhwc_ptr);
}

// NHWC版本，数据级并行+pld预取
void conv2d_process_neon_pld_nhwc(const float *input,
                              const float *weight,
                              const float *bias,
                              const int output_channel,
                              const int *input_size,
                              const int *kernel_size,
                              const int *stride,
                              const int *padding,
                              float *output,
                              int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    int neno_loop = shape_c >> 2;
    int neno_loop_remain = shape_c & 3;
    float *tmp_input_ic_ptr = NULL;
    float *tmp_weight_ic_ptr = NULL;

    // 输入转换为NHWC格式
    float *input_nhwc_ptr = NULL;
    posix_memalign((void **)&input_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * shape_h * shape_w * shape_c * sizeof(float));

    nchw_to_nhwc(input, input_nhwc_ptr, shape_n, shape_c, shape_h, shape_w);

    // 权重转换为HOWI格式
    float *weight_howi_ptr = NULL;
    posix_memalign((void **)&weight_howi_ptr, ALIGNED_ADDR_BYTE, output_channel * shape_c * kernel_h * kernel_w * sizeof(float));

    reorder_conv_weight_oihw_to_howi(weight, weight_howi_ptr, shape_c, output_channel, kernel_h, kernel_w);

    float *output_nhwc_ptr = NULL;
    posix_memalign((void **)&output_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * output_channel * output_size[2] * output_size[3] * sizeof(float));

    // NHWC版本计算卷积
    for (int n = 0; n < shape_n; n++)
    {
        for (int h = 0; h < output_size[2]; h++)
        {
            for (int w = 0; w < output_size[3]; w++)
            {
                for (int c = 0; c < output_channel; c++)
                {
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    float sum = 0;
                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            int in_h = h * stride_h + kh - pad_h;
                            int in_w = w * stride_w + kw - pad_w;
                            if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                tmp_input_ic_ptr = &input_nhwc_ptr[n * shape_h * shape_w * shape_c + in_h * shape_w * shape_c + in_w * shape_c];
                                tmp_weight_ic_ptr = &weight_howi_ptr[kh * output_channel * kernel_w * shape_c + c * kernel_w * shape_c + kw * shape_c];

                                for (int n = neno_loop; n > 0; n--)
                                {
                                    pld(tmp_input_ic_ptr + 4);
                                    pld(tmp_weight_ic_ptr + 4);

                                    float32x4_t input_v = vld1q_f32(tmp_input_ic_ptr);
                                    float32x4_t weight_v = vld1q_f32(tmp_weight_ic_ptr);

                                    sum_vec = vmlaq_f32(sum_vec, input_v, weight_v);

                                    tmp_input_ic_ptr += 4;
                                    tmp_weight_ic_ptr += 4;
                                }

                                // 横向累加
                                sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                                      vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);

                                for (int re = neno_loop_remain; re > 0; re--)
                                {
                                    sum += *tmp_input_ic_ptr * *tmp_weight_ic_ptr;
                                    tmp_input_ic_ptr++;
                                    tmp_weight_ic_ptr++;
                                }
                            }
                        }
                    }

                    // 输出赋值:NHWC版本
                    output_nhwc_ptr[n * output_size[2] * output_size[3] * output_channel + h * output_size[3] * output_channel + w * output_channel + c] = sum + bias[c];
                }
            }
        }
    }

    nhwc_to_nchw(output_nhwc_ptr, output, shape_n, output_channel, output_size[2], output_size[3]);

    free(input_nhwc_ptr);
    free(weight_howi_ptr);
    free(output_nhwc_ptr);
}


// NHWC版本，数据级并行+output channel 数据级并行
void conv2d_process_neon_v2_nhwc(const float *input,
                              const float *weight,
                              const float *bias,
                              const int output_channel,
                              const int *input_size,
                              const int *kernel_size,
                              const int *stride,
                              const int *padding,
                              float *output,
                              int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    int neno_loop = shape_c >> 2;
    int neno_loop_remain = shape_c & 3;
    float *tmp_input_ic_ptr = NULL;
    float *tmp_weight_ic_ptr = NULL;

    int output_channel_loop = output_channel >> 2;
    int output_channel_count = output_channel_loop << 2;
    // int output_channel_loop_remain = output_channel & 3;
    int output_channel_step = kernel_w * shape_c;   // output channel 数据级并行，每次循环步长

    // 输入转换为NHWC格式
    float *input_nhwc_ptr = NULL;
    posix_memalign((void **)&input_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * shape_h * shape_w * shape_c * sizeof(float));

    nchw_to_nhwc(input, input_nhwc_ptr, shape_n, shape_c, shape_h, shape_w);

    // 权重转换为HOWI格式
    float *weight_howi_ptr = NULL;
    posix_memalign((void **)&weight_howi_ptr, ALIGNED_ADDR_BYTE, output_channel * shape_c * kernel_h * kernel_w * sizeof(float));

    reorder_conv_weight_oihw_to_howi(weight, weight_howi_ptr, shape_c, output_channel, kernel_h, kernel_w);

    float *output_nhwc_ptr = NULL;
    posix_memalign((void **)&output_nhwc_ptr, ALIGNED_ADDR_BYTE, shape_n * output_channel * output_size[2] * output_size[3] * sizeof(float));

    // NHWC版本计算卷积
    for (int n = 0; n < shape_n; n++)
    {
        for (int h = 0; h < output_size[2]; h++)
        {
            for (int w = 0; w < output_size[3]; w++)
            {
                for (int c = 0; c < output_channel_count; c += 4)
                {
                    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
                    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
                    float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
                    float32x4_t sum_vec3 = vdupq_n_f32(0.0f);

                    float sum[4] = {0};
                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            int in_h = h * stride_h + kh - pad_h;
                            int in_w = w * stride_w + kw - pad_w;
                            if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                tmp_input_ic_ptr = &input_nhwc_ptr[n * shape_h * shape_w * shape_c + in_h * shape_w * shape_c + in_w * shape_c];

                                tmp_weight_ic_ptr = &weight_howi_ptr[kh * output_channel * kernel_w * shape_c + c * kernel_w * shape_c + kw * shape_c];

                                for (int n = neno_loop; n > 0; n--)
                                {
                                    float32x4_t input_v = vld1q_f32(tmp_input_ic_ptr);
                                    float32x4_t weight_v0 = vld1q_f32(tmp_weight_ic_ptr);
                                    float32x4_t weight_v1 = vld1q_f32(tmp_weight_ic_ptr + output_channel_step);
                                    float32x4_t weight_v2 = vld1q_f32(tmp_weight_ic_ptr + output_channel_step * 2);
                                    float32x4_t weight_v3 = vld1q_f32(tmp_weight_ic_ptr + output_channel_step * 3);

                                    sum_vec0 = vmlaq_f32(sum_vec0, input_v, weight_v0);
                                    sum_vec1 = vmlaq_f32(sum_vec1, input_v, weight_v1);
                                    sum_vec2 = vmlaq_f32(sum_vec2, input_v, weight_v2);
                                    sum_vec3 = vmlaq_f32(sum_vec3, input_v, weight_v3);

                                    tmp_input_ic_ptr += 4;
                                    tmp_weight_ic_ptr += 4;
                                }

                                // 横向累加
                                sum[0] = vgetq_lane_f32(sum_vec0, 0) + vgetq_lane_f32(sum_vec0, 1) +
                                         vgetq_lane_f32(sum_vec0, 2) + vgetq_lane_f32(sum_vec0, 3) + bias[c + 0];
                                sum[1] = vgetq_lane_f32(sum_vec1, 0) + vgetq_lane_f32(sum_vec1, 1) +
                                         vgetq_lane_f32(sum_vec1, 2) + vgetq_lane_f32(sum_vec1, 3) + bias[c + 1];
                                sum[2] = vgetq_lane_f32(sum_vec2, 0) + vgetq_lane_f32(sum_vec2, 1) +
                                         vgetq_lane_f32(sum_vec2, 2) + vgetq_lane_f32(sum_vec2, 3) + bias[c + 2];
                                sum[3] = vgetq_lane_f32(sum_vec3, 0) + vgetq_lane_f32(sum_vec3, 1) +
                                         vgetq_lane_f32(sum_vec3, 2) + vgetq_lane_f32(sum_vec3, 3) + bias[c + 3];

                                for (int re = neno_loop_remain; re > 0; re--)
                                {
                                    sum[0] += *tmp_input_ic_ptr * *tmp_weight_ic_ptr;
                                    sum[1] += *tmp_input_ic_ptr * *(tmp_weight_ic_ptr + output_channel_step);
                                    sum[2] += *tmp_input_ic_ptr * *(tmp_weight_ic_ptr + output_channel_step * 2);
                                    sum[3] += *tmp_input_ic_ptr * *(tmp_weight_ic_ptr + output_channel_step * 3);

                                    tmp_input_ic_ptr++;
                                    tmp_weight_ic_ptr++;
                                }
                            }
                        }
                    }

                    // 输出赋值:NHWC版本
                    vst1q_f32(&output_nhwc_ptr[n * output_size[2] * output_size[3] * output_channel + h * output_size[3] * output_channel + w * output_channel + c], vld1q_f32(sum));
                }

                for (int c = output_channel_count; c < output_channel; c++)
                {
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    float sum = 0;
                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            int in_h = h * stride_h + kh - pad_h;
                            int in_w = w * stride_w + kw - pad_w;
                            if (in_h >= 0 && in_h < shape_h && in_w >= 0 && in_w < shape_w) {
                                tmp_input_ic_ptr = &input_nhwc_ptr[n * shape_h * shape_w * shape_c + in_h * shape_w * shape_c + in_w * shape_c];
                                tmp_weight_ic_ptr = &weight_howi_ptr[kh * output_channel * kernel_w * shape_c + c * kernel_w * shape_c + kw * shape_c];

                                for (int n = neno_loop; n > 0; n--)
                                {
                                    pld(tmp_input_ic_ptr + 4);
                                    pld(tmp_weight_ic_ptr + 4);

                                    float32x4_t input_v = vld1q_f32(tmp_input_ic_ptr);
                                    float32x4_t weight_v = vld1q_f32(tmp_weight_ic_ptr);

                                    sum_vec = vmlaq_f32(sum_vec, input_v, weight_v);

                                    tmp_input_ic_ptr += 4;
                                    tmp_weight_ic_ptr += 4;
                                }

                                // 横向累加
                                sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                                      vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);

                                for (int re = neno_loop_remain; re > 0; re--)
                                {
                                    sum += *tmp_input_ic_ptr * *tmp_weight_ic_ptr;
                                    tmp_input_ic_ptr++;
                                    tmp_weight_ic_ptr++;
                                }
                            }
                        }
                    }

                    // 输出赋值:NHWC版本
                    output_nhwc_ptr[n * output_size[2] * output_size[3] * output_channel + h * output_size[3] * output_channel + w * output_channel + c] = sum + bias[c];
                }
            }
        }
    }

    nhwc_to_nchw(output_nhwc_ptr, output, shape_n, output_channel, output_size[2], output_size[3]);

    free(input_nhwc_ptr);
    free(weight_howi_ptr);
    free(output_nhwc_ptr);
}

void img2col_nchw(const float *input,   // 输入数据 NCHW
                  int N, int C, int H, int W,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  float *col)           // 输出矩阵 [C×kH×kW][out_h×out_w]
{
    int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        int col_idx = 0;
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;

                            float val = 0.f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val = input[((n * C + c) * H + ih) * W + iw];
                            }

                            col[col_idx++] = val;
                        }
                    }
                }
            }
        }
    }
}

void reshape_output_to_nchw(const float *output_matmul, // [N × out_h × out_w][out_channels]
                            float *output_nchw,          // [N][out_channels][out_h][out_w]
                            int N,
                            int out_channels,
                            int out_h,
                            int out_w)
{
    int out_spatial = out_h * out_w;
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    int matmul_idx = (n * out_spatial + oh * out_w + ow) * out_channels + oc;
                    output_nchw[out_idx] = output_matmul[matmul_idx];
                }
            }
        }
    }
}

void conv2d_process_img2col(const float *input,
                            const float *weight,
                            const float *bias,
                            const int output_channel,
                            const int *input_size,
                            const int *kernel_size,
                            const int *stride,
                            const int *padding,
                            float *output,
                            int *output_size)
{
    int shape_n = input_size[0];
    int shape_c = input_size[1];
    int shape_h = input_size[2];
    int shape_w = input_size[3];

    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];

    int stride_h = stride[0];
    int stride_w = stride[1];

    int pad_h = padding[0];
    int pad_w = padding[1];

    output_size[0] = shape_n;
    output_size[1] = output_channel;
    output_size[2] = (shape_h + 2 * pad_h - kernel_h) / stride_h + 1;
    output_size[3] = (shape_w + 2 * pad_w - kernel_w) / stride_w + 1;

#if DEBUG_LOG_ENABLE
    printf("shape_n:%d,shape_c:%d,shape_h:%d,shape_w:%d\n", shape_n, shape_c, shape_h, shape_w);
    printf("kernel_h:%d,kernel_w:%d\n", kernel_h, kernel_w);
    printf("stride_h:%d,stride_w:%d\n", stride_h, stride_w);
    printf("pad_h:%d,pad_w:%d\n", pad_h, pad_w);
    printf("output_size:%d,%d,%d,%d\n", output_size[0], output_size[1], output_size[2], output_size[3]);
#endif

    // img2col 输入矩阵大小（C_in*K_h*K_w*h_out*w_out）
    float *input_matrix_ptr = NULL;
    posix_memalign((void **)&input_matrix_ptr, ALIGNED_ADDR_BYTE, output_size[2] * output_size[3] * shape_c * kernel_h * kernel_w * sizeof(float));

    img2col_nchw(input,
                 shape_n, shape_c, shape_h, shape_w,
                 kernel_h, kernel_w,
                 pad_h, pad_w,
                 stride_h, stride_w,
                 input_matrix_ptr);

    // img2col 输出矩阵大小（C_out*h_out*w_out）
    float *output_matrix_ptr = NULL;
    posix_memalign((void **)&output_matrix_ptr, ALIGNED_ADDR_BYTE, output_size[2] * output_size[3] * output_channel * sizeof(float));

    // 矩阵乘法
    // output = col_matrix × weight_matrix^T,函数内部实现了转置
#if (GEMM_MODE == GEMM_MODE_BASE)
    gemm_conv2d(input_matrix_ptr, weight, output_matrix_ptr, output_size[2] * output_size[3], shape_c * kernel_h * kernel_w, output_channel);
#elif (GEMM_MODE == GEMM_MODE_NEON)
    gemm_conv2d_neon(input_matrix_ptr, weight, output_matrix_ptr, output_size[2] * output_size[3], shape_c * kernel_h * kernel_w, output_channel);
#elif (GEMM_MODE == GEMM_MODE_NEON_V2)
    gemm_conv2d_neon_v2(input_matrix_ptr, weight, output_matrix_ptr, output_size[2] * output_size[3], shape_c * kernel_h * kernel_w, output_channel);
#endif

    // 加上偏置
    for (int i = 0; i < output_size[2] * output_size[3] * output_channel; ++i) {
        output_matrix_ptr[i] += bias[i % output_channel];
    }

    reshape_output_to_nchw(output_matrix_ptr, output, shape_n, output_channel, output_size[2], output_size[3]);

    free(input_matrix_ptr);
    free(output_matrix_ptr);
}
