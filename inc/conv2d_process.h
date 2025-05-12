#ifndef _CONV2D_PROCESS_H_
#define _CONV2D_PROCESS_H_

#include <stdint.h>

#define N_IN        1
#define C_IN        4
#define H_IN        1
#define W_IN        32

#define K_H        1
#define K_W        5

#define N_OUT       1
#define C_OUT       24
#define H_OUT       1
#define W_OUT       32


#define CONV2D_PROCESS_MODE_BASIC           0
#define CONV2D_PROCESS_MODE_BASIC_NHWC      1
#define CONV2D_PROCESS_MODE_NEON_NHWC       2
#define CONV2D_PROCESS_MODE_NEON_PLD_NHWC   3
#define CONV2D_PROCESS_MODE_NEON_V2_NHWC    4
#define CONV2D_PROCESS_MODE_IMG2COL_NWCH    5

#define CONV2D_PROCESS_MODE                 CONV2D_PROCESS_MODE_IMG2COL_NWCH


#define FLOAT_DATA_PROCESS                  0
#define FIXED_DATA_PROCESS                  1

#define DATA_PROCESS_MODE                   FLOAT_DATA_PROCESS

void conv2d_process_basic(const float *input,
                          const float *weight,
                          const float *bias,
                          const int output_channel,
                          const int *input_size,
                          const int *kernel_size,
                          const int *stride,
                          const int *padding,
                          float *output,
                          int *output_size);

void conv2d_process_basic_nhwc(const float *input,
                               const float *weight,
                               const float *bias,
                               const int output_channel,
                               const int *input_size,
                               const int *kernel_size,
                               const int *stride,
                               const int *padding,
                               float *output,
                               int *output_size);

void conv2d_process_neon_nhwc(const float *input,
                            const float *weight,
                            const float *bias,
                            const int output_channel,
                            const int *input_size,
                            const int *kernel_size,
                            const int *stride,
                            const int *padding,
                            float *output,
                            int *output_size);

void conv2d_process_neon_pld_nhwc(const float *input,
                                  const float *weight,
                                  const float *bias,
                                  const int output_channel,
                                  const int *input_size,
                                  const int *kernel_size,
                                  const int *stride,
                                  const int *padding,
                                  float *output,
                                  int *output_size);


void conv2d_process_neon_v2_nhwc(const float *input,
                              const float *weight,
                              const float *bias,
                              const int output_channel,
                              const int *input_size,
                              const int *kernel_size,
                              const int *stride,
                              const int *padding,
                              float *output,
                              int *output_size);

void conv2d_process_img2col(const float *input,
                            const float *weight,
                            const float *bias,
                            const int output_channel,
                            const int *input_size,
                            const int *kernel_size,
                            const int *stride,
                            const int *padding,
                            float *output,
                            int *output_size);

void conv2d_process_basic_fixed(const short *input,
                                const short *weight,
                                const short *bias,
                                const int output_channel,
                                const int *input_size,
                                const int *kernel_size,
                                const int *stride,
                                const int *padding,
                                int32_t *output,
                                int *output_size);

void conv2d_process_basic_nhwc_fixed(const short *input,
                                     const short *weight,
                                     const short *bias,
                                     const int output_channel,
                                     const int *input_size,
                                     const int *kernel_size,
                                     const int *stride,
                                     const int *padding,
                                     int32_t *output,
                                     int *output_size);

void conv2d_process_neon_nhwc_fixed(const short *input,
                                    const short *weight,
                                    const short *bias,
                                    const int output_channel,
                                    const int *input_size,
                                    const int *kernel_size,
                                    const int *stride,
                                    const int *padding,
                                    int32_t *output,
                                    int *output_size);

void conv2d_process_neon_pld_nhwc_fixed(const short *input,
                                        const short *weight,
                                        const short *bias,
                                        const int output_channel,
                                        const int *input_size,
                                        const int *kernel_size,
                                        const int *stride,
                                        const int *padding,
                                        int32_t *output,
                                        int *output_size);

void conv2d_process_neon_v2_nhwc_fixed(const short *input,
                                       const short *weight,
                                       const short *bias,
                                       const int output_channel,
                                       const int *input_size,
                                       const int *kernel_size,
                                       const int *stride,
                                       const int *padding,
                                       int32_t *output,
                                       int *output_size);

#endif
