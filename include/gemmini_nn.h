#ifndef GEMMINI_NN_H
#define GEMMINI_NN_H

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "include/gemmini.h"

struct ConvParams {
    int batch_size;
    int in_dim, out_dim;
    int kernel_size;
    int in_channels;
    int out_channels;
    int stride;
    int padding;
    bool bias;
    bool depthwise;
    int n_patches;
    int patch_size;
    int output_scale;
    int res_scale;
    int pool_size, pool_stride, pool_padding, out_dim_pooled;
    
    int I, J, K;
};

struct ConvIndex {
    int n_batch;
    int row;
    int col;
    int chan;
    bool valid;
};

struct FcParams {
    int batch_size;
    int in_features;
    int out_features;
    int output_scale;
    bool bias;

    int I, J, K;
};

#define HIST_IMAGES(IMAGES) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(IMAGES)/sizeof(IMAGES[0]); i++) { \
            for (int j = 0; j < sizeof(IMAGES[0])/sizeof(IMAGES[0][0]); j++) { \
                for (int k = 0; k < sizeof(IMAGES[0][0])/sizeof(IMAGES[0][0][0]); k++) { \
                    for (int l = 0; l < sizeof(IMAGES[0][0][0])/sizeof(IMAGES[0][0][0][0]); l++) { \
                        if (IMAGES[i][j][k][l] == num) { \
                            count++; \
                        } \
                    } \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }

#define HIST_MATRIX(MATRIX) \
    for (int num = -128; num <= 127; num++) { \
        int count = 0; \
        for (int i = 0; i < sizeof(MATRIX)/sizeof(MATRIX[0]); i++) { \
            for (int j = 0; j < sizeof(MATRIX[0])/sizeof(MATRIX[0][0]); j++) { \
                if (MATRIX[i][j] == num) { \
                    count++; \
                } \
            } \
        } \
        if (count > 0) \
            printf("%d: %d times\n", num, count); \
    }

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_nn_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, size_t shift, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul_auto(dim_I, dim_J, dim_K,
        A, B, D, C, act, shift, relu6_shift, repeating_bias,
        tiled_matmul_type);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            A, B, D, gold, act, shift, relu6_shift, repeating_bias,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

static void conv_dw(size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t in_dim, const size_t out_dim, const size_t kernel_size,
    const elem_t input[batch_size][in_dim][in_dim][channels],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

                    if (shifted > elem_t_max) {
                        shifted = elem_t_max;
                    } else if (shifted < elem_t_min) {
                        shifted = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = shifted;
                    // output[batch][out_row][out_col][channel] = shifted;
                }
            }
        }
    }
}

static void conv_dw_with_col2im(size_t prev_I, size_t prev_J, size_t I, size_t J,
    const size_t batch_size, const size_t channels, const size_t out_dim, const size_t kernel_size,
    const elem_t input[prev_I][prev_J],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_dim && in_col >= 0 && in_col < params->in_dim) {
                                // result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];

                                size_t r = batch * params->in_dim * params->in_dim + in_row * params->in_dim + in_col;

                                result += input[r][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t shifted = ROUNDING_RIGHT_SHIFT(result, params->output_scale);

                    if (shifted > elem_t_max) {
                        shifted = elem_t_max;
                    } else if (shifted < elem_t_min) {
                        shifted = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_dim * params->out_dim + out_row * params->out_dim + out_col;
                    output[r][channel] = shifted;
                    // output[batch][out_row][out_col][channel] = shifted;
                }
            }
        }
    }
}

struct ConvIndex im2col_index(int row, int col, const struct ConvParams* params) {
    /* 
    Calculate input-tensor indicies n_batch, row, col, channel, 
    from im2col output indicies row, col 
    Returns a `ConvIndex` with valid=true for elements in the input tensor, 
    and valid=false for non-existent entries, e.g. those from padding. 
    */

    // This first bit can eventually be done offline per `ConvParams` struct, each of which are generally widely re-used. 
    int kernels_per_row = (params->in_dim - params->kernel_size + 2*params->padding) / params->stride + 1; // rows
    int kernels_per_col = (params->in_dim - params->kernel_size + 2*params->padding) / params->stride + 1; // cols
    int output_rows_per_batch = kernels_per_row * kernels_per_col;
    
    // Calculate our batch and channel index
    int ks = params->kernel_size;
    int chan = col / (ks*ks);
    int n_batch = row / output_rows_per_batch;

    if (n_batch >= params->batch_size || chan >= params->in_channels) {
        // Batch or channel out-of-bounds due to output-matrix size-padding
        struct ConvIndex ci = {.n_batch=-1, .row=-1, .col=-1, .chan=-1, .valid=false}; 
        return ci;
    }

    // Figure out which of the rows/filters/kernels this index maps onto 
    int row_in_batch = row % output_rows_per_batch;
    
    // Find the element position within that kernel 
    int idx_in_kernel = col % (ks*ks);
    int row_in_kernel = idx_in_kernel / ks;
    int col_in_kernel = idx_in_kernel % ks;
    
    // Figure out the kernel's top-left indices
    int kernel_start_row = params->stride * (row_in_batch / kernels_per_row) - params->padding;
    int kernel_start_col = params->stride * (row_in_batch % kernels_per_row) - params->padding;

    // And finally, total the start-index and offset
    int input_row = kernel_start_row + row_in_kernel;
    int input_col = kernel_start_col + col_in_kernel;

    if (input_row < 0 || input_row >= params->in_dim || 
        input_col < 0 || input_col >= params->in_dim ) {
        // Out of bounds due to padding
        struct ConvIndex ci = {.n_batch=-1, .row=-1, .col=-1, .chan=-1, .valid=false}; 
        return ci;
    } 
    // Making it here means we've got a valid entry in the input tensor. 
    struct ConvIndex ci = {.n_batch=n_batch, .row=input_row, .col=input_col, .chan=chan, .valid=true}; 
    return ci; 
}

static void im2col(size_t batch_size, size_t channels, size_t im_dim,
    size_t I, size_t K,
    const elem_t input[batch_size][im_dim][im_dim][channels],
    elem_t output[I][K],
    const struct ConvParams * params)
{
    printf("IM2COL_PARAMS:    - \n");
    printf("IM2COL_PARAMS:      batch_size: %d \n", params->batch_size);
    printf("IM2COL_PARAMS:      padding: %d \n", params->padding);
    printf("IM2COL_PARAMS:      in_dim: %d \n", params->in_dim);
    printf("IM2COL_PARAMS:      kernel_size: %d \n", params->kernel_size);
    printf("IM2COL_PARAMS:      stride: %d \n", params->stride);
    printf("IM2COL_PARAMS:      in_channels: %d \n", params->in_channels);
    printf("IM2COL_PARAMS:      I: %d \n", I);
    printf("IM2COL_PARAMS:      K: %d \n", K);
    
    // Output-referred im2col edition 
    // FIXME: consistency checks for sizes
    // I >= output_rows_per_batch * params-> n_batches
    // K > something else 
    
    for (int row = 0; row < I; row++) {
        for (int col = 0; col < K; col++) {
            struct ConvIndex ii = im2col_index(row, col, params);
            if (ii.valid) output[row][col] = input[ii.n_batch][ii.row][ii.col][ii.chan];
            else output[row][col] = 0;
        }
    }
}

// Enumeration of which of our four matrices are being configured per CONFIG_ADDR_MODE
typedef enum {CFG_A=0, CFG_B=1, CFG_C=2, CFG_D=3} WhichMatrix;

#ifdef USE_HW_TILER

void setup_im2col_addr_mode(const WhichMatrix mat, const struct ConvParams * params)
{
    printf("IM2COL CONFIG CMD\n");
    printf("IM2COL CONFIG PARAMS:    - [%u, %u, %u, %u, %u, %u]\n", 
        params->batch_size, params->padding, params->in_dim, params->kernel_size, params->stride, params->in_channels); // YAML-ish 
    gemmini_config_addr_mode(mat, 1, params->in_dim, params->in_dim, params->batch_size, params->in_channels, params->padding, params->kernel_size, params->stride);
}

static void im2col_and_matmul(  // This parameter list is, well, a mouthful. 
    size_t batch_size, size_t channels, size_t im_dim,
    const elem_t input[batch_size][im_dim][im_dim][channels],
    const struct ConvParams * params,
    size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
    const void * D, elem_t C[dim_I][dim_J],
    int act, size_t shift, size_t relu6_shift, bool repeating_bias,
    enum tiled_matmul_type_t tiled_matmul_type,
    bool check, char * layer_name,
    uint64_t * im2col_cycles, uint64_t * matmul_cycles, WhichMatrix mat)
{
    // Im2Col + Matmul, Hardware Edition
    // "Im2Col" here just sends an addressing-config command to Gemmini
    // Note the `A` parameter is ignored altogether. 
    // The address of `input` is instead passed to Gemmini for inline im2col-ing. 

    uint64_t start, end;
    start = read_cycles();
    setup_im2col_addr_mode(mat, params);
    end = read_cycles();
    *im2col_cycles += end - start;

    start = read_cycles();
    tiled_matmul_nn_auto(params->I, params->J, params->K,
        input, B, D, C,  // Note `A` is ignored
        act, shift, relu6_shift, repeating_bias, 
        tiled_matmul_type, check, layer_name);
    end = read_cycles();
    *matmul_cycles += end - start;

    // Undo our addressing-mode changes
    gemmini_config_reset();
}

#else 

static void im2col_and_matmul(  // This parameter list is, well, a mouthful. 
    size_t batch_size, size_t channels, size_t im_dim,
    const elem_t input[batch_size][im_dim][im_dim][channels],
    const struct ConvParams * params,
    size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
    const void * D, elem_t C[dim_I][dim_J],
    int act, size_t shift, size_t relu6_shift, bool repeating_bias,
    enum tiled_matmul_type_t tiled_matmul_type,
    bool check, char * layer_name,
    uint64_t * im2col_cycles, uint64_t * matmul_cycles, WhichMatrix mat)
{
    // Only supports software-im2col on A-matrix; throw an error otherwise
    if (mat != 0) {
        printf("GEMMINI ERROR: software im2col+matmul supported on A-matrix only");
        exit(1);
    }

    uint64_t start, end;
    start = read_cycles();
    im2col(params->batch_size, params->in_channels, params->in_dim,
        params->I, params->K,
        input, A, params);
    end = read_cycles();
    *im2col_cycles += end - start;

    start = read_cycles();
    tiled_matmul_nn_auto(params->I, params->J, params->K,
        A, B, D, C, 
        act, shift, relu6_shift, repeating_bias,
        tiled_matmul_type, check, layer_name);
    end = read_cycles();
    *matmul_cycles += end - start;
}

#endif // USE_HW_TILER

static void im2col_with_col2im(size_t prev_I, size_t prev_J,
    size_t next_I, size_t next_K,
    const elem_t input[prev_I][prev_J],
    elem_t output[next_I][next_K],
    const struct ConvParams * params)
{
    printf("IM2COL_COL2IM_PARAMS:    - [%u, %u, %u, %u, %u, %u, %u, %u, %u, %u]\n", prev_I, prev_J, next_I, next_K, 
        params->batch_size, params->padding, params->in_dim, params->kernel_size, params->stride, params->in_channels); // YAML-ish 
    int out_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int out_col = 0;

                for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                    for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                        for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;

                            if (pixel_row < 0 || pixel_row >= params->in_dim
                                || pixel_col < 0 || pixel_col >= params->in_dim) {
                                // output[out_row][out_col] = 0;
                            } else {
                                int in_row = n_batch * params->in_dim * params->in_dim + pixel_row * params->in_dim + pixel_col;
                                int in_col = im_channel;

                                output[out_row][out_col] = input[in_row][in_col];
                            }

                            out_col++;
                        }
                    }
                }

                out_row++;
            }
        }
    }
}

// Compute C = A + B with saturating add
void vecadd(size_t len, const elem_t * A, const elem_t * B, elem_t * C, int A_shift) {
    for (size_t i = 0; i < len; i++) {
        acc_t result = ROUNDING_RIGHT_SHIFT(A[i], A_shift) + B[i];

        if (result > elem_t_max) {
            result = elem_t_max;
        } else if (result < elem_t_min) {
            result = elem_t_min;
        }

        C[i] = result;
    }
}

void resadd1(const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[batch_size][im_dim][im_dim][channels],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    acc_t result = ROUNDING_RIGHT_SHIFT(A[batch][row][col][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd2(const size_t I, const size_t J,
    const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[I][J],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = ROUNDING_RIGHT_SHIFT(A[r][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd3(const size_t I, const size_t J,
    const elem_t A[I][J],
    const elem_t B[I][J],
    elem_t C[I][J],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = ROUNDING_RIGHT_SHIFT(A[r][channel], params->res_scale) + B[r][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[r][channel] = result;
                }
            }
        }
    }
}

// Pooling
void pool(size_t batch_size, size_t channels, size_t in_dim, size_t out_dim,
    elem_t input[batch_size][in_dim][in_dim][channels],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    // size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch][in_row][in_col][channel] > result) {
                                    result = input[batch][in_row][in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }
                    
                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

void pool_with_col2im(size_t I, size_t J,
    size_t batch_size, size_t channels, size_t out_dim,
    elem_t input[I][J],
    elem_t output[batch_size][out_dim][out_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    size_t in_dim = params->out_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_dim; out_row++) {
                for (int out_col = 0; out_col < out_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_dim && in_col >= 0 && in_col < in_dim) {
                                if (input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel] > result) {
                                    result = input[batch * in_dim * in_dim + in_row * in_dim + in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

#endif // GEMMINI_NN_H

