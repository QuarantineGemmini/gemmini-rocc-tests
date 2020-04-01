// See LICENSE for license details.
//============================================================================
// - this file contains the original gemmini matmul tiler (original isa)
// - this doesn't need "#includes" since its included at the end of gemmini.h
//============================================================================
#ifndef __GEMMINI_TILER_ORIG_H__
#define __GEMMINI_TILER_ORIG_H__

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

//============================================================================
// Tiling functions
//============================================================================
static void sp_tiled_matmul_ws(const elem_t * A, const elem_t * B,
        const acc_t * D, elem_t * C,
        size_t I, size_t J, size_t K, 
        size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_len, size_t B_row_len, size_t D_row_len, size_t C_row_len,
        bool no_bias, bool repeating_bias) 
{
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = I * K * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  const int A_blocks = min(K, MAX_BLOCK_LEN);
  const int B_blocks = min(J, MAX_BLOCK_LEN);
  const int D_blocks = min(J, MAX_BLOCK_LEN_ACC);

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_len * sizeof(acc_t);
    gemmini_config_ld(D_stride);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_len + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j == J-1 ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);

        gemmini_extended_mvin(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }

  // Move-in B
  gemmini_config_ld(B_row_len * sizeof(elem_t));
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_len + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j == J-1 ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      gemmini_extended_mvin(B_dram_addr, B_sp_addr, cols, rows);
    }
  }

  // Move-in A
  gemmini_config_ld(A_row_len * sizeof(elem_t));
  for (size_t k = 0; k < K; k += A_blocks) {
    for (size_t i = 0; i < I; i++) {
      const elem_t * const A_dram_addr = A + (i * A_row_len + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k == K-1 ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
    }
  }

  // Compute
  // gemmini_loop_ws(A_sp_addr_start, B_sp_addr_start, I, J, K, !no_bias || D == NULL);

  // The above "gemmini_loop_ws" command will be unrolled in hardware into the
  // following loop:
  for (size_t j = 0; j < J; j++) {
    for (size_t k = 0; k < K; k++) {
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
        uint32_t out_sp_addr = C_sp_addr;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == 0;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_preload(
            pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);

        if (i == 0) { // First iteration
          gemmini_extended_compute_preloaded(
              A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        } else { // All other iterations
          gemmini_extended_compute_accumulated(
              A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        elem_t * const C_dram_addr = C + (i*C_row_len + j)*DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_cols, C_rows);
      }
    }
  }
}

static void tiled_matmul_outer(size_t dim_I, size_t dim_J, size_t dim_K,
  const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
  const acc_t * D, elem_t C[dim_I][dim_J],
  size_t tile_I, size_t tile_J, size_t tile_K,
  int act, int shift, size_t relu6_shift, bool repeating_bias) 
{
  // quantized
  const size_t dim_I_pad = round_up(dim_I, DIM);
  const size_t dim_J_pad = round_up(dim_J, DIM);
  const size_t dim_K_pad = round_up(dim_K, DIM);

  // how many iterations for each direction
  const size_t I0 = div_round_up(dim_I_pad, tile_I*DIM);
  const size_t J0 = div_round_up(dim_J_pad, tile_J*DIM);
  const size_t K0 = div_round_up(dim_K_pad, tile_K*DIM);

  // how many tiles in the block for each direction in the LAST ITERATION
  const size_t last_I = ((dim_I_pad / DIM) % tile_I) || tile_I;
  const size_t last_J = ((dim_J_pad / DIM) % tile_J) || tile_J;
  const size_t last_K = ((dim_K_pad / DIM) % tile_K) || tile_K;

  // how much padding the hardware is supposed to add for the final tile
  const size_t pad_I = dim_I_pad - dim_I;
  const size_t pad_J = dim_J_pad - dim_J;
  const size_t pad_K = dim_K_pad - dim_K;

  const bool no_bias = (D == NULL);

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  gemmini_config_ex(WS, act, 0, shift, relu6_shift);
  gemmini_config_st(dim_J * sizeof(elem_t));

  for (size_t i0 = 0; i0 < I0; i0++) {
    for (size_t j0 = 0; j0 < J0; j0++) {
      for (size_t k0 = 0; k0 < K0; k0++) {

        const acc_t * pre = NULL;
        if (k0 == 0) {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          pre = &((acc_t (*)[dim_J])D)[bias_row][j0*tile_J*DIM];
        }
        elem_t * out = (k0 == K0-1) ? &C[i0*tile_I*DIM][j0*tile_J*DIM] : NULL;

        const size_t I     = (i0 < I0-1) ? tile_I : last_I;
        const size_t J     = (j0 < J0-1) ? tile_J : last_J;
        const size_t K     = (k0 < K0-1) ? tile_K : last_K;

        const size_t pad_I = (i0 < I0-1) ? 0      : pad_I;
        const size_t pad_J = (j0 < J0-1) ? 0      : pad_J;
        const size_t pad_K = (k0 < K0-1) ? 0      : pad_K;

        sp_tiled_matmul_ws(&A[i0*tile_I*DIM][k0*tile_K*DIM],
            &B[k0*tile_K*DIM][j0*tile_J*DIM],
            pre, out,
            I, J, K,
            pad_I, pad_J, pad_K,
            dim_K, dim_J, dim_J, dim_J,
            no_bias, repeating_bias);
      }
    }
  }
  gemmini_fence();
}

// This function runs a tiled matrix multiplication, with hardcoded tiling
// factors
void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const acc_t * D, elem_t C[dim_I][dim_J],
        int act, size_t shift, size_t relu6_shift, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I + DIM - 1) / DIM;
  const size_t dim_J_padded = (dim_J + DIM - 1) / DIM;
  const size_t dim_K_padded = (dim_K + DIM - 1) / DIM;

  if (tile_I * DIM - dim_I > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > BANK_NUM * BANK_ROWS) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > ACC_ROWS) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, ");
    printf("to fit within the bounds of the LOOP_WS function\n");
    exit(1);
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS) {
    printf("gemmini2 does not support output-stationary dataflow!\n");
    exit(1);
  } 
  else if(tiled_matmul_type == WS) {
      tiled_matmul_outer(dim_I, dim_J, dim_K,
              A, B, D, C,
              tile_I, tile_J, tile_K,
              act, shift, relu6_shift, repeating_bias);
  } 
  else /*if (tiled_matmul_type == CPU)*/ {
      matmul_cpu(dim_I, dim_J, dim_K,
              A, B, D, C,
              act, shift, relu6_shift, repeating_bias);
  }
}

//============================================================================
// Entry Point
//============================================================================
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const acc_t * D, elem_t C[dim_I][dim_J],
        int act, size_t shift, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type)
{
  // [ssteffl] NOTE: this requires the output-group to be a square. inefficient
  // for tall/skinny or wide/short output matrices.
  const size_t partition_rows    = BANK_NUM * BANK_ROWS / 2;
  const size_t mats_in_partition = partition_rows / DIM;
  const size_t mats_in_acc       = ACC_ROWS / DIM;
  const size_t max_tile_i_j      = (int)sqrt(mats_in_acc);
  const size_t max_tile_k        = mats_in_partition / max_tile_i_j;

  // how many DIMxDIM tiles in each direction
  const size_t dim_I_padded = ((dim_I + DIM - 1) / DIM);
  const size_t dim_J_padded = ((dim_J + DIM - 1) / DIM);
  const size_t dim_K_padded = ((dim_K + DIM - 1) / DIM);

  // how many tiles per matmul block (acc-output=IxJ, sp is 2-way split=Kx1)
  const size_t tile_I = min(dim_I_padded, max_tile_i_j);
  const size_t tile_J = min(dim_J_padded, max_tile_i_j);
  const size_t tile_K = min(dim_K_padded, max_tile_k);

  tiled_matmul(dim_I, dim_J, dim_K,
      A, B, D, C, act, shift, relu6_shift, repeating_bias,
      tile_I, tile_J, tile_K,
      tiled_matmul_type);
}


#endif // __GEMMINI_TILER_ORIG_H__

