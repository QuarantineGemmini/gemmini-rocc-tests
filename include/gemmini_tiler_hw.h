// See LICENSE for license details.
//===========================================================================
// - contains "tiled_matmul_auto" implemented in hw using the gemmini2 isa
// - this doesn't need "#includes" since its included at the end of gemmini.h
//===========================================================================
#ifndef __GEMMINI_TILER_HW_H__
#define __GEMMINI_TILER_HW_H__

#include <stdint.h>
#include <string.h>
#include <stdio.h>

//===========================================================================
// debugging printfs
//===========================================================================
#ifdef NODEBUG
#define DBG(...)
#else
#define DBG(...) printf(__VA_ARGS__)
#endif

//============================================================================
// Input Validation
//============================================================================
static bool is_valid_to_continue(
  size_t M, size_t N, size_t K, 
  const elem_t A[M][K], const elem_t B[K][N],
  const acc_t * D, elem_t C[M][N],
  int act, int shift, bool repeating_bias,
  enum tiled_matmul_type_t tiled_matmul_type) 
{
  // basic sanity checks
  if (tiled_matmul_type == OS) {
    printf("Output-stationary dataflow unsupported!\n");
    exit(1);
  } else if (tiled_matmul_type == CPU) {
    matmul_cpu(M, N, K, A, B, D, C, act, shift, repeating_bias);
    return false;
  }
  return true;
}

//============================================================================
// Entry Point
//============================================================================
static void 
tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                  const elem_t A[dim_I][dim_K], 
                  const elem_t B[dim_K][dim_J],
                  const acc_t * D, elem_t C[dim_I][dim_J],
                  int act, int shift, bool repeating_bias,
                  enum tiled_matmul_type_t tiled_matmul_type) {
  DBG("tiled_matmul_auto started M,N,K=(%d,%d,%d)\n", dim_I, dim_J, dim_K);

  // sanitize inputs before starting
  if(is_valid_to_continue(dim_I, dim_J, dim_K, A, B, D, C, 
                          act, shift, repeating_bias, tiled_matmul_type)) {
    // [ssteffl] TODO: should we reset every time?
    // [df] TODO: not if we want to retain config state, e.g. im2col addressing
    // gemmini_config_reset();
    gemmini_config_addr_ab((uintptr_t)A, (uintptr_t)B);
    gemmini_config_addr_cd((uintptr_t)C, (uintptr_t)D);
    gemmini_config_size0(dim_I, dim_J);
    gemmini_config_size1(dim_K);
    gemmini_config_repeating_bias(repeating_bias);
    // [ssteffl] TODO: we might also want input-stationary, but not OS...
    gemmini_config_ex(WEIGHT_STATIONARY, act, 0, shift, 0);

    printf("GEMMINI_SIZES:    - [%u, %u, %u]\n", dim_I, dim_K, dim_J); // YAML 
    gemmini_compute();
  } else {
    printf("Invalid Gemmini Matmul");
  }
  gemmini_fence();
} 

#endif // __GEMMINI_TILER_HW_H__

