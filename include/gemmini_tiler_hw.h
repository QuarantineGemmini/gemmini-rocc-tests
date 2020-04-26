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

#include "include/util.h"

//============================================================================
// Input Validation
//============================================================================
static bool is_valid_to_continue(
  size_t M, size_t N, size_t K, 
  const elem_t *A, const elem_t *B, const acc_t *D, elem_t *C,
  int act, size_t shift, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t mm_type) 
{
  // basic sanity checks
  ASSERT(mm_type != OS, "gemmini does not support OS dataflow!");

  if (mm_type == CPU) {
    matmul_cpu_raw(
      M, N, K, A, B, D, C, act, shift, relu6_shift, repeating_bias);
    return false;
  }
  return true;
}

//============================================================================
// Entry Point
//============================================================================
static void tiled_matmul_auto_raw(
  size_t M, size_t N, size_t K,
  const elem_t *A, const elem_t *B, const acc_t *D, elem_t *C,
  int act, size_t shift, size_t relu6_shift, bool repeating_bias,
  enum tiled_matmul_type_t mm_type) 
{
  DEBUG("tiled_matmul_auto started M,N,K=(%d,%d,%d)", M, N, K);

  // sanitize inputs before starting
  if(is_valid_to_continue(M, N, K, A, B, D, C, act, shift, 
                          relu6_shift, repeating_bias, mm_type)) {
    // [ssteffl] TODO: should we reset every time?
    // [df] TODO: not if we want to retain config state, e.g. im2col addressing
    // gemmini_config_reset();
    gemmini_config_addr_ab((uintptr_t)A, (uintptr_t)B);
    gemmini_config_addr_cd((uintptr_t)C, (uintptr_t)D);
    gemmini_config_size0(M, N);
    gemmini_config_size1(K);
    gemmini_config_repeating_bias(repeating_bias);
    // [ssteffl] TODO: we might also want input-stationary, but not OS... 
    printf("GEMMINI_SIZES:    - [%u, %u, %u]\n", dim_I, dim_K, dim_J); // YAML 
    gemmini_config_ex(WEIGHT_STATIONARY, act, 0, shift, relu6_shift);
    gemmini_compute();
  } else {
    printf("Invalid Gemmini Matmul");
  }
  gemmini_fence();
} 

#endif // __GEMMINI_TILER_HW_H__

