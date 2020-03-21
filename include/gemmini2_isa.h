//===========================================================================
// This contains "tiled_matmul_auto" implemented using the gemmini2 isa
//===========================================================================
#ifndef __GEMMINI2_ISA_H__
#define __GEMMINI2_ISA_H__

#include <stdint.h>
#include <string.h>
#include <stdio.h>

// [ssteffl] TODO: eventually get rid of dependency on gemmini_params.h
#include "include/gemmini_params.h"
#include "include/gemmini.h"

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
static bool is_valid_to_continue(size_t M, size_t N, size_t K, 
                                 const elem_t A[M][K], 
                                 const elem_t B[K][N],
                                 const acc_t * D, elem_t C[M][N],
                                 int act, int shift, bool repeating_bias,
                                 enum tiled_matmul_type_t tiled_matmul_type) {
  // validate inputs
  if(!(M % DIM == 0 && M > 0)) {
    printf("invalid M: %d, not multiple of %d\n", M, DIM);
    exit(1);
  }
  if(!(N % DIM == 0 && N > 0)) {
    printf("invalid N: %d, not multiple of %d\n", N, DIM);
    exit(1);
  }
  if(!(K % DIM == 0 && K > 0)) {
    printf("invalid K: %d, not multiple of %d\n", K, DIM);
    exit(1);
  }

  // basic sanity checks
  if (tiled_matmul_type == OS) {
    printf("Output-stationary dataflow unsupported for EE290 class\n");
    exit(1);
  } else if (tiled_matmul_type == CPU) {
    matmul_cpu(M, N, K, A, B, D, C, 
               act, shift, repeating_bias);
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

  // [ssteffl] TODO: isa needs support for repeating_bias!!!

  // sanitize inputs before starting
  if(is_valid_to_continue(dim_I, dim_J, dim_K, A, B, D, C, 
                          act, shift, repeating_bias, tiled_matmul_type)) {
    // [ssteffl] TODO: do we need to reset every time?
    // gemmini_reset();
    gemmini_config_addr_ab(A, B);
    gemmini_config_addr_cd(C, D);
    gemmini_config_size0(I, J);
    gemmini_config_size1(K);
    // [ssteffl] TODO: we might also want input-stationary, instead of OS...
    gemmini_config_ex(WEIGHT_STATIONARY, act, 0, shift, 0);
    gemmini_compute();
  }
  gemmini_fence();
}

#endif // __GEMMINI2_ISA_H__

