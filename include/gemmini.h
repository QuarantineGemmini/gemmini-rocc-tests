// See LICENSE for license details.
//============================================================================
// - this file contains the main entry point for user code targeting gemmini
// - several useful utilities for user code are included (like "matmul")
// - it abstracts away details of the underlying implementation, which can be:
//     1) original gemmini software tiler (original isa)
//     2) new gemmini fsm tiler (original isa)
//     3) new hardware tiler (new isa)
//============================================================================

#ifndef __GEMMINI_H__
#define __GEMMINI_H__

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "gemmini_params.h"
#include "gemmini_isa.h"

//============================================================================
// pk/linux page-fault prevention mechanisms
// - YOU MUST PIN ALL MEMORY IN PK/LINUX BEFORE MVIN OR MVOUT!
// - gemmini cannot raise a page-fault, since there is currently no mechanism
//   to pass the requested vaddr to the rocket-chip from a rocc-accelerator
// - in the future, it might be worth implementing support in rocket-core and
//   gemmini for accelerator-initiated page-fault handling, but this seems
//   like a lot of work.
//============================================================================
static bool all_pinned = false;

#ifdef GEMMINI_LINUX
#include <sys/mman.h>
static inline void pin_all() {
  if(all_pinned) return;
  all_pinned = true;
  if (mlockall(MCL_CURRENT) != 0) {
    perror("mlockall failed");
    exit(1);
  }
}
static inline void unpin_all() {
  if(!all_pinned) return;
  all_pinned = false;
  if (munlockall()) {
    perror("munlockall failed");
    exit(1);
  }
}
#define pin_matrices(M,N,K,A,B,D,C,r) do {} while(0)
#define unpin_matrices() do {} while(0)
#else
#ifdef GEMMINI_PK
#define PAGESIZE 0x1000
#define pin_all() do {} while(0)
#define unpin_all() do {} while(0)
static inline void __pin_vector(const char*vec, size_t len) {
  volatile char item[4];
  size_t i;
  for(i=3*PAGESIZE; i<len; i+=4*PAGESIZE) {
    item[0] = vec[i-3*PAGESIZE];
    item[1] = vec[i-2*PAGESIZE];
    item[2] = vec[i-1*PAGESIZE];
    item[3] = vec[i-0*PAGESIZE];
  }
  if(i-3*PAGESIZE < len) item[0] = vec[i-3*PAGESIZE];
  if(i-2*PAGESIZE < len) item[1] = vec[i-2*PAGESIZE];
  if(i-1*PAGESIZE < len) item[2] = vec[i-1*PAGESIZE];
                         item[3] = vec[len-1];
}
static inline void pin_matrices(size_t M, size_t N, size_t K,
        const elem_t A[M][K], const elem_t B[K][N],
        const acc_t * D, elem_t C[M][N], bool repeating_bias) 
{
  // this is really inefficient, but we don't have mlockall() in newlib, so the
  // best we can do is just touch every page before the accelerator uses it
  const char* A_vec = (const char*)A;
  const char* B_vec = (const char*)B;
  const char* C_vec = (const char*)C;
  size_t A_len = sizeof(elem_t)*M*K;
  size_t B_len = sizeof(elem_t)*K*N;
  size_t C_len = sizeof(elem_t)*M*N;
  __pin_vector(A_vec, A_len);
  __pin_vector(B_vec, B_len);
  __pin_vector(C_vec, C_len);

  const char* D_vec = (const char*)D;
  size_t D_len = sizeof(acc_t)*(repeating_bias ? N : M*N);
  if(D != NULL) 
    __pin_vector(D_vec, D_len);
}
#define unpin_matrices() do {} while(0)
#else 
// GEMMINI_BAREMETAL
#define pin_all() do {} while(0)
#define unpin_all() do {} while(0)
#define pin_matrices(M,N,K,A,B,D,C,r) do {} while(0)
#define unpin_matrices() do {} while(0)
#endif // GEMMINI_PK
#endif // GEMMINI_LINUX

//============================================================================
// miscellaneous utility functions
//============================================================================

// Matmul utility functions
void matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], int64_t C_full[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void matmul_short(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C[r][c] += A[r][k]*B[k][c];
    }
}

void matmul_full(elem_t A[DIM][DIM], elem_t B[DIM][DIM], int64_t D[DIM][DIM], int64_t C_full[DIM][DIM]) {
  // Identical to the other matmul function, but with a 64-bit bias
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void matadd(int64_t sum[DIM][DIM], int64_t m1[DIM][DIM], int64_t m2[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      sum[r][c] = m1[r][c] + m2[r][c];
}

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ({(shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift)));})

// THIS IS A ROUNDING SHIFT! It also performs a saturating cast
void matshift(int64_t full[DIM][DIM], elem_t out[DIM][DIM], int shift) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      // Bitshift and round element
      int64_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
      int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
    }
}

void matrelu(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[r][c] = in[r][c] > 0 ? in[r][c] : 0;
}

void matrelu6(elem_t in[DIM][DIM], elem_t out[DIM][DIM], int scale) {
  // int max = 6;
  int max = 6 * scale;

  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++) {
      elem_t positive = in[r][c] > 0 ? in[r][c] : 0;
      out[r][c] = positive > max ? max : positive;
    }
}

void transpose(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
  for (size_t r = 0; r < DIM; r++)
    for (size_t c = 0; c < DIM; c++)
      out[c][r] = in[r][c];
}

void printMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int is_equal(elem_t x[DIM][DIM], elem_t y[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j)
      if (x[i][j] != y[i][j])
          return 0;
  return 1;
}

// This is a GNU extension known as statment expressions
#define MAT_IS_EQUAL(dim_i, dim_j, x, y) \
    ({int result = 1; \
      for (size_t i = 0; i < dim_i; i++) \
        for (size_t j = 0; j < dim_j; ++j) \
          if (x[i][j] != y[i][j]) { \
            result = 0; \
            printf("C[%d][%d]=%d, gold[%d][%d]=%d\n", i, j, x[i][j], i, j, y[i][j]); \
            break; \
          } \
      result;})

int rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}

uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}

static void matmul_cpu(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J], const void * D,
        elem_t C[dim_I][dim_J],
        int act, int shift, bool repeating_bias) {

  const bool no_bias = D == NULL;

  for (size_t i = 0; i < dim_I; i++) {
    for (size_t j = 0; j < dim_J; j++) {
      size_t bias_row = repeating_bias ? 0 : i;
      acc_t result = no_bias ? 0 : ((acc_t (*)[dim_J])D)[bias_row][j];

      for (size_t k = 0; k < dim_K; k++) {
        result += A[i][k] * B[k][j];
      }

      // Shift while rounding to nearest integer (ties round to negative infinity)
      result = ROUNDING_RIGHT_SHIFT(result, shift);

      // Clip result
      result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

      // Apply activation function
      if (act == RELU) {
        result = result < 0 ? 0 : result;
      }

      C[i][j] = (elem_t)result;
    }
  }
}

//============================================================================
// tiled_matmul_auto() implementations
//============================================================================

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU};

#ifdef USE_HW_TILER
  #include "include/gemmini_tiler_hw.h"
#else
#ifdef USE_FSM_TILER
  #include "include/gemmini_tiler_fsm.h"
#else
  #include "include/gemmini_tiler_orig.h"
#endif // USE_FSM_TILER
#endif // USE_HW_TILER

#endif // __GEMMINI_H__

