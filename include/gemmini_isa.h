// See LICENSE for license details.
//============================================================================
// - this file contains raw macros to wrap gemmini/gemmini2 instructions
// - the gemmini_params.h file contains hardware-dependent constants that 
//   some of these custom instructions need
//============================================================================

#ifndef __GEMMINI_ISA_H__
#define __GEMMINI_ISA_H__

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// [ssteffl] TODO: generated by chisel generator. fix this dependency.
#include "include/gemmini_params.h"

// Accelerator interface
#include "rocc-software/src/xcustom.h"

//============================================================================
// original gemmini opcodes
//============================================================================
#define k_CONFIG 0
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7
#define k_LOOP_WS 8

//============================================================================
// New Gemmini2 opcodes
//============================================================================
#define k_ADDR_AB  10
#define k_ADDR_CD  11
#define k_SIZE0    12
#define k_SIZE1    13
#define k_RPT_BIAS 14
#define k_RESET    15
#define k_COMPUTE  16
#define k_CFG_A    17
#define k_CFG_B    18
#define k_CFG_C    19
#define k_CFG_D    20

//============================================================================
// config-opcode parameters
//============================================================================
#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2

//============================================================================
// miscellaneous opcode constants
//============================================================================
#define XCUSTOM_ACC 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

//============================================================================
// base rocc insn formats
//============================================================================
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct, 10, 11)

#define ROCC_INSTRUCTION_RD_RS1_RS2(x, rd, rs1, rs2, funct) \
  ROCC_INSTRUCTION_R_R_R(x, rs1, rs2, funct, 10, 11, 12)

//============================================================================
// original gemmini isa
//============================================================================

// mvin
#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, \
      ((uint64_t)  (rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(cols) <<  ADDR_LEN) | \
        ((uint64_t)(spad_addr)), \
      k_MVIN)

#define gemmini_mvin(dram_addr, spad_addr) \
  gemmini_extended_mvin(dram_addr, spad_addr, DIM, DIM)

#define gemmini_block_mvin(dram_addr, spad_addr, len) \
  gemmini_extended_mvin(dram_addr, spad_addr, (len) * DIM, DIM)

// mvout
#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, \
      ((uint64_t)  (rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(cols) <<  ADDR_LEN) | \
        ((uint64_t)(spad_addr)), \
      k_MVOUT)

#define gemmini_mvout(dram_addr, spad_addr) \
  gemmini_extended_mvout(dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
      ((uint64_t)  (A_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(A_cols) <<  ADDR_LEN) | \
        ((uint64_t)(A)), \
      ((uint64_t)  (BD_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(BD_cols) <<  ADDR_LEN) | \
        ((uint64_t)(BD)), \
      k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
      ((uint64_t)  (A_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(A_cols) <<  ADDR_LEN) | \
        ((uint64_t)(A)), \
      ((uint64_t)  (BD_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(BD_cols) <<  ADDR_LEN) | \
        ((uint64_t)(BD)), \
      k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(A, BD) \
  gemmini_extended_compute_preloaded(A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(A, BD) \
  gemmini_extended_compute_accumulated(A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
      ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(BD_cols) << ADDR_LEN) | \
        (uint64_t)(BD), \
      ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | \
        ((uint64_t)(C_cols) << ADDR_LEN) | \
        (uint64_t)(C), \
      k_PRELOAD)

#define gemmini_preload(BD, C) \
  gemmini_extended_preload(BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(C) \
  gemmini_preload(GARBAGE_ADDR, C)

// weight-stationary matmul loop
#define gemmini_loop_ws(A, B, I, J, K, bias) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
      ((uint64_t)(B) << 32) | (A), \
      ((uint64_t)(bias) << 48) | ((uint64_t)(K) << 32) | ((J) << 16) | (I),\
      k_LOOP_WS)

// config
#define gemmini_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, \
    ((uint64_t)(acc_shift) << 32) | \
      ((act) << 3) | \
      ((mode) << 2) | \
      CONFIG_EX,\
    ((uint64_t)(relu6_shift) << 32) | \
      (sys_shift), \
    k_CONFIG)

#define gemmini_config_ld(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)

#define gemmini_config_st(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_ST, stride, k_CONFIG)

// flush
#define gemmini_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

//============================================================================
// Gemmini2 ISA 
//============================================================================
#define gemmini_config_addr_ab(A, B) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_ADDR_AB)

#define gemmini_config_addr_cd(C, D) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, C, D, k_ADDR_CD)

#define gemmini_config_size0(M, N) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, M, N, k_SIZE0)

#define gemmini_config_size1(K) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, K, 0, k_SIZE1)

// [ssteffl] HACK: need better interface for repeating_bias!
#define gemmini_config_repeating_bias(repeating_bias) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, repeating_bias, 0, k_RPT_BIAS)

#define gemmini_compute() \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_COMPUTE)

#define gemmini_config_reset() \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0, k_RESET)

#define gemmini_config_addr_mode(mat, mode, rows, cols, batch_size, channels, padding, kernel_size, stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (((uint64_t)(rows) << 32) | (uint64_t)(cols)), \
  (((uint64_t)(mat) << 60) | ((uint64_t)(mode) << 56) | ((uint64_t)(stride) << 48) | ((uint64_t)(padding) << 40) | ((uint64_t)(channels) << 32) | \
   ((uint64_t)(kernel_size)) << 16 | ((uint64_t)batch_size)), \
    k_CFG_A)

#endif // __GEMMINI_ISA_H__

