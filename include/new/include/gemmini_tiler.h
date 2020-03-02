//===========================================================================
// This contains "tiled_matmul_auto", but implemented in a way that can
// be easily converted to a FSM in RTL. I am doing this to pipeclean the
// FSM's scheduling algorithm in software before doing it in hardware, to
// get a rough idea of what the hardware performance might be.
//===========================================================================
#ifndef __GEMMINI_TILER_H__
#define __GEMMINI_TILER_H__

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "include/gemmini_params.h"
#include "include/gemmini.h"

//===========================================================================
// convenient macros
//===========================================================================
#define ACC_ADDR_RD(addr)  (((2 << (ADDR_LEN - 2)) | (addr)) & 0xffffffff)
#define ACC_ADDR_NEW(addr) (((2 << (ADDR_LEN - 2)) | (addr)) & 0xffffffff)
#define ACC_ADDR_ACC(addr) (((3 << (ADDR_LEN - 2)) | (addr)) & 0xffffffff)
//#define ACC_ADDR_RD(addr)  ((2 << (ADDR_LEN - 2)) | (addr))
//#define ACC_ADDR_NEW(addr) ((2 << (ADDR_LEN - 2)) | (addr))
//#define ACC_ADDR_ACC(addr) ((3 << (ADDR_LEN - 2)) | (addr))

#define MIN(a,b) ({ __typeof__ (a) _a = (a); \
                    __typeof__ (b) _b = (b); \
                    _a < _b ? _a : _b; })

//===========================================================================
// debugging printfs
//===========================================================================
#ifdef NODEBUG
#define DBG(...)
#else
#define DBG(...) printf(__VA_ARGS__)
#endif

#define DBG_OG(name)                                                    \
  DBG(name " = (%d,%d), (%d,%d)\n",                                     \
      self->loop1_tile_col_start, self->loop1_tile_row_start,           \
      self->loop1_tile_col_end,   self->loop1_tile_row_end)

#define DBG_LOOP(name)                                                  \
  DBG(name " = (row,col,k), (%d,%d,%d)\n",                              \
      self->gbl_tile_row, self->gbl_tile_col, self->loop2_k_tile_col)

#define DBG_MVIN_B                                                      \
  DBG("        mvin(B,stride=%u,mem=%x,sp=%u)\n",                       \
      B_mem_stride, B_mem_addr, B_sp_row_addr)

#define DBG_MVIN_A                                                      \
  DBG("        mvin(A,stride=%u,mem=%x,sp=%u)\n",                       \
      A_mem_stride, A_mem_addr, A_sp_row_addr)

#define DBG_MVIN_D                                                      \
  DBG("        mvin(D,stride=%u,mem=%x,acc=%u,%u)\n",                   \
      D_mem_stride, D_mem_addr,                                         \
      (D_acc_row_addr >> 30) & 0x3, D_acc_row_addr & 0x3fffffff)

#define DBG_PRELOAD_B                                                   \
  DBG("        preload(B=%u,C=%u,%u)\n", B_sp_row_addr,                 \
      (C_acc_row_addr >> 30) & 0x3, C_acc_row_addr & 0x3fffffff)

#define DBG_COMPUTE_PRE                                                 \
  DBG("        compute.pre(A=%u)\n", A_sp_row_addr)

#define DBG_COMPUTE_ACC                                                 \
  DBG("        compute.acc(A=%u)\n", A_sp_row_addr)

#define DBG_MVOUT_C                                                     \
  DBG("        mvout(C,stride=%u,mem=%x,acc=%u,%u)\n",                  \
      C_mem_stride, C_mem_addr,                                         \
      (C_acc_row_addr >> 30) & 0x3, C_acc_row_addr & 0x3fffffff)

//===========================================================================
// state objects
//===========================================================================
typedef size_t    bytes_t;
typedef size_t    tile_t;
typedef size_t    sp_row_t;
typedef uintptr_t mem_addr_t;

typedef struct gemmini {
  //-------------------------------------
  // hardware-specific global constants
  sp_row_t    GBL_B_SP_ROW_ADDR_1;    // sp row of 1st tmp slot for B-tiles
  sp_row_t    GBL_B_SP_ROW_ADDR_2;    // sp row of 2nd tmp slot for B-tiles

  tile_t      TILE_COLS_PER_GROUP;    // num tiles wide is an output-group
  tile_t      TILE_ROWS_PER_GROUP;    // num tiles tall is an output-group

  bytes_t     BYTE_ROWS_PER_TILE;     // num-rows of tile A,B,C,D
  bytes_t     I_BYTE_COLS_PER_GROUP;  // byte-width of output-group A,B,C
  bytes_t     O_BYTE_COLS_PER_GROUP;  // byte-width of output-group D
  bytes_t     I_TILE_BYTE_WIDTH;      // byte-width of tile A,B,C
  bytes_t     O_TILE_BYTE_WIDTH;      // byte-width of tile D

  //-------------------------------------
  // input data-specific global constants
  bool        HAS_BIAS;               // if computing A*B+D=C, not A*B=C
  bool        REPEATING_BIAS;         // if HAS_BIAS, repeat 1st row only

  int         DATAFLOW;
  int         ACTIVATION;
  int         SYSTOLIC_OUT_RSHIFT;
  int         ACC_OUT_RSHIFT;
  int         RELU6_IN_LSHIFT;

  mem_addr_t  A_MEM_ADDR;             // mem-addr of A-matrix
  mem_addr_t  B_MEM_ADDR;
  mem_addr_t  C_MEM_ADDR;
  mem_addr_t  D_MEM_ADDR;
  bytes_t     A_BYTES_PER_TILE_ROW;   // bytes in A-matrix row * rows-per-tile
  bytes_t     B_BYTES_PER_TILE_ROW;
  bytes_t     C_BYTES_PER_TILE_ROW;
  bytes_t     D_BYTES_PER_TILE_ROW;
  bytes_t     A_BYTES_PER_ROW;        // bytes in A-matrix row
  bytes_t     B_BYTES_PER_ROW;
  bytes_t     C_BYTES_PER_ROW;
  bytes_t     D_BYTES_PER_ROW;

  tile_t      TILE_COL_END;           // last x tile index in C matrix
  tile_t      TILE_ROW_END;           // last y tile index in C matrix
  tile_t      K_TILE_COL_END;         // last x tile index in A matrix

  //-------------------------------------
  // global state persistent across all loops
  tile_t      gbl_tile_row;           // current output-group tile x
  tile_t      gbl_tile_col;           // current output-group tile y
  sp_row_t    gbl_B_cur_sp_row_addr;  // which tmp-slot in sp being used now
  sp_row_t    gbl_B_alt_sp_row_addr;  // the other tmp-slot in sp 

  //-------------------------------------
  // global state that is reset for each output-group
  sp_row_t    gbl_CD_acc_row_addr;    // where to put next C/D-tile in acc 

  //-------------------------------------
  // loop1-local state
  tile_t      loop1_tile_col_start;   // upper-left  tile x in output-group
  tile_t      loop1_tile_col_end;     // lower-right tile x in output-group
  tile_t      loop1_tile_row_start;   // upper-left  tile y in output-group
  tile_t      loop1_tile_row_end;     // lower-right tile x in output-group
  mem_addr_t  loop1_A_mem_addr;       // initialized from global-constants
  mem_addr_t  loop1_B_mem_addr;
  mem_addr_t  loop1_C_mem_addr;
  mem_addr_t  loop1_D_mem_addr;

  //-------------------------------------
  // loop2-local state
  tile_t      loop2_k_tile_col;       // which tile-column in A we are in
  mem_addr_t  loop2_A_mem_addr;       // initialized from loop1 values
  mem_addr_t  loop2_B_mem_addr;
  mem_addr_t  loop2_C_mem_addr;
  mem_addr_t  loop2_D_mem_addr;

  //-------------------------------------
  // loop3-local state
  mem_addr_t  loop3_A_mem_addr;       // initialized from loop2 values
  mem_addr_t  loop3_B_mem_addr;
  mem_addr_t  loop3_C_mem_addr;
  mem_addr_t  loop3_D_mem_addr;

  //-------------------------------------
  // loop4-local state
  sp_row_t    loop4_A_sp_row_addr;    // where in the sp is the next A tile
  mem_addr_t  loop4_A_mem_addr;       // initialized from loop3 values
  mem_addr_t  loop4_B_mem_addr;
  mem_addr_t  loop4_C_mem_addr;
  mem_addr_t  loop4_D_mem_addr;

} gemmini_t;

//============================================================================
// create_gemmini subroutines
//============================================================================
static gemmini_t static_self;

static gemmini_t* 
init_gemmini_state(size_t M, size_t N, size_t K, 
                   mem_addr_t A, mem_addr_t B, mem_addr_t C, mem_addr_t D,
                   int act, int shift, bool bias, bool repeating_bias) {
  // create the state struct
  gemmini_t *self = &static_self;
  memset(self, 0, sizeof(gemmini_t));

  // define hardcoded constants
  const size_t I_TILE_BYTE_WIDTH = DIM * sizeof(elem_t);
  const size_t O_TILE_BYTE_WIDTH = DIM * sizeof(acc_t);
  const size_t A_BYTE_WIDTH      = K * sizeof(elem_t);
  const size_t BC_BYTE_WIDTH     = N * sizeof(elem_t);
  const size_t D_BYTE_WIDTH      = N * sizeof(acc_t);

  //------------------------------------------------------------------------
  // hardware specific constants
  //------------------------------------------------------------------------
  self->GBL_B_SP_ROW_ADDR_1   = (BANK_NUM * BANK_ROWS) - 2*DIM;
  self->GBL_B_SP_ROW_ADDR_2   = (BANK_NUM * BANK_ROWS) - 1*DIM;

  self->TILE_ROWS_PER_GROUP   = (BANK_NUM * BANK_ROWS / DIM) - 2;
  self->TILE_COLS_PER_GROUP   = (ACC_ROWS / DIM) / self->TILE_ROWS_PER_GROUP;
  if(self->TILE_COLS_PER_GROUP == 0) {
    // NOTE: this happens if accumulator size < scratchpad size. Don't do 
    //       this! your accumulator should be much larger than scratchpad!
    self->TILE_ROWS_PER_GROUP = 4;
    self->TILE_COLS_PER_GROUP = (ACC_ROWS / DIM) / self->TILE_ROWS_PER_GROUP;
  }

  self->BYTE_ROWS_PER_TILE    = DIM;
  self->I_BYTE_COLS_PER_GROUP = self->TILE_COLS_PER_GROUP * I_TILE_BYTE_WIDTH;
  self->O_BYTE_COLS_PER_GROUP = self->TILE_COLS_PER_GROUP * O_TILE_BYTE_WIDTH;
  self->I_TILE_BYTE_WIDTH     = I_TILE_BYTE_WIDTH;
  self->O_TILE_BYTE_WIDTH     = O_TILE_BYTE_WIDTH;

  //------------------------------------------------------------------------
  // input data-specific constants
  //------------------------------------------------------------------------
  self->HAS_BIAS              = bias;
  self->REPEATING_BIAS        = repeating_bias;

  self->DATAFLOW              = WEIGHT_STATIONARY;
  self->ACTIVATION            = act;
  self->SYSTOLIC_OUT_RSHIFT   = 0;
  self->ACC_OUT_RSHIFT        = shift;
  self->RELU6_IN_LSHIFT       = 0;

  self->A_MEM_ADDR            = A;
  self->B_MEM_ADDR            = B;
  self->C_MEM_ADDR            = C;
  self->D_MEM_ADDR            = D;
  self->A_BYTES_PER_TILE_ROW  = DIM * A_BYTE_WIDTH;
  self->B_BYTES_PER_TILE_ROW  = DIM * BC_BYTE_WIDTH;
  self->C_BYTES_PER_TILE_ROW  = DIM * BC_BYTE_WIDTH;
  self->D_BYTES_PER_TILE_ROW  = DIM * D_BYTE_WIDTH;
  self->A_BYTES_PER_ROW       = A_BYTE_WIDTH;
  self->B_BYTES_PER_ROW       = BC_BYTE_WIDTH;
  self->C_BYTES_PER_ROW       = BC_BYTE_WIDTH;
  self->D_BYTES_PER_ROW       = D_BYTE_WIDTH;

  self->TILE_ROW_END          = (M / DIM) - 1;
  self->TILE_COL_END          = (N / DIM) - 1;
  self->K_TILE_COL_END        = (K / DIM) - 1;

  return self;
}

//============================================================================
// create and destroy gemmini: hides implementation details. NO 
//============================================================================
static gemmini_t* 
create_gemmini(size_t M, size_t N, size_t K,
               const elem_t A[M][K], const elem_t B[K][N],
               const acc_t * D, elem_t C[M][N],
               int act, int shift, bool repeating_bias) {
  // initialize state
  gemmini_t * self = init_gemmini_state(M, N, K,
                                        (mem_addr_t) A, (mem_addr_t) B,
                                        (mem_addr_t) C, (mem_addr_t) D,
                                        act, shift,
                                        (D != NULL), repeating_bias);

  // issue gemini commands
  gemmini_config_ex(self->DATAFLOW, self->ACTIVATION, 
                    self->SYSTOLIC_OUT_RSHIFT, self->ACC_OUT_RSHIFT, 
                    self->RELU6_IN_LSHIFT);
  gemmini_config_st(self->C_BYTES_PER_ROW);

  // state and hardware are now initialized
  return self;
}

static void destroy_gemmini(gemmini_t *self) {
  // nothing to do here
  return;
}

//============================================================================
// Tiling Loop #1 subroutines
//============================================================================
static void reset_output_group(gemmini_t *self) {
  // define mutable global state mutable by all loops, persist across all ogs
  self->gbl_tile_row          = 0;
  self->gbl_tile_col          = 0;
  self->gbl_B_cur_sp_row_addr = self->GBL_B_SP_ROW_ADDR_1;
  self->gbl_B_alt_sp_row_addr = self->GBL_B_SP_ROW_ADDR_2;

  // define mutable global state mutable by all loops, reset after each og
  self->gbl_CD_acc_row_addr = 0;

  // update the start/end tiles for this output-group (inclusive)
  // NOTE: duplicated with next_output_group!!
  self->loop1_tile_col_start = self->gbl_tile_col;
  self->loop1_tile_col_end   = MIN(self->gbl_tile_col + 
                                   self->TILE_COLS_PER_GROUP - 1,
                                   self->TILE_COL_END);
  self->loop1_tile_row_start = self->gbl_tile_row;
  self->loop1_tile_row_end   = MIN(self->gbl_tile_row + 
                                   self->TILE_ROWS_PER_GROUP - 1,
                                   self->TILE_ROW_END);

  // derived pointers to matrices in memory for this og
  self->loop1_A_mem_addr = self->A_MEM_ADDR;
  self->loop1_B_mem_addr = self->B_MEM_ADDR;
  self->loop1_C_mem_addr = self->C_MEM_ADDR;
  self->loop1_D_mem_addr = self->D_MEM_ADDR;

  DBG_OG("reset_output_group");
}

static bool next_output_group(gemmini_t *self) {
  bool did_row_incr = false;
  bool did_col_incr = false;

  if(self->gbl_tile_col == self->TILE_COL_END) {
    if(self->gbl_tile_row == self->TILE_ROW_END) {
      // we finished the last output group. so we're done
      DBG_OG("output_group finished");
      return false;
    } else {
      self->gbl_tile_col = 0;
      self->gbl_tile_row += 1;
      did_row_incr = true;
    }
  } else {
    self->gbl_tile_col += 1;
    self->gbl_tile_row  = self->loop1_tile_row_start;
    did_col_incr = true;
  }

  // reset global state that resets for each new output-group
  self->gbl_CD_acc_row_addr = 0;

  // update the start/end tiles for this output-group (inclusive)
  self->loop1_tile_col_start = self->gbl_tile_col;
  self->loop1_tile_col_end   = MIN(self->gbl_tile_col + 
                                   self->TILE_COLS_PER_GROUP - 1,
                                   self->TILE_COL_END);
  self->loop1_tile_row_start = self->gbl_tile_row;
  self->loop1_tile_row_end   = MIN(self->gbl_tile_row + 
                                   self->TILE_ROWS_PER_GROUP - 1,
                                   self->TILE_ROW_END);

  // update all derived pointers to matrices in memory
  if(did_row_incr) {
    self->loop1_A_mem_addr  = self->A_MEM_ADDR + (self->loop1_tile_row_start *
                              self->A_BYTES_PER_TILE_ROW);
    self->loop1_B_mem_addr  = self->B_MEM_ADDR;
    self->loop1_C_mem_addr  = self->C_MEM_ADDR + (self->loop1_tile_row_start *
                              self->C_BYTES_PER_TILE_ROW);
    self->loop1_D_mem_addr  = !self->HAS_BIAS ? 0 :
                              (self->REPEATING_BIAS ? self->D_MEM_ADDR :
                               self->D_MEM_ADDR + (self->loop1_tile_row_start *
                               self->D_BYTES_PER_TILE_ROW));
  } else if(did_col_incr) {
    self->loop1_A_mem_addr += 0;
    self->loop1_B_mem_addr += self->I_BYTE_COLS_PER_GROUP;
    self->loop1_C_mem_addr += self->I_BYTE_COLS_PER_GROUP;
    self->loop1_D_mem_addr += !self->HAS_BIAS ? 0 : 
                              self->O_BYTE_COLS_PER_GROUP;
  }

  DBG_OG("next_output_group ");
  return true;
}

//============================================================================
// Tiling Loop #2 subroutines
//============================================================================

static void reset_A_tile_subcol(gemmini_t *self) {
  // this scope modifies: self->gbl_B_cur_sp_row_addr;
  //                      self->gbl_B_alt_sp_row_addr;
  //                      self->gbl_CD_acc_row_addr
  
  self->loop2_k_tile_col    = 0;
  self->gbl_tile_row        = self->loop1_tile_row_start;
  self->gbl_tile_col        = self->loop1_tile_col_start;
  self->gbl_CD_acc_row_addr = 0;

  self->loop2_A_mem_addr = self->loop1_A_mem_addr;
  self->loop2_B_mem_addr = self->loop1_B_mem_addr;
  self->loop2_C_mem_addr = self->loop1_C_mem_addr;
  self->loop2_D_mem_addr = self->loop1_D_mem_addr;

  DBG_LOOP("  reset_A_tile_subcol              ");
}

static bool next_A_tile_subcol(gemmini_t *self) {
  if(self->loop2_k_tile_col == self->K_TILE_COL_END) {
    // we just accumulated the last A-column into the output-group. were done
    DBG_LOOP("<-next_A_tile_subcol               ");
    return false;
  }
  self->loop2_k_tile_col    += 1;
  self->gbl_tile_row         = self->loop1_tile_row_start;
  self->gbl_tile_col         = self->loop1_tile_col_start;
  self->gbl_CD_acc_row_addr  = 0;

  self->loop2_A_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop2_B_mem_addr += self->B_BYTES_PER_TILE_ROW;
  self->loop2_C_mem_addr += 0;
  self->loop2_D_mem_addr += 0;

  // swap current/alternate B-tile scratchpad addrs
  const size_t tmp_B_sp_row_addr = self->gbl_B_cur_sp_row_addr;
  self->gbl_B_cur_sp_row_addr    = self->gbl_B_alt_sp_row_addr;
  self->gbl_B_alt_sp_row_addr    = tmp_B_sp_row_addr;

  DBG_LOOP("  next_A_tile_subcol               ");
  return true;
}

static void move_first_B_tile_into_sp(gemmini_t *self) {
  // calculate mvin parameters
  const size_t B_mem_addr    = self->loop2_B_mem_addr;
  const size_t B_mem_stride  = self->B_BYTES_PER_ROW;
  const size_t B_sp_row_addr = self->gbl_B_cur_sp_row_addr;

  DBG_MVIN_B;
  // issue gemmini commands
  gemmini_config_ld(B_mem_stride);
  gemmini_mvin(B_mem_addr, B_sp_row_addr);
}

//============================================================================
// Tiling Loop #3 subroutines
//============================================================================

static void reset_B_tile_subcol_in_subrow(gemmini_t *self) {
  // this scope modifies: self->gbl_tile_col
  //                      self->gbl_tile_row
  //                      self->gbl_CD_acc_row_addr
  //                      self->gbl_B_cur_sp_row_addr
  //                      self->gbl_B_alt_sp_row_addr

  self->loop3_A_mem_addr = self->loop2_A_mem_addr;
  self->loop3_B_mem_addr = self->loop2_B_mem_addr;
  self->loop3_C_mem_addr = self->loop2_C_mem_addr;
  self->loop3_D_mem_addr = self->loop2_D_mem_addr;

  DBG_LOOP("    reset_B_tile_subcol_in_subrow  ");
}

static bool next_B_tile_subcol_in_subrow(gemmini_t *self) {
  if(self->gbl_tile_col == self->loop1_tile_col_end) {
    // we have already done the last column in the output-group, so were done
    DBG_LOOP("  <-next_B_tile_subcol_in_subrow   ");
    return false;
  }
  // modify global state
  self->gbl_tile_row             = self->loop1_tile_row_start;
  self->gbl_tile_col            += 1;
  self->gbl_CD_acc_row_addr     += self->BYTE_ROWS_PER_TILE;

  const size_t tmp_B_sp_row_addr = self->gbl_B_cur_sp_row_addr;
  self->gbl_B_cur_sp_row_addr    = self->gbl_B_alt_sp_row_addr;
  self->gbl_B_alt_sp_row_addr    = tmp_B_sp_row_addr;

  // modify loop3-local state
  self->loop3_A_mem_addr += 0;
  self->loop3_B_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop3_C_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop3_D_mem_addr += self->O_TILE_BYTE_WIDTH;

  DBG_LOOP("    next_B_tile_subcol_in_subrow   ");
  return true;
}

static void maybe_move_next_B_tile_into_sp(gemmini_t *self) {
  if(self->gbl_tile_col == self->loop1_tile_col_end) {
    // can't load next B-tile if we are already on the last one
    return;
  }

  // calculate mvin parameters
  const size_t B_mem_addr    = self->loop3_B_mem_addr + 
                               self->I_TILE_BYTE_WIDTH;
  const size_t B_mem_stride  = self->B_BYTES_PER_ROW;
  const size_t B_sp_row_addr = self->gbl_B_alt_sp_row_addr;

  DBG_MVIN_B;
  // issue gemmini commands
  gemmini_config_ld(B_mem_stride);
  gemmini_mvin(B_mem_addr, B_sp_row_addr);
}

//============================================================================
// Tiling Loop #4 subroutines
//============================================================================

static void reset_A_tile_subrow_in_subcol(gemmini_t *self) {
  // this scope modifies: self->gbl_tile_row
  //                      self->gbl_CD_acc_row_addr

  self->loop4_A_mem_addr    = self->loop3_A_mem_addr;
  self->loop4_B_mem_addr    = self->loop3_B_mem_addr;
  self->loop4_C_mem_addr    = self->loop3_C_mem_addr;
  self->loop4_D_mem_addr    = self->loop3_D_mem_addr;

  self->loop4_A_sp_row_addr = 0;

  DBG_LOOP("      reset_A_tile_subrow_in_subcol");
}

static bool next_A_tile_subrow_in_subcol(gemmini_t *self) {
  if(self->gbl_tile_row == self->loop1_tile_row_end) {
    // just finished the final row of tiles in the 4th loop, so were done
    DBG_LOOP("    <-next_A_tile_subrow_in_subcol ");
    return false;
  }
  // modify global state
  self->gbl_tile_row        += 1;
  self->gbl_CD_acc_row_addr += self->BYTE_ROWS_PER_TILE;

  // modify loop4-local state
  self->loop4_A_mem_addr    += self->A_BYTES_PER_TILE_ROW;
  self->loop4_C_mem_addr    += self->C_BYTES_PER_TILE_ROW;
  self->loop4_D_mem_addr    += (self->HAS_BIAS && !self->REPEATING_BIAS) ?
                               self->D_BYTES_PER_TILE_ROW : 0;

  self->loop4_A_sp_row_addr += self->BYTE_ROWS_PER_TILE;

  DBG_LOOP("      next_A_tile_subrow_in_subcol ");
  return true;
}

static void maybe_move_A_tile_into_sp(gemmini_t *self) {
  if(self->gbl_tile_col != self->loop1_tile_col_start) {
    // only move A-tiles in during first column of tiles in the output-group
    return;
  }

  // calculate mvin parameters
  const size_t A_mem_addr    = self->loop4_A_mem_addr;
  const size_t A_mem_stride  = self->A_BYTES_PER_ROW;
  const size_t A_sp_row_addr = self->loop4_A_sp_row_addr;

  DBG_MVIN_A;
  // issue gemmini commands
  gemmini_config_ld(A_mem_stride);
  gemmini_mvin(A_mem_addr, A_sp_row_addr);
}

static void maybe_move_D_tile_into_acc(gemmini_t *self) {
  if(!((self->loop2_k_tile_col == 0) && self->HAS_BIAS)) {
    // only move D-tiles in during first partial-sum in an output-group
    return;
  }

  // calculate mvin parameters (NOTE: we know D is valid at this point)
  const size_t D_mem_addr     = self->loop4_D_mem_addr;
  const size_t D_mem_stride   = self->REPEATING_BIAS ? 0 : 
                                self->D_BYTES_PER_ROW;
  const size_t D_acc_row_addr = ACC_ADDR_NEW(self->gbl_CD_acc_row_addr);

  DBG_MVIN_D;
  // issue gemmini commands
  gemmini_config_ld(D_mem_stride);
  gemmini_mvin(D_mem_addr, D_acc_row_addr);
}

static void preload_B_tile_into_array_and_set_C_addr_in_acc(gemmini_t *self) {
  // on first tile in 4th loop: preload this B-tile
  // else:                      preload garbage B-tile (no scratchpad load)
  const size_t B_sp_row_addr = (self->gbl_tile_row == 
                                self->loop1_tile_row_start) ?
                               self->gbl_B_cur_sp_row_addr :
                               GARBAGE_ADDR;

  // if has D-bias already loaded: accumulate c in accumulator
  // elif first k-col in 2nd loop: overwrite c in accumulator
  // else:                         accumulate c in accumulator
  const size_t C_acc_row_addr = self->HAS_BIAS || (self->loop2_k_tile_col > 0) ?
                                ACC_ADDR_ACC(self->gbl_CD_acc_row_addr) :
                                ACC_ADDR_NEW(self->gbl_CD_acc_row_addr);
  DBG_PRELOAD_B;
  // execute preload command
  gemmini_preload(B_sp_row_addr, C_acc_row_addr);
}

static void do_matmul(const gemmini_t *self) {
  // calculate compute parameters
  const size_t A_sp_row_addr = self->loop4_A_sp_row_addr;

  // on first tile in 4th loop: compute_preloaded, else: compute_accumulated
  if(self->gbl_tile_row == self->loop1_tile_row_start) {
    DBG_COMPUTE_PRE;
    gemmini_compute_preloaded(A_sp_row_addr, GARBAGE_ADDR);
  } else {
    DBG_COMPUTE_ACC;
    gemmini_compute_accumulated(A_sp_row_addr, GARBAGE_ADDR);
  }
}

static void maybe_move_C_tile_into_mem(gemmini_t *self) {
  if(self->loop2_k_tile_col != self->K_TILE_COL_END) {
    // only move C-tiles out during last partial-sum in an output-group
    return;
  }

  // calculate mvout parameters
  const size_t C_mem_addr     = self->loop4_C_mem_addr;
  const size_t C_mem_stride   = self->C_BYTES_PER_ROW;
  const size_t C_acc_row_addr = ACC_ADDR_RD(self->gbl_CD_acc_row_addr);

  DBG_MVOUT_C;
  // issue gemmini commands
  //gemmini_config_st(C_mem_stride);
  gemmini_mvout(C_mem_addr, C_acc_row_addr);
}

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

  // sanitize inputs before starting
  if(is_valid_to_continue(dim_I, dim_J, dim_K, A, B, D, C, 
                          act, shift, repeating_bias, tiled_matmul_type)) {
    // create the state object
    gemmini_t *self = create_gemmini(dim_I, dim_J, dim_K, A, B, D, C, 
                                     act, shift, repeating_bias);
    // actually do the tiled matmul
    reset_output_group(self);
    do {
      reset_A_tile_subcol(self);
      do {
        move_first_B_tile_into_sp(self);
        reset_B_tile_subcol_in_subrow(self);
        do {
          maybe_move_next_B_tile_into_sp(self);
          reset_A_tile_subrow_in_subcol(self);
          do {
            maybe_move_A_tile_into_sp(self);
            maybe_move_D_tile_into_acc(self);
            preload_B_tile_into_array_and_set_C_addr_in_acc(self);
            do_matmul(self);
            maybe_move_C_tile_into_mem(self);

          } while(next_A_tile_subrow_in_subcol(self));
        } while(next_B_tile_subcol_in_subrow(self));
      } while(next_A_tile_subcol(self));
      // TODO: bug. using this to synchronize
      FIX-ME THIS DOESN"T DO AANYTHING USEFUL
      gemmini_config_ex(self->DATAFLOW, self->ACTIVATION, 
                        self->SYSTOLIC_OUT_RSHIFT, self->ACC_OUT_RSHIFT, 
                        self->RELU6_IN_LSHIFT);
    } while(next_output_group(self));

    // cleanup the state object
    destroy_gemmini(self);
  }
  gemmini_fence();
}

#endif // __GEMMINI_TILER_H__

