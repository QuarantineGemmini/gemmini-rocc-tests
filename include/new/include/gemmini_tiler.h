//===========================================================================
// WARNING!!!!
// this header contains implementation code. do not use in library code.
//===========================================================================

#ifndef __GEMMINI_TILER_H__
#define __GEMMINI_TILER_H__

#include <stdint.h>
#include <string.h>

#include "include/gemmini_params.h"
#include "include/gemmini.h"

//===========================================================================
// accumulator addressing helpers
//===========================================================================
#define ACC_ADDR_RD(addr)  ((2 << (ADDR_LEN - 2)) & (addr))
#define ACC_ADDR_NEW(addr) ((2 << (ADDR_LEN - 2)) & (addr))
#define ACC_ADDR_ACC(addr) ((3 << (ADDR_LEN - 2)) & (addr))

//===========================================================================
// state objects
//===========================================================================
typedef uint16_t tile_t;
typedef uint16_t sp_addr_t;
typedef uintptr_t mem_addr_t;

typedef struct gemmini {
  // global constants
  sp_row_t    GBL_B_SP_ROW_ADDR_1;
  sp_row_t    GBL_B_SP_ROW_ADDR_2;

  tile_t      TILE_COLS_PER_GROUP;
  tile_t      TILE_COL_END;
  tile_t      TILE_ROWS_PER_GROUP;
  tile_t      TILE_ROW_END;
  tile_t      K_TILE_COL_END;

  mem_addr_t  A_MEM_ADDR;            // mem-addr of A-matrix
  mem_addr_t  B_MEM_ADDR;
  mem_addr_t  C_MEM_ADDR;
  mem_addr_t  D_MEM_ADDR;
  size_t      A_BYTES_PER_TILE_ROW;  // bytes in A-matrix row * rows-per-tile
  size_t      B_BYTES_PER_TILE_ROW;
  size_t      C_BYTES_PER_TILE_ROW;
  size_t      D_BYTES_PER_TILE_ROW;
  size_t      A_BYTES_PER_ROW;       // bytes in A-matrix row
  size_t      B_BYTES_PER_ROW;
  size_t      C_BYTES_PER_ROW;
  size_t      D_BYTES_PER_ROW;

  bool        HAS_BIAS;              // if computing A*B+D=C, not A*B=C
  bool        REPEATING_BIAS;        // if HAS_BIAS, repeat 1st row only

  size_t      I_BYTE_COLS_PER_GROUP; // byte-width of output-group A,B,C
  size_t      O_BYTE_COLS_PER_GROUP; // byte-width of output-group D
  size_t      I_TILE_BYTE_WIDTH;     // byte-width of tile A,B,C
  size_t      O_TILE_BYTE_WIDTH;     // byte-width of tile D
  size_t      BYTE_ROWS_PER_TILE;    // num-rows of tile A,B,C,D

  // global state across all loops
  tile_t      gbl_tile_row;
  tile_t      gbl_tile_row;
  sp_row_t    gbl_B_cur_sp_row_addr;
  sp_row_t    gbl_B_alt_sp_row_addr;

  // global state that is reset for each output-group
  sp_row_t    gbl_C_acc_row_addr;
  sp_row_t    gbl_D_acc_row_addr;

  // loop1 local state
  tile_t      loop1_tile_col_start;
  tile_t      loop1_tile_col_end;
  tile_t      loop1_tile_row_start;
  tile_t      loop1_tile_row_end;
  mem_addr_t  loop1_A_mem_addr;
  mem_addr_t  loop1_B_mem_addr;
  mem_addr_t  loop1_C_mem_addr;
  mem_addr_t  loop1_D_mem_addr;

  // loop2 local state
  tile_t      loop2_k_tile_col;
  mem_addr_t  kloop2_A_mem_addr;
  mem_addr_t  kloop2_B_mem_addr;
  mem_addr_t  kloop2_C_mem_addr;
  mem_addr_t  kloop2_D_mem_addr;

  // loop3 local state
  mem_addr_t  loop3_A_mem_addr;
  mem_addr_t  loop3_B_mem_addr;
  mem_addr_t  loop3_C_mem_addr;
  mem_addr_t  loop3_D_mem_addr;

  // loop4 local state
  sp_row_t    loop4_A_sp_row_addr;
  mem_addr_t  loop4_A_mem_addr;
  mem_addr_t  loop4_B_mem_addr;
  mem_addr_t  loop4_C_mem_addr;
  mem_addr_t  loop4_D_mem_addr;

} gemmini_t;

//============================================================================
// create_gemmini subroutines
//============================================================================
gemmini_t* 
init_gemmini_state(size_t M, size_t N, size_t K, 
                   mem_addr_t A, mem_addr_t B, mem_addr_t C, mem_addr_t D,
                   bool bias, bool repeating_bias) {
  // create the state struct
  gemmini_t *self = (gemmini_t *) malloc(sizeof(gemmini_t));
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
    //
    self->TILE_ROWS_PER_GROUP = 4;
    self->TILE_ROWS_PER_GROUP = (ACC_ROWS / DIM) / self->TILE_ROWS_PER_GROUP;
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
gemmini_t* 
create_gemmini(size_t M, size_t N, size_t K,
               const elem_t A[M][K], const elem_t B[K][N],
               const acc_t * D, elem_t C[M][N],
               int act, int shift, bool repeating_bias) {

  // define constants
  const size_t DATAFLOW                  = WEIGHT_STATIONARY;
  const size_t ACTIVATION                = act;
  const size_t SYSTOLIC_OUTPUT_RSHIFT    = 0;
  const size_t ACCUMULATOR_OUTPUT_RSHIFT = shift;
  const size_t RELU6_INPUT_LSHIFT        = 0;

  // initialize state
  struct gemmini_t * self = init_gemmini_state(M, N, K,
                                              (mem_addr_t) A, (mem_addr_t) B,
                                              (mem_addr_t) C, (mem_addr_t) D,
                                              (D != NULL), repeating_bias);

  // issue gemini commands
  gemmini_config_ex(DATAFLOW,
                    ACTIVATION, 
                    SYSTOLIC_OUTPUT_RSHIFT, 
                    ACCUMULATOR_OUTPUT_RSHIFT,
                    RELU6_INPUT_LSHIFT);

  // state and hardware are now initialized
  return self;
}

void destroy_gemmini(gemmini_t *self) {
  free(self);
}

//============================================================================
// Tiling Loop #1 subroutines
//============================================================================
void reset_output_group(gemmini_t *self) {
  // define mutable global state mutable by all loops, persist across all ogs
  self->gbl_tile_row          = 0;
  self->gbl_tile_col          = 0;
  self->gbl_B_cur_sp_row_addr = self->GBL_B_SP_ROW_ADDR_1;
  self->gbl_B_alt_sp_row_addr = self->GBL_B_SP_ROW_ADDR_2;

  // define mutable global state mutable by all loops, reset after each og
  self->gbl_C_acc_row_addr = 0;
  self->gbl_D_acc_row_addr = 0;

  // update the start/end tiles for this output-group (inclusive)
  // NOTE: duplicated with next_output_group!!
  self->loop1_tile_col_start = self->gbl_tile_col;
  self->loop1_tile_col_end   = min(self->gbl_tile_col + 
                                   self->TILE_COLS_PER_GROUP - 1,
                                   self->TILE_COL_END);
  self->loop1_tile_row_start = self->gbl_tile_row;
  self->loop1_tile_row_end   = min(self->gbl_tile_row + 
                                   self->TILE_ROWS_PER_GROUP - 1,
                                   self->TILE_ROW_END);

  // derived pointers to matrices in memory for this og
  self->loop1_A_mem_addr = self->A_MEM_ADDR;
  self->loop1_B_mem_addr = self->B_MEM_ADDR;
  self->loop1_C_mem_addr = self->C_MEM_ADDR;
  self->loop1_D_mem_addr = self->D_MEM_ADDR;
}

bool next_output_group(gemmini_t *self) {
  bool did_row_incr = false;
  bool did_col_incr = false;

  if(self->gbl_tile_col == self->TILE_COL_END) {
    if(self->gbl_tile_row == self->TILE_ROW_END) {
      // we finished the last output group. so we're done
      return;
    } else {
      self->gbl_tile_col = 0;
      self->gbl_tile_row += 1;
      did_row_incr = true;
    }
  } else {
    self->gbl_tile_col += 1;
    self->gbl_tile_row -= self->TILE_ROWS_PER_GROUP;
    did_col_incr = true;
  }

  // reset global state that resets for each new output-group
  self->gbl_C_acc_row_addr = 0;
  self->gbl_D_acc_row_addr = 0;

  // update the start/end tiles for this output-group (inclusive)
  self->loop1_tile_col_start = self->gbl_tile_col;
  self->loop1_tile_col_end   = min(self->gbl_tile_col + 
                                   self->TILE_COLS_PER_GROUP - 1,
                                   self->TILE_COL_END);
  self->loop1_tile_row_start = self->gbl_tile_row;
  self->loop1_tile_row_end   = min(self->gbl_tile_row + 
                                   self->TILE_ROWS_PER_GROUP - 1,
                                   self->TILE_ROW_END);

  // update all derived pointers to matrices in memory
  if(did_row_incr) {
    self->loop1_A_mem_addr  = self->A_MEM_ADDR + (self->loop1_tile_row_start *
                              self->A_BYTES_PER_TILE_ROW);
    self->loop1_B_mem_addr  = self->B_MEM_ADDR;
    self->loop1_C_mem_addr  = self->C_MEM_ADDR + (self->loop1_tile_row_start *
                              self->C_BYTES_PER_TILE_ROW);
    self->loop1_D_mem_addr  = !self->HAS_BIAS ? NULL :
                              (self->HAS_REPEATING_BIAS ? self->D_MEM_ADDR :
                               self->D_MEM_ADDR + (self->loop1_tile_row_start *
                               self->D_BYTES_PER_TILE_ROW);
  } else if(did_col_incr) {
    self->loop1_A_mem_addr += 0;
    self->loop1_B_mem_addr += self->I_BYTE_COLS_PER_GROUP;
    self->loop1_C_mem_addr += self->I_BYTE_COLS_PER_GROUP;
    self->loop1_D_mem_addr += !self->HAS_BIAS ? 0 : 
                              self->O_BYTE_COLS_PER_GROUP;
  }

  return true;
}

//============================================================================
// Tiling Loop #2 subroutines
//============================================================================

void reset_A_tile_subcol(gemmini_t *self) {
  // this scope modifies: self->gbl_B_cur_sp_row_addr;
  //                      self->gbl_B_alt_sp_row_addr;
  
  self->loop2_k_tile_col = 0;

  self->loop2_A_mem_addr = self->loop1_A_mem_addr;
  self->loop2_B_mem_addr = self->loop1_B_mem_addr;
  self->loop2_C_mem_addr = self->loop1_C_mem_addr;
  self->loop2_D_mem_addr = self->loop1_D_mem_addr;
}

bool next_A_tile_subcol(gemmini_t *self) {
  if(self->loop2_k_tile_col == self->K_TILE_COL_END) {
    // we just accumulated the last A-column into the output-group. were done
    return;
  }
  self->loop2_k_tile_col += 1;

  self->loop2_A_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop2_B_mem_addr += self->B_BYTES_PER_TILE_ROW;
  self->loop2_C_mem_addr += 0;
  self->loop2_D_mem_addr += 0;

  // swap current/alternate B-tile scratchpad addrs
  const size_t tmp_B_sp_row_addr = self->gbl_B_cur_sp_row_addr;
  self->gbl_B_cur_sp_row_addr    = self->gbl_B_alt_sp_row_addr;
  self->gbl_B_alt_sp_row_addr    = tmp_B_sp_row_addr;

  return true;
}

void move_first_B_tile_into_sp(gemmini_t *self) {
  // calculate mvin parameters
  const size_t B_mem_addr    = self->loop2_B_mem_addr;
  const size_t B_mem_stride  = self->B_BYTES_PER_ROW;
  const size_t B_sp_row_addr = self->gbl_B_cur_sp_row_addr;

  // issue gemmini commands
  gemmini_config_ld(B_mem_stride);
  gemmini_mvin(B_mem_addr, B_sp_row_addr);
}

//============================================================================
// Tiling Loop #3 subroutines
//============================================================================

void reset_B_tile_subcol_in_subrow(gemmini_t *self) {
  // this scope modifies: self->gbl_tile_col
  //                      self->gbl_B_cur_sp_row_addr;
  //                      self->gbl_B_alt_sp_row_addr;

  self->loop3_A_mem_addr = self->loop2_A_mem_addr;
  self->loop3_B_mem_addr = self->loop2_B_mem_addr;
  self->loop3_C_mem_addr = self->loop2_C_mem_addr;
  self->loop3_D_mem_addr = self->loop2_D_mem_addr;
}

bool next_B_tile_subcol_in_subrow(gemmini_t *self) {
  if(self->gbl_tile_col == self->loop1_tile_col_end) {
    // we have already done the last column in the output-group, so were done
    return false;
  }
  self->gbl_tile_col += 1;

  self->loop3_A_mem_addr += 0;
  self->loop3_B_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop3_C_mem_addr += self->I_TILE_BYTE_WIDTH;
  self->loop3_D_mem_addr += self->O_TILE_BYTE_WIDTH;

  // swap current/alternate B-tile scratchpad addrs
  const size_t tmp_B_sp_row_addr = self->gbl_B_cur_sp_row_addr;
  self->gbl_B_cur_sp_row_addr    = self->gbl_B_alt_sp_row_addr;
  self->gbl_B_alt_sp_row_addr    = tmp_B_sp_row_addr;

  return true;
}

void maybe_move_next_B_tile_into_sp(gemmini_t *self) {
  if(self->gbl_tile_col != self->loop1_tile_col_end) {
    // can't load next B-tile if we are already on the last one
    return;
  }

  // calculate mvin parameters
  const size_t B_mem_addr    = self->loop3_B_mem_addr + 
                               self->I_TILE_BYTE_WIDTH;
  const size_t B_mem_stride  = self->A_BYTES_PER_ROW;
  const size_t B_sp_row_addr = self->gbl_B_alt_sp_row_addr;

  // issue gemmini commands
  gemmini_config_ld(B_mem_stride);
  gemmini_mvin(B_mem_addr, B_sp_row_addr);
}

//============================================================================
// Tiling Loop #4 subroutines
//============================================================================

void reset_A_tile_subrow_in_subcol(gemmini_t *self) {
  // this scope modifies: self->gbl_tile_row
  //                      self->gbl_C_acc_row_addr
  //                      self->gbl_D_acc_row_addr

  self->loop4_A_mem_addr    = self->loop3_A_mem_addr;
  self->loop4_B_mem_addr    = self->loop3_B_mem_addr;
  self->loop4_C_mem_addr    = self->loop3_C_mem_addr;
  self->loop4_D_mem_addr    = self->loop3_D_mem_addr;

  self->loop4_A_sp_row_addr = 0;
}

bool next_A_tile_subrow_in_subcol(gemmini_t *self) {
  if(self->gbl_tile_row == self->loop1_tile_row_end) {
    // just finished the final row of tiles in the 4th loop, so were done
    return false;
  }
  self->gbl_tile_row += 1;

  self->loop4_A_mem_addr    += self->A_BYTES_PER_TILE_ROW;
  self->loop4_C_mem_addr    += self->C_BYTES_PER_TILE_ROW;
  self->loop4_D_mem_addr    += (self->HAS_BIAS && !self->REPEATING_BIAS) ?
                               self->D_BYTES_PER_TILE_ROW : 0;

  self->loop4_A_sp_row_addr += self->BYTE_ROWS_PER_TILE;
  self->gbl_C_acc_row_addr  += self->BYTE_ROWS_PER_TILE;
  self->gbl_D_acc_row_addr  += self->BYTE_ROWS_PER_TILE;
}

void maybe_move_A_tile_into_sp(gemmini_t *self) {
  if(self->gbl_tile_col != self->loop1_tile_col_start) {
    // only move A-tiles in during first column of tiles in the output-group
    return;
  }

  // calculate mvin parameters
  const size_t A_mem_addr    = self->loop4_A_mem_addr;
  const size_t A_mem_stride  = self->A_BYTES_PER_ROW;
  const size_t A_sp_row_addr = self->loop4_A_sp_row_addr

  // issue gemmini commands
  gemmini_config_ld(A_mem_stride);
  gemmini_mvin(A_mem_addr, A_sp_row_addr);
}

void maybe_move_D_tile_into_acc(gemmini_t *self) {
  if(!(self->loop2_k_tile_col == 0 && self->HAS_BIAS)) {
    // only move D-tiles in during first partial-sum in an output-group
    return;
  }

  // calculate mvin parameters (NOTE: we know D is valid at this point)
  const size_t D_mem_addr     = self->loop4_D_mem_addr;
  const size_t D_mem_stride   = self->REPEATING_BIAS ? 0 : 
                                self->D_BYTES_PER_ROW;
  const size_t D_acc_row_addr = ACC_ADDR_NEW(self->gbl_D_acc_row_addr);

  // issue gemmini commands
  gemmini_config_ld(D_mem_stride);
  gemmini_mvin(D_mem_addr, D_acc_row_addr);
}

void preload_B_tile_into_array_and_set_C_addr_in_acc(gemmini_t *self) {
  // on first tile in 4th loop: preload this B-tile
  // else:                      preload garbage B-tile (no scratchpad load)
  const size_t B_sp_row_addr = (self->gbl_tile_row == 
                                self->loop1_tile_row_start) ?
                               self->gbl_B_cur_sp_row_addr :
                               GARBAGE_ADDR;

  // if has D-bias already loaded: accumulate c in accumulator
  // elif first k-col in 2nd loop: overwrite c in accumulator
  // else:                         accumulate c in accumulator
  const size_t C_acc_row_addr = self->HAS_BIAS || self->loop2_k_tile_col > 0 ?
                                ACC_ADDR_ACC(self->gbl_C_acc_row_addr) :
                                ACC_ADDR_NEW(self->gbl_C_acc_row_addr);

  // execute preload command
  gemmini_preload(B_sp_row_addr, C_acc_row_addr);
}

void do_matmul(gemmini_t *self) {
  // calculate compute parameters
  const size_t A_sp_row_addr = self->loop4_A_sp_row_addr;

  // on first tile in 4th loop: compute_preloaded, else: compute_accumulated
  if(self->gbl_tile_row == self->loop1_tile_row_start) {
    gemmini_compute_preloaded(A_sp_row_addr, GARBAGE_ADDR);
  } else {
    gemmini_compute_accumulated(A_sp_row_addr, GARBAGE_ADDR);
  }
}

void maybe_move_C_tile_into_mem(gemmini_t *self) {
  if(self->loop2_k_tile_col != self->K_TILE_COL_END) {
    // only move C-tiles out during last partial-sum in an output-group
    return;
  }

  // calculate mvout parameters
  const size_t C_mem_addr     = self->loop4_C_mem_addr;
  const size_t C_mem_stride   = self->C_BYTES_PER_ROW;
  const size_t C_acc_row_addr = ACC_ADDR_READ(self->gbl_C_acc_row_addr);

  // issue gemmini commands
  gemmini_config_st(C_mem_stride);
  gemmini_mvout(C_mem_addr, C_acc_row_addr);
}

//============================================================================
// Input Validation
//============================================================================
bool is_valid_to_continue(size_t M, size_t N, size_t K, 
                          const elem_t A[M][K], 
                          const elem_t B[K][N],
                          const acc_t * D, elem_t C[M][N],
                          int act, int shift, bool repeating_bias,
                          enum tiled_matmul_type_t tiled_matmul_type) 

  // validate inputs
  assert(M % DIM == 0 && M > 0);
  assert(N % DIM == 0 && N > 0);
  assert(K % DIM == 0 && K > 0);

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
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                       const elem_t A[dim_I][dim_K], 
                       const elem_t B[dim_K][dim_J],
                       const acc_t * D, elem_t C[dim_I][dim_J],
                       int act, int shift, bool repeating_bias,
                       enum tiled_matmul_type_t tiled_matmul_type) 
{
  // sanitize inputs before starting
  if(should_continue(dim_I, dim_J, dim_K, A, B, D, C, 
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
    } while(next_output_group(self));

    // cleanup the state object
    destroy_gemmini(self);
  }
}

#endif // __GEMMINI_TILER_H__

// main todos:
// TODO: add a bunch of asserts/debug/printf everywhere
// TODO: enable error and warning messages 
// TODO: make 2 new instructions to preload B and set C separately
// TODO: what to do about input matrices that are not at tile-granularity?
//       we need to add support for this in hardware by automatically adding
//       zeros
// TODO: fix the messed up ratio of spad and accumulator rows. there should
//       be way more acc rows than sp rows in this scheme im using, for max
//       utilization and data-reuse
//
// DONE: make instruction to load D-tile straight into an accumulator bank,
//       while expanding the bit-width automatically if needed
//       SOLUTION: you can load D straight into accumulator already. the
//       input D matrix has element-sizes of acc_t!!
// DONE: when accumulator writes out, write out in size of elem_t, not acc_t
//       SOLUTION: the AccumulatorMem always reads-out with inputType elements.
//       in the "activation/shift pipeline", it clips the shifted value to 
//       the input size. This is why the C_matrix is of elem_t, and not acc_t
//
// other TODO's
// TODO: should we enable bias D-matrices with element size of elem_t instead
//       of acc_t? the bias matrix would be smaller, but then we wouldn't
//       be able to load it into the accumulator
// TODO: when accumulator writes out, write out in size of elem_t, not acc_t
// TODO: in our gemmini-hardware-tiler,
//       create a config insn to set C addr in acc without also preloading 
//       B into array. this causes uneccesary data-movement through array
//       when we just want to set a new C-addr for a new insn
// TODO: investigate: if I call preload 10 times in a row, will they just
//       overwrite the same flops in the systolic-array? so when i call
//       matmul.compute.preloaded, it loads the last one i preload()ed?
//       This would be IDEAL. I think the answer is yes.
// TODO: right now, mesh MUST be square (block_size) since ctrl signals are
//       hardcoded to think that way. this is ok with me

// - only use add, sub, compare operations. need to implement in hardware
//   as next-state logic!

// wieght-stationary semantics:
//   perform_single_preload:
//     - only loads B from scratchpad, accumulator, or all zeros
//     - does NOT perform a matmul as well
//   overlap matmul and next preload
//     - you MUST issue the matmul first and the preload 2nd
//     - only loads B from scratchpad, accumulator, or all zeros
//     - make sure you use matmul.compute.accumulated command, which won't flip
//       the B if used.
//   matmul only on same B
//     - make sure you use matmul.compute.accumulated command, which won't flip
//   matmul only on other preloaded B
//     - make sure you use matmul.compute.preloaded command


// setting B == GARBAGE_ADDR: preloads zeros (i won't use this ever)

// output tiles in accumulator:
// - on first iteration:
//   set D == GARBAGE_ADDR: sets the initial D to all zeros
//   set C == destination addr in accumulator (high 2 bits are 10
//   
// - on successive iterations
//   set C == destination addr in accumulator (high 2 bits are 11)
//   set D == C
 
// on last output_group iteration:
// - mvout the accumulator to 


//foreach output group from L->R, T->B
//  foreach K-col in input matrix for this output group
//    if first K-col in input group, zero the accumulator, else keep it
//    load first W tile into sp
//    for each weight tile in row
//      preload W tile into array from sp
//      load next W tile into sp
//      for each vertical tile in K-col image group
//        if first col in weight-row
//          load K-col-tile into sp
//        matmul.accum I*A -> accum
//        if last K-col in input group
//          mvout accum to memory

// inside ex_ctrl:
// ---------------
// - flush, compute, cmd_wait states:
//   - only does a flush when matmul_is_in_progress and a new preload or 
//     new matmul have not been issued. Flush means a ex_config is pending
//   - ctrl will not send new preload/matmul cmds to mesh until the all rows 
//     previous rows have entered the shifters. 
// - propagate notes
//   - in_prop_flush is ALWAYS 0 for WS flows
//   - in_prop is set to 1 ONLY if preload() is called, regardless of B/C addrs
// - mesh_ctrl_signals for propagate in WS mode:
//   1) 0 if flushing (no command being sent to mesh)
//   2) 0 if issuing a single_preload with no overlapping matmul
//   3) 0 if issuing both a preload and a matmul-accumulate
//   4) 0 if issuing a single matmul-accumulate
//   5) 1 if issuing both a preload and a matmul-preloaded
//   5) 1 if issuing a single matmul-preloaded
// - C_addr handling
//   - you MUST issue a preload and then a matmul.{prop,accum} immediatly
//   - when you do this, they both get scheduled to the mesh together
//   - C_addr does not persist if you don't issue both right away
// - HOW TO SCHEDULE B-prop AND C-accum addrs
//   - on first tile in 4th loop: 
//       1) preload this B-tile and set this C_addr
//       2) matmul.preloaded
//   - on subsequent tiles in 4th loop:
//       1) preload garbage B-tile (avoid scratchpad load) and set this C_addr
//       2) matmul.accumulated 
//          - THIS DOES NOT MEAN C IS ACCUMULATED/OVERWRITTEN -- ORTHOGONAL!!!
//          - THIS MEANS USE NEW C VALUE BUT KEEP THE B VALUE IN ARRAY
//
// inside mesh_with_delays:
// ---------------
// - in_prop is 0 or 1
// - in_prop will toggle when the mesh_ctrl_signals.propagate is 1 AND
//   it just finished loading the previous matmul inputs into shifters+mesh

  //==========================================================================
  // - the 2-D shape of tiles in the accumulator. This shape depends on the 
  //   relative size of the scratchpad, accumulator, and systolic array. 
  // - where the formula comes from:
  //   - a single column of 'acc_group_height' tiles from the input matrix are 
  //     loaded into scratchpad one by one
  //   - the '-1' is subtracted from 'acc_group_height' because I always save
  //     1 slot in the last bank for preloading the next weights from DRAM.
  //   - the input tiles in the scratchpad are reused 'acc_group_width' times
  //   - each weight tile is pre-loaded into sp before it is needed, and
  //     when it is needed, it is used 'acc_group_height' times in a row w/o 
  //     leaving the systolic array. Then it is discarded.
  // -------------------------------------------------------------------------
  // - I believe this formula optimally uses the scratchpad and accumulator
  //   for data-reuse regardless of their relative sizes.
  //==========================================================================

