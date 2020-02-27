#ifndef __GEMMINI_TILER_H__
#define __GEMMINI_TILER_H__

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

//============================================================================
// WARNING!!!!
// -----------
// - this header contains implementation code. do not use in library code.
// - i use static inline a lot, which typically doesn't make sense in headers
//============================================================================

typedef struct gemmini {
  size_t state...
} gemmini_t;

//============================================================================
gemmini_t* create_gemmini(size_t M, size_t N, size_t K,
                          const elem_t A[M][K], 
                          const elem_t B[K][N],
                          const acc_t * D, elem_t C[M][N],
                          int act, int shift, bool repeating_bias)
{
  struct Gemmini* g = (gemmini_t *) malloc(sizeof(gemmini_t));

  // C is accumulated, which is why it starts with 0b11
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

  // TODO: deal with non-repeated, and repeated_bias rows in D
  const acc_t * pre;
  if (k0 != 0) {
    pre = NULL;
  } else {
    size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
    pre = &((acc_t (*)[dim_J])D)[bias_row][j0*tile_J*DIM];
  }


  const size_t tile_rows    = DIM;
  const size_t tile_cols    = DIM;
  const size_t tile_elems   = tile_rows * tile_cols;
  const size_t tile_width_b = tile_cols * sizeof(elem_t);
  const size_t tile_size_b  = tile_rows * tile_width_b;

  // each accum and scratchpad-bank row holds 1 tile
  const size_t sp_banks           = BANK_NUM;
  const size_t tiles_per_sp_bank  = BANK_ROWS / tile_rows;
  const size_t tiles_per_acc      = ACC_ROWS / tile_rows;

  // configure the systolic array and accumulator
  self->DATAFLOW                  = WEIGHT_STATIONARY;
  self->ACTIVATION                = act;
  self->SYSTOLIC_OUTPUT_RSHIFT    = 0;
  self->ACCUMULATOR_OUTPUT_RSHIFT = shift;
  self->RELU6_INPUT_LSHIFT        = 0;

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
  g->acc_group_height = sp_banks * tiles_per_sp_bank - 2;
  g->acc_group_width  = tiles_per_acc / acc_group_height;
  if (g->acc_group_width == 0) {
    // HACK HACK HACK: make sp/accumulator ratio more reasonable
    g->acc_group_height = 4;
    g->acc_group_width  = tiles_per_acc / acc_group_height;
  }

  ASSERT(acc_group_height > 0);
  ASSERT(acc_group_width > 0);

  const size_t acc_group_rows = (M + acc_group_height - 1) / acc_group_height;
  const size_t acc_group_cols = (N + acc_group_width - 1) / acc_group_width;

  g->OG_COUNT = ???;
  g->OG_HEIGHT_B = ???;
  g->OG_WIDTH_B = ???;
  g->OG_ROWS = (g->M + g->OG_HEIGHT_B - 1) / g->OG_HEIGHT_B;
  g->OG_COLS = (g->N + g->OG_WIDTH_B  - 1) / g->OG_WIDTH_B;

  g->OM_TILE_COLS = ; // ceiling of total output matrix tile columns
  g->OM_TILE_ROWS = ; // ceiling of total output matrix tile rows
}

#define ACC_ADDR_RD(addr)  ((0b10 << 30) & (addr))
#define ACC_ADDR_NEW(addr) ((0b10 << 30) & (addr))
#define ACC_ADDR_ACC(addr) ((0b11 << 30) & (addr))

//============================================================================
// Tiling Loop #2: Helpers
//============================================================================

void is_first_A_tile_subcol(gemmini_t *self) {
  // - returns true if we are currently at the first tile column in the 
  //   A-matrix, which is useful for resetting the partial sums of the
  //   current output group to zero inside gemmini's accumulators
  return self->og_cur_tile_k_index == 0;
}

static inline void reset_accumulators(gemmini_t *self) {
  self->og_use_accumulators = 0;
}

void use_accumulators(gemmini_t *self) {
  self->og_use_accumulators = 1;
}

void is_first_B_tile_subrow(gemmini_t *self) {
  return self->og_cur_tile_row_offset == 0;
}

void load_first_B_tile_into_sp(gemmini_t *self) {
  // calculate mvin parameters
  const size_t mem_start  = self->og_B_offset;
  const size_t mem_stride = self->B_BYTE_WIDTH;
  const size_t sp_start   = self->ROWS_PER_SP - (self->ROWS_PER_TILE * 
                            (1 + self->og_cur_B_tile_sp_index));

  // issue gemmini commands
  gemmini_config_ld(mem_stride);
  gemmini_mvin(mem_start, sp_start);

  // update state
  self->og_cur_B_tile_sp_start = sp_start;
}

//===========================================================================
// Tiling Loop #3: Helpers
//===========================================================================

void preload_B_tile_into_arry(gemmini_t *self) {
  // calculate preload parameters
  const size_t B_sp_start = self->og_cur_B_tile_sp_start;
  const size_t C_sp_start = self->og_use_accumulators ? 
                            ACC_ADDR_ACC(self->og_cur_tile_index) :
                            ACC_ADDR_NEW(self->og_cur_tile_index);
  // issue gemmini commands
  gemmini_preload(B_sp_start, C_sp_start);
}

bool is_last_tile_subrow(gemmini_t *self) {
  // are we currently computing a tile in the last subrow of the output-group
  return self->og_cur_tile_row_offset == (self->og_end_tile_row - 1);
}

bool is_first_tile_subcol(gemmini_t *self) {
  // are we currently computing a tile in the first subcol of the output-group
  return self->og_cur_tile_col_offset == 0;
}

bool is_last_tile_subcol(gemmini_t *self) {
  // are we currently computing a tile in the last subcol of the output-group
  return self->og_cur_tile_col_offset == (self->og_end_tile_col - 1);
}

bool is_on_first_k_subcol(gemmini_t *self) {
  // are we currently on the first k-subcol in the input A-matrix
  return self->og_cur_k_subcol_index == 0;

bool is_on_last_k_subcol(gemmini_t *self) {
  // are we currently on the last k-subcol in the input A-matrix
  // return self->og_cur_tile_k_index == (self->A_TILE_WIDTH - 1);
  return self->cur_k_subcol_index == self->max_k_subcol_index;
}

void load_next_B_tile_into_sp(gemmini_t *self) {
  // calculate mvin parameters
  const size_t next_B_tile_col = (self->og_cur_tile_col_offset + 1) %
                                  self->og_tile_cols;
  const size_t next_B_tile_row = (next_B_tile_col == 0) ?
                                  (self->og_cur_tile_row_offset + 1) : 0;

  const size_t mem_start = self->og_B_offset + 
                           (next_B_tile_row * 
                            self->BYTE_ROWS_PER_TILE * self->B_BYTE_WIDTH) +
                           (next_B_tile_col * self->TILE_BYTE_WIDTH);
  const size_t mem_stride = self->B_BYTE_WIDTH;
  const size_t sp_start   = self->og_next_B_tile_sp_start;

  // issue gemmini commands
  gemmini_config_ld(mem_stride);
  gemmini_mvin(mem_start, sp_start);

  // swap pointers to cur/next B-tile in the scratchpad
  self->og_next_B_tile_sp_start = self->og_cur_B_tile_sp_start;
  self->og_cur_B_tile_sp_start = sp_start;
}

//============================================================================
// Tiling Loop #1: helpers
//============================================================================
void reset_output_group(gemmini_t *self) {
  // primary output-group counters
  self->cur_og_tile_row_start  = 0; 
  self->cur_og_tile_row_end    = self->TIL_ROWS_PER_GROUP - 1;
  self->cur_og_tile_col_start  = 0;
  self->cur_og_tile_col_end    = self->TIL_COLS_PER_GROUP - 1;
  self->cur_og_tile_index      = 0;
  self->cur_og_tile_row_offset = 0;
  self->cur_og_tile_col_offset = 0;
  self->cur_k_subcol           = 0;

  // maybe trim group-width if we are in last og in an og row
  if(self->cur_og_tile_col_end >= self->C_TILE_WIDTH) {
    self->cur_og_tile_col_end = self->C_TILE_WIDTH - 1;
    ASSERT(self->cur_og_tile_col_end >= 0, 
      "self->cur_og_tile_col_end == %d", self->cur_og_tile_col_end);
  }
  
  // maybe trim group-height if we are in the last og-row
  if(self->cur_og_tile_row_end >= self->C_TILE_HEIGHT) {
    self->cur_og_tile_row_end = self->C_TILE_HEIGHT - 1;
    ASSERT(self->cur_og_tile_row_end >= 0, 
      "self->cur_og_tile_row_end == %d", self->cur_og_tile_row_end);
  }

  //--------------------------------------------------------------------------
  // derived pointers to matrices in memory
  self->cur_A_og_mem_start = self->A_MEM_ADDR;
  self->cur_B_og_mem_start = self->B_MEM_ADDR;
  self->cur_C_og_mem_start = self->C_MEM_ADDR;
  self->cur_D_og_mem_start = self->D_MEM_ADDR;

  self->cur_accumulating   = 0;
  self->cur_B_tile_sp_addr = self->SP_ROWS - self->BYTE_ROWS_PER_TILE;
  self->nxt_B_tile_sp_addr = self->cur_B_tile_sp_addr - 
                             self->BYTE_ROWS_PER_TILE;
}

bool next_output_group_counters(gemmini_t *self) {
  // returns true if we successfully updated the output group 
 
  //--------------------------------------------------------------------------
  // calculate the tile (x,y) of the top-left and bottom right of the next og
  //--------------------------------------------------------------------------
  int32_t next_og_tile_row_start = self->cur_og_tile_row_start;
  int32_t next_og_tile_row_end   = self->cur_og_tile_row_end;
  int32_t next_og_tile_col_start = self->cur_og_tile_col_end + 1;
  int32_t next_og_tile_col_end   = self->cur_og_tile_col_end + 
                                   self->TILE_COLS_PER_GROUP;

  // wrap tile-group to the next row if needed
  if(next_og_tile_col_start >= self->C_TILE_WIDTH) {
    next_og_tile_row_start = self->cur_og_tile_row_end + 1;
    next_og_tile_row_end   = self->cur_og_tile_row_end
                             self->TILE_ROWS_PER_GROUP;
    next_og_tile_col_start = 0;
    next_og_tile_col_end   = self->TILE_COLS_PER_GROUP - 1;
  }

  // maybe trim group-width if we are in last og in an og row
  if(next_og_tile_col_end >= self->C_TILE_WIDTH) {
    next_og_tile_col_end = self->C_TILE_WIDTH - 1;
    ASSERT(next_og_tile_col_end >= 0, 
      "next_og_tile_col_end == %d", next_og_tile_col_end);
  }

  if(next_of_tile_row_start >= self->C_TILE_HEIGHT) {
    // we are done, finished all output-groups
    return false;
  }
  
  // maybe trim group-height if we are in the last og-row
  if(next_of_tile_row_end >= self->C_TILE_HEIGHT) {
    next_og_tile_row_end = self->C_TILE_HEIGHT - 1;
    ASSERT(next_og_tile_row_end >= 0, 
      "next_og_tile_row_end == %d", next_og_tile_row_end);
  }

  bool did_row_incr = next_og_tile_row_start != self->cur_og_tile_row_start;
  bool did_col_incr = next_og_tile_col_start != self->cur_og_tile_col_start;

  //--------------------------------------------------------------------------
  // update all derived pointers to matrices in memory
  //--------------------------------------------------------------------------
  if(did_row_incr) {
    self->cur_A_og_mem_start += self->A_INCR_OG_ROW_BYTES;
    self->cur_B_og_mem_start  = self->B_MEM_ADDR;
    self->cur_C_og_mem_start += self->C_INCR_OG_ROW_BYTES;
    self->cur_D_og_mem_start  = !self->HAS_BIAS ? NULL :
                                 self->HAS_REPEATING_BIAS ? self->D_MEM_ADDR :
                                   self->cur_D_og_mem_start + 
                                   self->D_INCR_OG_ROW_BYTES;
  } 
  else if(did_col_incr) {
    self->cur_A_og_mem_start += 0;
    self->cur_B_og_mem_start += self->OG_BYTE_WIDTH;
    self->cur_C_og_mem_start += self->OG_BYTE_WIDTH;
    self->cur_D_og_mem_start += self->OG_BYTE_WIDTH;
  }

  // scratchpad and accumulor configs
  self->cur_accumulating   = 0;
  self->cur_B_tile_sp_addr = self->cur_B_tile_sp_addr; // don't swap!
  self->nxt_B_tile_sp_addr = self->nxt_B_tile_sp_addr; // don't swap!

  //--------------------------------------------------------------------------
  // finish updating the output group
  //--------------------------------------------------------------------------
  self->cur_og_tile_row_start   = next_og_tile_row_start;
  self->cur_og_tile_row_end     = next_og_tile_row_end;
  self->cur_og_tile_col_start   = next_og_tile_col_start;
  self->cur_og_tile_col_end     = next_og_tile_col_end;
  self->cur_og_tile_index       = 0;
  self->cur_og_tile_row_offset  = 0;
  self->cur_og_tile_col_offset  = 0;
  self->cur_k_subcol            = 0;

  return true;
}

//============================================================================
// Tiling Loop #2: accumulate 1 output-group partial sum in the accumulators
//============================================================================
void reset_A_tile_subcol_counters(gemmini_t *self) {
  self->og_cur_k_subcol = 0;
}

void reset_A_tile_subcol_counters(gemmini_t *self) {
  self->og_cur_k_subcol = 0;
}

void reset_B_tile_subcol_in_subrow_counters(gemmini_t *self) {
  self->cur_og_col_row_offset = 0;
  self->cur_B_mem_start = self->cur_og_B_start_byte + 
                          self->cur_og_tile_col_offset * self->TILE_BYTE_WIDTH;
  self->cur_C_mem_start = self->cur_og_C_start_byte + 
                          self->cur_og_tile_col_offset * self->TILE_BYTE_WIDTH;
  self->cur_D_mem_start = self->cur_og_D_start_byte + 
                          self->cur_og_tile_col_offset * self->TILE_BYTE_WIDTH;
}

void reset_A_tile_subrow_in_subcol_counters(gemmini_t *self) {
  self->cur_og_tile_row_offset = 0;
  self->cur_A_mem_start = self->cur_og_A_start_byte + 
                          self->cur_og_tile_col_offset * self->TILE_BYTE_WIDTH;
}

void incr_A_tile_subrow_in_subcol_counters(gemmini_t *self) {
  self->cur_og_tile_row_offset = 0;


}

void load_D_tile_into_sp(gemmini_t *self) {
  // only load D into spad if we are using a bias
  if (self->USE_D_MATRIX) {
    const size_t D_tile_row = self->cur_og_tile_row_offset;
    const size_t D_tile_col = self->cur_og_tile_col_offset;
    const size_t mem_start = self->og_D_offset + 
                             (D_tile_row * 
                              self->BYTE_ROWS_PER_TILE * self->D_BYTE_WIDTH) +
                             (D_tile_col * self->TILE_BYTE_WIDTH);
    const size_t mem_stride = self->repeating_bias ? 0 : self->D_BYTE_WIDTH;
    const size_t acc_start   = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;
  const size_t sp_start   = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;
    if (D != NULL && !no_bias) {
      // see sp_tiled_matmul_ws
    }
  }
}

void load_A_tile_into_sp(gemmini_t *self) {
  // calculate mvin parameters
  const size_t A_tile_row = (next_B_tile_col == 0) ?
                             (self->og_cur_tile_row_offset + 1) : 0;
  const size_t A_tile_col = (self->og_cur_tile_col_offset + 1) %
                             self->og_tile_cols;

  const size_t mem_start = self->og_A_offset + 
                           (A_tile_row * 
                            self->BYTE_ROWS_PER_TILE * self->A_BYTE_WIDTH) +
                           (A_tile_col * self->TILE_BYTE_WIDTH);
  const size_t mem_stride = self->A_BYTE_WIDTH;
  const size_t sp_start   = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;

  // issue gemmini commands
  gemmini_config_ld(mem_stride);
  gemmini_mvin(mem_start, sp_start);
}

void matmul_and_accumulate(gemmini_t *self) {
  // calculate compute parameters
  //const size_t A_sp_start = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;
  const size_t A_sp_row_start = self->cur_a_tile_sp_addr;
  const size_t D_sp_row_start = self->USE_D_MATRIX ? self->cur_d_tile_sp_addr 
                                                   : GARBAGE_ADDR;

  // issue gemini commands
  gemmini_config_ex(self->DATAFLOW, 
                    self->ACTIVATION, 
                    self->SYSTOLIC_OUTPUT_RSHIFT, 
                    self->ACCUMULATOR_OUTPUT_RSHIFT,
                    self->RELU6_INPUT_LSHIFT);

  gemmini_compute_accumulated(A_sp_row_start, D_sp_row_start);
}

void store_C_tile_into_mem(gemmini_t *self) {
  // calculate mvout parameters
  const size_t C_tile_row = self->og_cur_tile_row_offset;
  const size_t C_tile_col = self->og_cur_tile_col_offset;
  const size_t mem_start = self->og_C_offset + 
                           (C_tile_row * 
                            self->BYTE_ROWS_PER_TILE * self->C_BYTE_WIDTH) +
                           (C_tile_col * self->TILE_BYTE_WIDTH);
  const size_t mem_stride = self->C_BYTE_WIDTH;
  const size_t sp_start   = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;

  // issue gemmini commands
  gemmini_config_st(mem_stride);
  gemmini_mvout(mem_start, sp_start);
}

//============================================================================
// ...
//============================================================================

void is_A_at_last_subcol(gemmini_t *self) {
  return self->og_cur_tile_col_offset == self->;
  return self->og_cur_tile_row_offset == self->og_end_tile_row;
}

void is_B_at_first_subcol_in_subrow(gemmini_t *self) {
  return self->og_cur_tile_col_offset == 0;
}

void load_A_tile_into_sp(gemmini_t *self) {
  const size_t mem_start = self->A_addr + 
      (self->og_start_tile_row + self->og_cur_tile_row_offset) *
        self->ROWS_PER_TILE * self->A_WIDTH_B +
      (self->og_cur_tile_k_index * 
        self->ROWS_PER_TILE * self->TILE_WIDTH_B);

  const size_t mem_stride = self->A_WIDTH_B;

  const size_t A_bank_num = self->og_cur_tile_row_offset % self->NUM_SP_BANK;
  const size_t A_bank_row = (self->og_cur_tile_row_offset / 
                             self->TILES_PER_BANK) * self->ROWS_PER_TILE;
  const size_t A_sp_addr = (A_bank_num * self->ROWS_PER_SP_BANK) + A_bank_row;

  gemmini_config_ld(mem_stride);
  gemmini_mvin(mem_start, sp_start);
}

void matmul_into_accum(gemmini_t *self) {
  const size_t A_bank_num = self->og_cur_tile_row_offset % self->NUM_SP_BANK;
  const size_t A_bank_row = (self->og_cur_tile_row_offset / 
                             self->TILES_PER_BANK) * self->ROWS_PER_TILE;
  const size_t A_addr = (A_bank_num * self->ROWS_PER_SP_BANK) + A_bank_row;

  gemmini_compute_accumulated(A_addr, GARBAGE_ADDR);
}

void store_C_tile_into_mem(gemmini_t *self) {
  const size_t mem_start = self->C_addr + 
      (self->og_start_tile_row + self->og_cur_tile_row_offset) *
        self->ROWS_PER_TILE * self->C_WIDTH_B +
      (self->og_start_tile_col + self->og_cur_tile_col_offset) *
          self->ROWS_PER_TILE * self->TILE_WIDTH_B;

  const size_t mem_stride = self->C_WIDTH_B;

  const size_t sp_start = 
    ACC_ADDR_RD(self->og_cur_tile_index * self->ROWS_PER_TILE);

  gemmini_config_st(mem_stride);
  gemmini_mvout(mem_start, sp_start);
}




// 2-D array coordinates: upper-left is (0,0), and lower-right is (M,N)
void incr_output_group(gemmini_t *self) {
  const size_t next_og_row = self->og_index / self->OG_ROW_MAX;
  const size_t next_og_col = self->og_index / self->OG_COL_MAX;
  self->og_index += 1;

  // x,y location of output-group in output-group 2-D map
  self->og_row = next_og_row;
  self->og_col = next_og_col;

  // the x,y location of the upper-left tile in the output-group
  self->og_start_tile_col = next_og_col * self->OG_TILE_WIDTH;
  self->og_start_tile_row = next_og_row * self->OG_TILE_HEIGHT;

  // the x,y location of the lower-right tile in the output-group
  self->og_end_tile_col = (self->og_col < (self->OG_COL_MAX - 1))
                        ? (self->og_start_tile_col + self->OG_TILE_WIDTH - 1)
                        : (self->OM_TILE_COLS - 1);

  self->og_end_tile_row = (self->og_row < (self->OG_ROW_MAX - 1))
                        ? (self->og_start_tile_row + self->OG_TILE_HEIGHT - 1)
                        : (self->OM_TILE_ROWS - 1);
}

void do_matmul(struct Gemmini* self) {
  const size_t num_output_groups = self->num_output_groups();
  for(size_t ogroup=0; ogroup < num_output_groups; ogroup += 1) {
    self->incr_output_group(self);
    self->do_next_output_group(self);
  }
}

void is_last_subrow(gemmini_t *self) {
  return self->og_cur_tile_row_offset == (self->og_tile_rows - 1);
}

void is_last_subcol(gemmini_t *self) {
  return self->og_cur_tile_col_offset == (self->og_tile_cols - 1);
}


// - if 32nd bit of scratchpad addr is high, it refers to accumulator
//   - if 31st bit is high, results are accumulated, otherwise, overwritten
void begin_initial_weight_row(struct Gemmini* self) {
  // load this weigth tile into accum from memory
  // prefetch the next weight tile into prefetch scratchpad bank
  gemmini_config_ld(
  matmul.preload(

}

  gemmini_config_ld();

//============================================================================
// Tiling Loop #1: update output-group, a 2D tile-array within the C matrix
//============================================================================
void has_next_output_group(const gemmini_t *self) {
  return self->og_index < (self->NUM_OUTPUT_GROUPS - 1);
}

void incr_output_group(gemmini_t *self) {
  // set tile (x,y) coordinates of top-left tile in output-group
  if((self->og_start_tile_col + self->TILE_COLS_PER_GROUP) >=
      self->C_TILE_WIDTH) {
    self->og_start_tile_col = 0;
    self->og_start_tile_row += self->TILE_ROWS_PER_GROUP;
  } else {
    self->og_start_tile_col += self->TILE_COLS_PER_GROUP;
  }

  // how many tiles rows and cols are in this output-group 
  self->og_tile_rows = min(self->TILE_ROWS_PER_GROUP,
                           self->C_TILE_HEIGHT - self->og_start_tile_row);
  self->og_tile_cols = min(self->TILE_COLS_PER_GROUP,
                           self->C_TILE_WIDTH - self->og_start_tile_col);

  // set tile (x,y) coordinates of bottom-right tile in output-group
  self->og_end_tile_row = self->og_start_tile_row + self->og_tile_rows - 1;
  self->og_end_tile_col = self->og_start_tile_col + self->og_tile_cols - 1;

  // TODO: should these be C_BYTE_WIDTH or C_ELEM_WIDTH?

  // set top-left address of A, B, C, sub-matrices in memory
  self->og_C_offset = (self->og_start_tile_row *
                       self->BYTE_ROWS_PER_TILE * self->C_BYTE_WIDTH) +
                      (self->og_start_tile_col * self->TILE_BYTE_WIDTH);
  self->og_A_offset = (self->og_start_tile_row *
                       self->BYTE_ROWS_PER_TILE * self->A_BYTE_WIDTH);
  self->og_B_offset = (self->og_start_tile_col * self->TILE_BYTE_WIDTH);

  if(bias) {
    if(repeating bias) {
      // set to first row offset
    } else {
      self->og_D_offset = (self->og_start_tile_row *
                           self->BYTE_ROWS_PER_TILE * self->D_BYTE_WIDTH) +
                          (self->og_start_tile_col * self->TILE_BYTE_WIDTH);
    }
  }

  // reset the mulable pointers used by sub-loops
  self->og_cur_tile_index       = 0;
  self->og_cur_tile_col_offset  = 0;
  self->og_cur_tile_row_offset  = 0;

  self->og_cur_tile_k_index       = 0;
  self->og_cur_tile_k_index_next  = 0;

  // 0 == accumulate with exisitng partial sums, 1 == first partial sum
  self->og_use_accumulators       = 0;

  // alternate between 0 and 1. the last 2 tile slots in scratchpad are 
  // reserved for prefetching the next B-tile into the array.
  self->og_cur_B_tile_sp_index = 0;

  // update the next output-group's index
  self->og_index += 1;
}

//============================================================================
// Tiling Loop #2: accumulate 1 output-group partial sum in the accumulators
//============================================================================
void has_next_A_tile_subcol(gemmini_t *self) {
  return self->og_cur_tile_k_index_next < self->A_TILE_WIDTH;
}

void incr_A_tile_subcol(gemmini_t *self) {
  // update the new K index for this output-group
  self->og_cur_tile_k_index = self->og_cur_tile_k_index_next;

  // now perform any actions 
  if(is_first_A_tile_subcol(self)) {
    reset_accumulators(self);
  } else {
    use_accumulators(self);
  }
  if(self->is_first_B_tile_subrow(self)) {
    load_first_B_tile_into_sp(self);
  }

  // update the next K index for this output-group
  self->og_cur_tile_k_index_next += 1;
}

//============================================================================
// Tiling Loop #3
//============================================================================
void has_next_B_tile_subcol_in_subrow(gemmini_t *self) {
  return self->og_cur_tile_col_offset < (self->og_tile_cols - 1);
}

void incr_B_tile_subcol_in_subrow(gemmini_t *self) {
  // update the new row index for this output-group
  self->og_cur_tile_row_offset = self->og_cur_tile_row_offset_next;

  // now perform any actions 
  self->preload_B_tile_into_array(self);
  if(!(self->is_last_tile_subrow(self) || self->is_last_tile_subcol(self))) {
    self->load_next_B_tile_into_sp(self);
  }

  // update the next row index for this output-group
  self->og_cur_tile_row_offset_next += 1;
}

//============================================================================
// Tiling Loop #4
//============================================================================
void has_next_A_tile_subrow_in_subcol(gemmini_t *self) {
  return self->og_cur_tile_row_offset < (self->og_tile_rows - 1);
}

void incr_A_tile_subrow_in_subcol(gemmini_t *self) {
  // update the new subrow-index for this output-group
  self->og_cur_tile_row_offset = self->og_cur_tile_row_offset_next;

  incr_loop_4_counters(self);

  // now perform any actions 
  if(self->is_first_k_tile_subcol(self)) {
    self->load_D_tile_into_sp(self);
  }
  if(self->is_first_tile_subcol(self)) {
    self->load_A_tile_into_sp(self);
  }
  self->matmul_and_accumulate(self);
  if(self->is_k_subcol_last_tile(self)) {
    self->store_C_tile_into_mem(self);
  }

  // update the next subrow-index for this output-group
  self->og_cur_tile_row_offset_next += 1;
}

//============================================================================
// Input Validation
//============================================================================
bool is_valid_to_continue(gemmini_t *self, 
                          enum tiled_matmul_type_t tiled_matmul_type) {
  // basic sanity checks
  if (tiled_matmul_type == OS) {
    printf("Output-stationary dataflow unsupported for EE290 class\n");
    exit(1);
  } else if (tiled_matmul_type == CPU) {
    matmul_cpu(self->M, self->N, self->K, 
               self->A, self->B, self->D, self->C, 
               self->act, self->shift, self->repeating_bias);
    return false;
  }

  // TODO: check that input matrices are all multiple of tile-size!
  // TODO: other validation
 
  // Make sure that the tiling factors make sense
  if (tile_I * DIM > dim_I) {
    printf("tile_I is too large (tile_I * DIM > dim_I)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J) {
    printf("tile_J is too large (tile_J * DIM > dim_J)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K) {
    printf("tile_K is too large (tile_K * DIM > dim_K)\n");
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
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
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
  // create the state object
  gemmini_t *self = create_gemmini(dim_I, dim_J, dim_K, A, B, D, C, 
                                   act, shift, repeating_bias);

  // sanitize inputs before starting
  if(!is_valid_to_continue(self, tiled_matmul_type)) {
    // cleanup the state object
    destroy_gemmini(self);
    return;
  }

  // actually do the tiled matmul
  reset_output_group(self);
  do {
    reset_A_tile_subcol(self);
    do {
      while(has_next_B_tile_subcol_in_subrow(self)) {
        incr_B_tile_subcol(self);
        while(has_next_A_tile_subrow_in_subcol(self)) {
          incr_A_tile_subrow_in_subcol(self);
        }
      }
    } while(next_A_tile_subcol(self));
  } while(next_output_group(self));

  // cleanup the state object
  destroy_gemmini(self);
}

#endif // __GEMMINI_TILER_H__

