
#include "include/gemmini_obj.h"

//============================================================================
struct Gemmini {
  (void)(do_matmul(struct Gemmini *));
  (void)(do_next_output_group(struct Gemmini *));

  size_t state...
};

//============================================================================
struct Gemmini* create_gemmini(size_t M, size_t, N, elem_t *I, elem_t *W, elem_t *O) {
  struct Gemmini* g = (struct state*)malloc(sizeof(struct state));

  const size_t tile_rows    = DIM;
  const size_t tile_cols    = DIM;
  const size_t tile_elems   = tile_rows * tile_cols;
  const size_t tile_width_b = tile_cols * sizeof(elem_t);
  const size_t tile_size_b  = tile_rows * tile_width_b;

  // each accum and scratchpad-bank row holds 1 tile
  const size_t sp_banks           = BANK_NUM;
  const size_t tiles_per_sp_bank  = BANK_ROWS / tile_rows;
  const size_t tiles_per_acc      = ACC_ROWS / tile_rows;

  //============================================================================
  // - the 2-D shape of tiles in the accumulator. This shape depends on the 
  //   relative size of the scratchpad, accumulator, and systolic array. 
  // - where the formula comes from:
  //   - a single column of 'acc_group_height' tiles from the input matrix are 
  //     loaded into scratchpad one by one
  //   - the '-1' is subtracted from 'acc_group_height' because I always save
  //     1 slot in the last bank for preloading the next weights from DRAM.
  //   - the input tiles in the scratchpad are reused 'acc_group_width' times
  //   - each weight tile is pre-loaded into scratchpad before it is needed, and
  //     when it is needed, it is used 'acc_group_height' times in a row without
  //     leaving the systolic array. Then it is discarded.
  // --------------------------------------------------------------------------
  // - I believe this formula optimally uses the scratchpad and accumulator
  //   for data-reuse regardless of their relative sizes.
  //============================================================================
  g->acc_group_height = sp_banks * tiles_per_sp_bank - 2;
  g->acc_group_width  = tiles_per_acc / acc_group_height;

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

void is_first_A_tile_subcol(Gemmini_t *self) {
  // - returns true if we are currently at the first tile column in the 
  //   A-matrix, which is useful for resetting the partial sums of the
  //   current output group to zero inside gemmini's accumulators
  return self->og_cur_tile_k_index == 0;
}

void reset_accumulators(Gemmini_t *self) {
  self->og_use_accumulators = 0;
}

void use_accumulators(Gemmini_t *self) {
  self->og_use_accumulators = 1;
}

void is_first_B_tile_subrow(Gemmini_t *self) {
  return self->og_cur_tile_row_offset == 0;
}

void load_first_B_tile_into_sp(Gemmini_t *self) {
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

void preload_B_tile_into_arry(Gemmini_t *self) {
  // calculate preload parameters
  const size_t B_sp_start = self->og_cur_B_tile_sp_start;
  const size_t C_sp_start = self->og_use_accumulators ? 
                            ACC_ADDR_ACC(self->og_cur_tile_index) :
                            ACC_ADDR_NEW(self->og_cur_tile_index);
  // issue gemmini commands
  gemmini_preload(B_sp_start, C_sp_start);
}

bool is_last_tile_subrow(Gemmini_t *self) {
  // are we currently computing a tile in the last subrow of the output-group
  return self->og_cur_tile_row_offset == (self->og_end_tile_row - 1);
}

bool is_first_tile_subcol(Gemmini_t *self) {
  // are we currently computing a tile in the first subcol of the output-group
  return self->og_cur_tile_col_offset == 0;
}

bool is_last_tile_subcol(Gemmini_t *self) {
  // are we currently computing a tile in the last subcol of the output-group
  return self->og_cur_tile_col_offset == (self->og_end_tile_col - 1);
}

bool is_k_last_tile_subcol(Gemmini_t *self) {
  // are we currently on the last k-subcol in the input A-matrix
  return self->og_cur_tile_k_index == (self->A_TILE_WIDTH - 1);
}

void load_next_B_tile_into_sp(Gemmini_t *self) {
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

//===========================================================================
// Tiling Loop #4: Helpers
//===========================================================================

void load_A_tile_into_sp(Gemmini_t *self) {
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

void matmul_and_accumulate(Gemmini_t *self) {
  // calculate compute parameters
  const size_t sp_start   = self->og_cur_tile_row * self->BYTE_ROWS_PER_TILE;

  // issue gemini commands
  gemmini_compute_accumulated(sp_start, GARBAGE_ADDR);
}

void store_C_tile_into_mem(Gemmini_t *self) {
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

void is_A_at_last_subcol(Gemmini_t *self) {
  return self->og_cur_tile_col_offset == self->;
  return self->og_cur_tile_row_offset == self->og_end_tile_row;
}

void is_B_at_first_subcol_in_subrow(Gemmini_t *self) {
  return self->og_cur_tile_col_offset == 0;
}

void load_A_tile_into_sp(Gemmini_t *self) {
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

void matmul_into_accum(Gemmini_t *self) {
  const size_t A_bank_num = self->og_cur_tile_row_offset % self->NUM_SP_BANK;
  const size_t A_bank_row = (self->og_cur_tile_row_offset / 
                             self->TILES_PER_BANK) * self->ROWS_PER_TILE;
  const size_t A_addr = (A_bank_num * self->ROWS_PER_SP_BANK) + A_bank_row;

  gemmini_compute_accumulated(A_addr, GARBAGE_ADDR);
}

void store_C_tile_into_mem(Gemmini_t *self) {
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
void incr_output_group(Gemmini_t *self) {
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

void is_last_subrow(Gemmini_t *self) {
  return self->og_cur_tile_row_offset == (self->og_tile_rows - 1);
}

void is_last_subcol(Gemmini_t *self) {
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
void has_next_output_group(Gemmini_t *self) {
  return self->og_index_next < self->NUM_OUTPUT_GROUPS;
}

void incr_output_group(Gemmini_t *self) {
  // update the new output-group's index
  self->og_index = self->og_index_next;

  // set tile (x,y) coordinates of top-left tile in output-group
  self->og_start_tile_row = (self->og_index / self->GROUPS_COLS_PER_OUTPUT) *
                             self->TILE_ROWS_PER_GROUP;
  self->og_start_tile_col = (self->og_index % self->GROUPS_COLS_PER_OUTPUT) *
                             self->TILE_COLS_PER_GROUP;

  // how many tiles rows and cols are in this output-group 
  self->og_tile_rows = min(self->TILE_ROWS_PER_GROUP,
                           self->C_TILE_HEIGHT - self->og_start_tile_row);
  self->og_tile_cols = min(self->TILE_COLS_PER_GROUP,
                           self->C_TILE_WIDTH - self->og_start_tile_col);

  // set tile (x,y) coordinates of bottom-right tile in output-group
  self->og_end_tile_row = self->og_start_tile_row + self->og_tile_rows - 1;
  self->og_end_tile_col = self->og_start_tile_col + self->og_tile_cols - 1;

  // set top-left address of A, B, C, sub-matrices in memory
  self->og_C_offset = (self->og_start_tile_row *
                       self->BYTE_ROWS_PER_TILE * self->C_BYTE_WIDTH) +
                      (self->og_start_tile_col * self->TILE_BYTE_WIDTH);
  self->og_A_offset = (self->og_start_tile_row *
                       self->BYTE_ROWS_PER_TILE * self->A_BYTE_WIDTH);
  self->og_B_offset = (self->og_start_tile_col * self->TILE_BYTE_WIDTH);

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
  self->og_index_next += 1;
}

//============================================================================
// Tiling Loop #2: accumulate 1 output-group partial sum in the accumulators
//============================================================================
void has_next_A_tile_subcol(Gemmini_t *self) {
  return self->og_cur_tile_k_index_next < self->A_TILE_WIDTH;
}

void incr_A_tile_subcol(Gemmini_t *self) {
  // update the new K index for this output-group
  self->og_cur_tile_k_index = self->og_cur_tile_k_index_next;

  // now perform any actions 
  if(self->is_first_A_tile_subcol(self)) {
    self->reset_accumulators(self);
  } else {
    self->use_accumulators(self);
  }
  if(self->is_first_B_tile_subrow(self)) {
    self->load_first_B_tile_into_sp(self);
  }

  // update the next K index for this output-group
  self->og_cur_tile_k_index_next += 1;
}

//============================================================================
// Tiling Loop #3
//============================================================================
void has_next_B_tile_subcol_in_subrow(Gemmini_t *self) {
  return self->og_cur_tile_col_offset < (self->og_tile_cols - 1);
}

void incr_B_tile_subcol_in_subrow(Gemmini_t *self) {
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
void has_next_A_tile_subrow_in_subcol(Gemmini_t *self) {
  return self->og_cur_tile_row_offset < (self->og_tile_rows - 1);
}

void incr_A_tile_subrow_in_subcol(Gemmini_t *self) {
  // update the new subrow-index for this output-group
  self->og_cur_tile_row_offset = self->og_cur_tile_row_offset_next;

  // now perform any actions 
  if(self->is_first_tile_subcol(self)) {
    self->load_A_tile_into_sp(self);
  }
  self->matmul_and_accumulate(self);
  if(self->is_k_last_tile_subcol(self)) {
    self->store_C_tile_into_mem(self);
  }

  // update the next subrow-index for this output-group
  self->og_cur_tile_row_offset_next += 1;
}

//============================================================================
// Entry Point to MatMul
//============================================================================

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

void do_matmul(Gemmini_t* self) {
  while(self->has_next_output_group(self)) {
    self->incr_output_group(self);
    while(self->has_next_A_tile_subcol(self)) {
      self->incr_A_tile_subcol(self);
      while(self->has_next_B_tile_subcol_in_subrow(self)) {
        self->incr_B_tile_subcol(self);
        while(self->has_next_A_tile_subrow_in_subcol(self)) {
          self->incr_A_tile_subrow_in_subcol(self);
        }
      }
    }
  }
}


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


void do_next_output_group(struct Gemmini* self) {
  const size_t acc_start_tile_col = self->og_start_tile_col;
  const size_t acc_end_tile_col   = self->og_end_tile_col;
  const size_t acc_start_tile_row = self->og_start_tile_row;
  const size_t acc_end_tile_row   = self->og_end_tile_row;

  for(size_t w_row = start_w_row; w_row <= end_w_row; w_row += 1) {
    if(w_row == 0) {
      self->begin_initial_weight_row(self);
    } else {
      self->begin_successor_weight_row(self);
    }

    for(weight_col = 0; weight_col < weight_cols ; weight_col += 1) {
      if(weight_col == 0) {
        self->do_initial_weight_col(self);
      } else {
        self->do_successor_weight_col(self);
      }
    }

    
      state->exec_accum();
        do_tiled_acc_group_first_weight_col(state);
      }

  for(row = 0; row < rows; row += 1) {


  do_tiled_acc_group_first_row(state);
  // 1) 


}


