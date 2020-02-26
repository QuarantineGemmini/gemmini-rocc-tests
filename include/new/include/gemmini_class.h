
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
  g->acc_group_height = sp_banks * tiles_per_sp_bank - 1;
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

#define ACC_ADDR_RD(addr) ((1 << 31) & (addr))

void store_C_tile_into_mem(Gemmini_t *self) {
  const size_t mem_start = self->C_addr + 
      (self->og_start_tile_row + self->og_cur_tile_row) *
        self->ROWS_PER_TILE * self->C_WIDTH_B +
      (self->og_start_tile_col + self->og_cur_tile_col) *
          self->ROWS_PER_TILE * self->TILE_WIDTH_B;

  const size_t mem_stride = self->C_WIDTH_B;

  const size_t sp_start = 
    ACC_ADDR_RD(self->og_cur_tile_index * self->ROWS_PER_TILE);

  gemmini_config_st(mem_stride);
  gemmini_mvout(mem_start, sp_start);
}

void has_next_output_group(Gemmini_t *self) {
  return self->output_group_index < self->NUM_OUTPUT_GROUPS;
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

// - if 32nd bit of scratchpad addr is high, it refers to accumulator
//   - if 31st bit is high, results are accumulated, otherwise, overwritten
void begin_initial_weight_row(struct Gemmini* self) {
  // load this weigth tile into accum from memory
  // prefetch the next weight tile into prefetch scratchpad bank
  gemmini_config_ld(
  matmul.preload(

}

  gemmini_config_ld();

void incr_A_tile_subcol(Gemmini_t *self) {
  // TODO: do stuff with indices and state
  if(self->is_first_A_tile_col()) {
    self->make_accumulators_zero();
  } else {
    self->use_accumulators();
  }
}

void incr_B_tile_subrow(Gemmini_t *self) {
  // TODO: do stuff with indices and state
  if(first_row in group) {
    self->load_first_B_tile_into_sp();
  }
}

void incr_B_tile_subcol(Gemmini_t *self) {
  // TODO: do stuff with indices and state
  self->preload_B_tile_into_array();
  if(not_last_row || not_last_col) {
    self->load_next_B_tile_into_sp();
  }
}

void incr_A_tile_subrow(Gemmini_t *self) {
  // TODO: do stuff with indices and state
  if(self->is_B_at_first_subcol_in_subrow()) {
    self->load_A_tile_into_sp();
  }
  self->matmul_into_accum();
  if(self->is_A_at_last_subrow()) {
    self->store_C_tile_into_mem();
  }
}



//foreach output group from L->R, T->B
//  foreach K-col in input matrix for this output group
//    if first K-col in input group, zero the accumulator, else keep it
//    for each corresponding row in weight matrix 
//      load first W tile into sp
//      for each weight tile in row
//        preload W tile into array from sp
//        load next W tile into sp
//        for each vertical tile in K-col image group
//          if first col in weight-row
//            load K-col-tile into sp
//          matmul.accum I*A -> accum
//          if last K-col in input group
//            mvout accum to memory

void do_matmul(Gemmini_t* self) {
  while(self->has_next_output_group(self)) {
    self->incr_output_group(self);
    while(self->has_next_A_tile_subcol(self)) {
      self->incr_A_tile_subcol(self);
      while(self->has_next_B_tile_subrow(self)) {
        self->incr_B_tile_subrow(self);
        while(self->has_next_B_tile_subcol(self)) {
          self->incr_B_tile_subcol(self);
          while(self->has_next_A_tile_subrow(self)) {
            self->incr_A_tile_subrow(self);
          }
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


