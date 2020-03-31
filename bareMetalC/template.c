// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini.h"

//============================================================================
void gemmini1_body(elem_t In[DIM][DIM],       acc_t D[DIM][DIM], 
                   elem_t Identity[DIM][DIM], elem_t Out[DIM][DIM]) 
{
  printf("Configure Gemmini1 ISA\n");
  printf("Calculate the scratchpad addresses of all our matrices\n");
  printf("  Note: The scratchpad is \"row-addressed\", where each address contains one matrix row\n");
  size_t In_sp_addr = 0;
  size_t Out_sp_addr = DIM;
  size_t Identity_sp_addr = 2*DIM;

  printf("Move \"In\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_mvin(In, In_sp_addr);

  printf("Move \"Identity\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_mvin(Identity, Identity_sp_addr);

  printf("Multiply \"In\" matrix with \"Identity\" matrix with a bias of 0\n");
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 0);
  gemmini_preload(Identity_sp_addr, Out_sp_addr);
  gemmini_compute_preloaded(In_sp_addr, GARBAGE_ADDR);

  printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  gemmini_mvout(Out, Out_sp_addr);
}

//============================================================================
void gemmini2_body(elem_t In[DIM][DIM],       acc_t D[DIM][DIM], 
                   elem_t Identity[DIM][DIM], elem_t Out[DIM][DIM]) 
{
  printf("Configure Gemmini2 ISA\n");
  gemmini_config_addr_ab(In, Identity);
  gemmini_config_addr_cd(Out, D);
  gemmini_config_size0(DIM, DIM);
  gemmini_config_size1(DIM);
  gemmini_config_repeating_bias(false);
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 0);

  printf("Multiply \"In\" matrix with \"Identity\" matrix with a bias of 0\n");
  gemmini_compute();

}

//============================================================================
int main() {
  pin_all();
  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);

  printf("Initialize our input and output matrices in main memory\n");
  elem_t In[DIM][DIM];
  // FIXME: in existing Gemmini tests, `In` is uninitialized. Why?
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      In[i][j] = i+j;
  
  acc_t D[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      D[i][j] = 0;
  
  elem_t Identity[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      Identity[i][j] = i == j;

  elem_t Out[DIM][DIM];

  pin_matrices(DIM, DIM, DIM, In, Identity, (const acc_t*)D, Out, false);
#ifdef USE_HW_TILER
  gemmini2_body(In, D, Identity, Out);
#else
  gemmini1_body(In, D, Identity, Out);
#endif
  unpin_matrices();

  printf("Fence till Gemmini completes all memory operations\n");
  gemmini_fence();

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal(In, Out)) {
    printf("Input and output matrices are different!\n");
    printf("\"In\" matrix:\n");
    printMatrix(In);
    printf("\"Out\" matrix:\n");
    printMatrix(Out);
    printf("\n");

    exit(1);
  }
  printMatrix(Out);
  printf("Input and output matrices are identical, as expected\n");
  exit(0);
}

