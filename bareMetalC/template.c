// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);

  printf("Initialize our input and output matrices in main memory\n");
  elem_t In[DIM][DIM];
  // FIXME: in existing Gemmini tests, `In` is uninitialized. Why?
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      In[i][j] = i+j;
  
  elem_t D[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      D[i][j] = 0;
  
  elem_t Identity[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      Identity[i][j] = i == j;

  elem_t Out[DIM][DIM];

  printf("Configure Gemmini\n");
  gemmini_config_addr_ab(In, Identity);
  gemmini_config_addr_cd(Out, D);
  gemmini_config_size0(DIM, DIM);
  gemmini_config_size1(DIM);
  gemmini_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 0);

  printf("Multiply \"In\" matrix with \"Identity\" matrix with a bias of 0\n");
  gemmini_compute();

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

