// See LICENSE for license details.

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/matrix_util.h"
#include "include/util.h"

#ifdef GEMMINI_BAREMETAL
int main (int argc, char * argv[]) {
  ERROR("gemm benchmark does not work on baremetal!");
}
# else

//==========================================================================
// usage message
//==========================================================================
void print_usage() {
  printf("\n\
\n\
 gemm [options] <M> <N> <K>\n\
 --------------\n\
 runs gemm on the gemmini accelerator. user can configure many parameters\n\
 such as M, N, K dimensions, data contents (all 1s or 0s, or random), and\n\
 whether to verify against the cpu. The computation performed is C=AB+D,\n\
 where A=MxK, B=KxN, C=MxN and D=MxN. A,B,C are all of input-type, and\n\
 D is of output-type. THIS DOES NOT WORK ON BARE-METAL (use pk or linux)!\n\
\n\
 options\n\
 -------\n\
 -verify            --> verify results against a CPU-only implementation\n\
 -no_d              --> do not use a D matrix in computation (only do C=AB)\n\
 -repeat_d          --> for D, use a repeated row instead of a full matrix\n\
 -zeros             --> all inputs are zeros\n\
 -diag              --> all matrices are diaganol, with 1's on the diaganol\n\
 -dump              --> print all matrices to stdout after computation\n\
 -h|-help           --> show this help\n\
\n\
");
  exit(1);
}

//==========================================================================
// usage message
//==========================================================================
int main (int argc, char * argv[]) {
  size_t m      = 0;
  size_t n      = 0;
  size_t k      = 0;
  bool verify   = false;
  bool no_d     = false;
  bool repeat_d = false;
  bool zeros    = false;
  bool diag     = false;
  bool dump     = false;

  size_t time_parse, time_init, time_pin, time_gemmini, time_cpu,
         time_verify, time_all;
  bool success  = false;
  
  //-------------
  // parse args
  //-------------
  time_parse = read_cycles();
  if(argc == 1) print_usage();
  if(argc < 4) ERROR("missing <M>, <N> or <K>. see usage with -h");

  size_t tmp;
  for(int i=1; i<argc; i+=1) {
    if(!strcmp(argv[i], "-h"))                print_usage();
    else if(!strcmp(argv[i], "-help"))        print_usage();
    else if(!strcmp(argv[i], "-verify"))      verify = true;
    else if(!strcmp(argv[i], "-no_d"))        no_d = true;
    else if(!strcmp(argv[i], "-repeat_d"))    repeat_d = true;
    else if(!strcmp(argv[i], "-zeros"))       zeros = true;
    else if(!strcmp(argv[i], "-diag"))        diag = true;
    else if(!strcmp(argv[i], "-dump"))        dump = true;
    else if(sscanf(argv[i], "%u", &tmp)) {
      if(tmp == 0) ERROR("cannot specify zero as an <M,N,K> dimension");
      else if(m == 0) m = tmp;
      else if(n == 0) n = tmp;
      else if(k == 0) k = tmp;
      else ERROR("too many arguments. try -h");
    }
    else ERROR("unrecognized argument: %s", argv[i]);
  }
  if (m==0 || n==0 || k==0) ERROR("you must specify all 3 <M,N,K> params!");

  time_parse = read_cycles() - time_parse;
  DEBUG("parse time: %.6d (s)", time_parse);

  //---------------------
  // initialize matrices
  //---------------------
  time_init = read_cycles();
  elem_t *A = zeros    ? create_zero_matrix_i(m, k) :
              diag     ? create_diag_matrix_i(m, k) :
                         create_rand_matrix_i(m, k);
  elem_t *B = zeros    ? create_zero_matrix_i(k, n) :
              diag     ? create_diag_matrix_i(k, n) :
                         create_rand_matrix_i(k, n);
  acc_t *D  = no_d     ? NULL :
              repeat_d ? (zeros ? create_zero_matrix_o(1, n) :
                                  create_rand_matrix_o(1, n)) :
              zeros    ? create_zero_matrix_o(m, n) :
              diag     ? create_diag_matrix_o(m, n) :
                         create_rand_matrix_o(m, n);

  elem_t *C_gemmini = create_zero_matrix_i(m, n);
  elem_t *C_gold = verify ? create_zero_matrix_i(m, n) : NULL;

  if(dump) {
    dump_matrix_i("A", A, m, k);
    dump_matrix_i("B", B, k, n);
    if(no_d) PRINT("D = NULL");
    else if(repeat_d) dump_matrix_o("D", D, 1, n);
    else dump_matrix_o("D", D, m, n);
  }

  time_init = read_cycles() - time_init;
  DEBUG("init time:  %.6d (s)", time_init);

  //---------------------
  // pin matrices
  //---------------------
  time_pin = read_cycles();
  pin_all();
  gemmini_flush(0);
  time_pin = read_cycles() - time_pin;
  DEBUG("pin time: %.6d (s)", time_pin);

  //---------------------
  // gemmini matmul
  //---------------------
  time_gemmini = read_cycles();
  gemm_auto(m, n, k, A, B, D, C_gemmini, repeat_d, WS);
  DEBUG("gemmini time: %.6d (s)", time_gemmini);
  if(dump) dump_matrix_i("C_gemmini", C_gemmini, m, n);

  //---------------------
  // cpu matmul
  //---------------------
  time_cpu = read_cycles();
  if(verify) gemm_auto(m, n, k, A, B, D, C_gold, repeat_d, CPU);
  time_cpu = read_cycles() - time_cpu;
  if(dump) dump_matrix_i("C_gold", C_gold, m, n);
  DEBUG("cpu time: %.6d (s)", time_cpu);

  //---------------------
  // verify matmul
  //---------------------
  time_verify = read_cycles();
  if(verify) {
    success = compare_matrices_i(C_gemmini, C_gold, m, n);
  }
  time_verify = read_cycles() - time_verify;
  DEBUG("verify time: %.6d (s)", time_verify);

  //---------------------
  // print summary
  //---------------------
  time_all = time_parse + time_init + time_pin + time_gemmini + 
             time_cpu + time_verify;
  PRINT("--------------------------------");
  PRINT("section          cycles        %%");
  PRINT("--------------------------------");
  PRINT("total:  %15llu   %6.2f", time_all,    PCT(time_all,    time_all));
  PRINT("parse:  %15llu   %6.2f", time_parse,  PCT(time_parse,  time_all));
  PRINT("init:   %15llu   %6.2f", time_init,   PCT(time_init,   time_all));
  PRINT("pin:    %15llu   %6.2f", time_pin,    PCT(time_pin,    time_all));
  PRINT("gemmini:%15llu   %6.2f", time_gemmini,PCT(time_gemmini,time_all));
  PRINT("cpu:    %15llu   %6.2f", time_cpu,    PCT(time_cpu,    time_all));
  PRINT("verify: %15llu   %6.2f", time_verify, PCT(time_verify, time_all));
  PRINT("--------------------------------");
  PRINT("STATUS: %s", (success ? "PASS" : "FAIL"));
  PRINT("--------------------------------");

  exit(success ? 0 : 1);
}

#endif // GEMMINI_BAREMETAL
