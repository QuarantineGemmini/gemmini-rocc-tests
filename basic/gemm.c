#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"

#include "include/get_real_time.h"
#include "include/util.h"

//==========================================================================
// usage message
//==========================================================================
void print_usage() {
  print("\n\
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

  double time_parse, time_init, time_pin, time_gemmini, time_cpu,
         time_verify, time_all;
  bool success  = false;
  
  //-------------
  // parse args
  //-------------
  time_parse = get_real_time();
  if(argc == 0) print_usage();
  if(argc < 4) ERROR("missing <M>, <N> or <K>. see usage with -h");

  size_t tmp;
  for(int i=0; i<argc; i=i+1) {
    if(!strcmp(argv[i], "-h"))                print_usage();
    else if(!strcmp(argv[i], "-help"))        print_usage();
    else if(!strcmp(argv[i], "-verify"))      verify = true;
    else if(!strcmp(argv[i], "-no_d"))        no_d = true;
    else if(!strcmp(argv[i], "-repeat_d"))    repeat_d = true;
    else if(!strcmp(argv[i], "-zeros"))       zeros = true;
    else if(!strcmp(argv[i], "-diag"))        diag = true;
    else if(!strcmp(argv[i], "-dump"))        dump = true;
    else if(!strcmp(argv[i], "%u", &tmp)) {
      if(tmp == 0) ERROR("cannot specify zero as an <M,N,K> dimension");
      else if(m == 0) m = tmp;
      else if(n == 0) n = tmp;
      else if(k == 0) k = tmp;
      else ERROR("too many arguments. try -h");
    }
    else ERROR("unrecognized argument: %s", argv[i]);
  }
  if (m==0 || n==0 || k==0) ERROR("you must specify all 3 <M,N,K> params!");

  time_parse = get_real_time() - time_parse;
  DEBUG("parse time: %.6d (s)", time_parse);

  //---------------------
  // initialize matrices
  //---------------------
  time_init = get_real_time();
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

  time_init = get_real_time() - time_init;
  DEBUG("init time:  %.6d (s)", time_init);

  //---------------------
  // pin matrices
  //---------------------
  time_pin = get_real_time();
  pin_all();
  gemmini_flush(0);
  time_pin = get_real_time() - time_pin;
  DEBUG("pin time: %.6d (s)", time_pin);

  //---------------------
  // gemmini matmul
  //---------------------
  time_gemmini = get_real_time();
  gemm_auto(m, n, k, A, B, C_gemmini, D, repeat_D, WS);
  DEBUG("gemmini time: %.6d (s)", time_gemmini);

  //---------------------
  // cpu matmul
  //---------------------
  time_cpu = get_real_time();
  if(verify) gemm_auto(m, n, k, A, B, C_gold, D, repeat_D, CPU);
  time_cpu = get_real_time() - time_cpu;
  DEBUG("cpu time: %.6d (s)", time_cpu);

  //---------------------
  // verify matmul
  //---------------------
  time_verify = get_real_time();
  if(verify) {
    success = compare_matrices_o(C_gemmini, C_gold, m, n);
  }
  time_verify = get_real_time() - time_verify;
  DEBUG("verify time: %.6d (s)", time_verify);

  //---------------------
  // print summary
  //---------------------
  time_all = time_parse + time_init + time_pin + time_gemmini + 
             time_cpu + time_verify;
  PRINT("-----------------------------");
  PRINT(" STATUS: %s", (status ? "PASS" : "FAIL"));
  PRINT("-----------------------------");
  PRINT("total   (s,%): %.6f, %d", time_all,    100*(time_all     /time_all));
  PRINT("parse   (s,%): %.6f, %d", time_parse,  100*(time_parse   /time_all));
  PRINT("init    (s,%): %.6f, %d", time_init,   100*(time_init    /time_all));
  PRINT("pin     (s,%): %.6f, %d", time_pin,    100*(time_pin     /time_all));
  PRINT("gemmini (s,%): %.6f, %d", time_gemmini,100*(time_gemmini /time_all));
  PRINT("cpu     (s,%): %.6f, %d", time_cpu,    100*(time_cpu     /time_all));
  PRINT("verify  (s,%): %.6f, %d", time_verify, 100*(time_verify  /time_all));

  exit(success ? 0 : 1);
}

