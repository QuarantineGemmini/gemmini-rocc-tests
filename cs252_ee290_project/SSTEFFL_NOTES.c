// main todos:
// TODO: create scratchpad->scratchpad move instruction. not sure what i'd use
//       this for though
// TODO: more powerful DMA semantics (stride, skip, count, repeat), which will
//       enable doing im2col and col2im simply by reading memory in/out with 
//       the DMA engine
// TODO: chisel printf formatting is infuriating. i can't format that shit with pads
// TODO: is there no way to get midas printf to work in a non-firesim build?
// TODO: when enabling verbose logging in rocket-chip, i can't selectively
//       turn off logging from the rocket core... it outputs shit every single
//       cycle. i only want output from MY modules. rocket-chip should have
//       printf-groups, which you can selectively enable. this needs a new
//       printf-command in chisel, as well as in firrtl, and a new backend
//       annotator. A grosser method would be to use annotations around your
//       printf, or statement block
// TODO: chisel elaboration fails with JavaNumberFormatException. There is no
//       way for me to debug this... this is less helpful than a vendor tool
// TODO: firesim worker runs forever at 100% CPU if invalid memory accesses are
//       done. Spike will exit immediately due to out-of-range exception, but
//       firesim simulates the actual hardware, so that never happens. i NEED
//       to figure out how to abort a firesim sim partway through
//
// TODO: D<->C untracked dependency bug in ROB since they are in different
//       address spaces??? is this a real bug, or is it a non-issue?
// TODO: why do i have to do & 0xffffffff? otherwise, 
//       D_acc_row_addr = 0xffffffff80000000, instead of 0x0000000180000000
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
// TODO: ALL MY HARDWARE CHANGES WERE NOT WORTH IT!!! The AGFI-id is baked
//       into the config_hwdb.ini file, so even if i convert the chisel to
//       the fpga-verilog, that is NOT the agfi image i will run. fuck.
// TODO: you can't simply change your workload and quickly re-run firesim, you
//       have to re-build the kernel image with the new benchmarks on it!!!
//       a better idea would be to git-clone, then build within the simulated
//       image, but then it needs your github credentials baked into it (unless
//       its a public repo). anyways, incremental software changes take forever
//       and this sucks
// TODO: when is firesim worth it. it takes 10-20 minutes to recompile, then
//       start the simulation. if the simulation only takes 20 minutes on 
//       vcs, its probably not worth it to run in firesim.
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

  // BUG: no malloc in bare-metal
  //gemmini_t *self = (gemmini_t *) malloc(sizeof(gemmini_t));
  // BUG: no assert in bare-metal
  // assert(M % DIM == 0 && M > 0);
  // assert(N % DIM == 0 && N > 0);
  // assert(K % DIM == 0 && K > 0);
