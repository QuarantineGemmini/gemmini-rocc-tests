include $(abs_top_srcdir)/Makefrag

#tests = \
#	identity \
#	identity_negative \
#	double \
#	double_keep_weights \
#	double_keep_weights_then_change \
#	random_matmuls \
#	large_matmul \
#	large_matmul_without_cpu \
#	very_large_matmul \
#	cifar_quant

tests = \
	large_matmul \
	large_matmul_without_cpu \

tests_baremetal := $(tests:=-baremetal)
ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
endif

BENCH_COMMON    := $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS := $(abs_top_srcdir)/include/gemmini.h						\
									 $(abs_top_srcdir)/include/gemmini_params.h			\
									 $(abs_top_srcdir)/include/gemmini_isa.h				\
									 $(abs_top_srcdir)/include/gemmini_tiler_orig.h	\
									 $(abs_top_srcdir)/include/gemmini_tiler_fsm.h	\
									 $(abs_top_srcdir)/include/gemmini_tiler_hw.h

ID_STRING ?=
LIBS ?=
LDFLAGS ?=

#CFLAGS_ALL := \
#  $(CFLAGS) \
#	-DPREALLOCATE=1 \
#	-DMULTITHREAD=1 \
#	-mcmodel=medany \
#	-std=gnu99 \
#	-O2 \
#	-ffast-math \
#	-fno-common \
#	-fno-builtin-printf \
#	-march=rv64gc -Wa,-march=rv64gcxhwacha \
#	-lm \
#	-lgcc \
#	-I$(abs_top_srcdir)/riscv-tests \
#	-I$(abs_top_srcdir)/riscv-tests/env \
#	-I$(abs_top_srcdir) \
#	-I$(BENCH_COMMON) \
#	-DID_STRING=$(ID_STRING) \

#CFLAGS_BAREMETAL := \
#	$(CFLAGS_ALL) \
#	-nostdlib \
#	-nostartfiles \
#	-static \
#	-T $(BENCH_COMMON)/test.ld \

CFLAGS_ALL := \
  $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-Og \
	-ffast-math \
	-fbranch-probabilities \
	-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-DID_STRING=$(ID_STRING) \

CFLAGS_BAREMETAL := \
	$(CFLAGS_ALL) \
	-static


all: $(tests_baremetal) $(tests_linux)

vpath %.c $(src_dir)

#%-baremetal: %.c $(GEMMINI_HEADERS)
#	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LDFLAGS) -o $@ \
#		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LDFLAGS) -o $@

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_ALL) $< $(LDFLAGS) -o $@

junk += $(tests_baremetal) $(tests_linux)

