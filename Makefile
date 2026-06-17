CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   := -O3 -Wall -Wextra
LDFLAGS  := -lm
OMPFLAGS := -fopenmp
NVCCFLAGS := -O3
BINDIR   := bin

.PHONY: all sequential openmp cuda clean run-sequential run-openmp run-cuda run-all check-openmp

all: sequential openmp cuda

sequential: $(BINDIR)/sequential
openmp: $(BINDIR)/openmp
cuda: $(BINDIR)/cuda

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/sequential: kmeans.c | $(BINDIR)
	$(CC) $(CFLAGS) kmeans.c -o $@ $(LDFLAGS)

$(BINDIR)/openmp: kmeans.c | $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) kmeans.c -o $@ $(LDFLAGS)

$(BINDIR)/cuda: kmeans.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) kmeans.cu -o $@

run-sequential: sequential
	./benchmark.sh run sequential

run-openmp: openmp
	./benchmark.sh run openmp

run-cuda: cuda
	./benchmark.sh run cuda

run-all: all
	./benchmark.sh run all

check-openmp: sequential openmp
	./benchmark.sh check openmp

clean:
	rm -rf $(BINDIR)
