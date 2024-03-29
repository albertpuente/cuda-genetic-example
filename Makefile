all: genetic cuda # geneticViewer

# CUDA
CUDA_HOME   = /Soft/cuda/6.5.14
NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 -code=sm_35 -rdc=true -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudadevrt -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib


# C
CFLAGS=-O2 -lm -std=c99 -w
GFLAGS=-lGL -lglut -lGLU

genetic: genetic.c
	gcc genetic.c -o genetic $(CFLAGS)


geneticViewer: geneticViewer.c
	gcc geneticViewer.c -o geneticViewer $(CFLAGS) $(GFLAGS)

show:	genetic geneticViewer
	./genetic > data && ./geneticViewer data

exec:	genetic
	./genetic 1

berry:	cuda
	./geneticCUDA 1

clean:
	rm -f *.o genetic geneticCUDA SESION* geneticViewer

cuda.o: genetic.cu
	$(NVCC) -c -o $@ genetic.cu $(NVCC_FLAGS)

cuda:	cuda.o
	$(NVCC) cuda.o -o geneticCUDA $(LD_FLAGS)

genetic_mgpu.o: genetic_mgpu.cu
	$(NVCC) -c -o $@ genetic_mgpu.cu $(NVCC_FLAGS)

genetic_mgpu: genetic_mgpu.o
	$(NVCC) genetic_mgpu.o -o genetic_mgpu $(LD_FLAGS)

sub:	cuda
	rm -f SESION*
	qsub -l cuda job.sh && watch -n 0.5 qstat

sub_mgpu:	genetic_mgpu
	rm -f SESION*
	qsub -l cuda job_mgpu.sh && watch -n 0.5 qstat

subseq: genetic
	qsub -l cuda jobseq.sh && watch -n 0.5 qstat

genetic_par.o: genetic_par.cu
	$(NVCC) -c -o $@ genetic_par.cu $(NVCC_FLAGS)

genetic_par: genetic_par.o
	$(NVCC) genetic_par.o -o genetic_par $(LD_FLAGS)

sub_par: genetic_par
	rm -f SESION*
	qsub -l cuda job.sh && watch -n 0.5 qstat
