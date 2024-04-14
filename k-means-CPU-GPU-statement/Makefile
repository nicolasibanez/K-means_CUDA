######################################################################
# k-means on CPU & GPU
# S. Vialle March 2022 (with the help of G. He)
######################################################################

CPUCC = g++
GPUCC = /usr/local/cuda/bin/nvcc

CXXFLAGS = -O3  #-DDP
CXXFLAGS += -I/usr/local/cuda/include/
CC_CXXFLAGS = -fopenmp -funroll-loops
# When atomic add at block scale ... might need to change the following line
CUDA_TARGET_FLAGS =
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_60 #RTX 1080
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_61 #RTX 2080
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_62 #RTX 2080
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_75 #RTX 2080
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_80 #RTX 3090
CUDA_CXXFLAGS = $(CUDA_TARGET_FLAGS)  

CC_LDFLAGS =  -fopenmp 
CUDA_LDFLAGS = -L/usr/local/cuda/lib64/   

CC_LIBS =  
CUDA_LIBS = -lcuda -lcublas -lcudart -lcurand

CC_SOURCES =  main.cc init.cc kmeans_cpu.cc 
CUDA_SOURCES = kmeans_gpu.cu
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

EXECNAME = kmeans

all:
	$(CPUCC) -c $(CXXFLAGS) $(CC_CXXFLAGS) $(CC_SOURCES)
	$(GPUCC) -c $(CUDA_SOURCES) $(CXXFLAGS) $(CUDA_CXXFLAGS)
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CC_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) $(CC_LIBS)
	
clean:
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend


#Regles automatiques pour les objets
#%.o:  %.cc
#	$(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#	$(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

