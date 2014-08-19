#  Makefile.sh
#  CUDADSMC
#
#  Created by Christopher Watkins on 1/08/2014.
#  Copyright (c) 2014 WIJ. All rights reserved.

# Common binaries
GCC   ?= g++
CLANG ?= /usr/bin/clang

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin) #If building on an OSX system
	#	CUDA_PATH= /usr/local/cuda
	CUDA_PATH= /Developer/NVIDIA/CUDA-6.5
	CUDA_INC = -I/Developer/NVIDIA/CUDA-6.5/include
	CUDA_LIB = -L/Developer/NVIDIA/CUDA-6.5/lib
	NVCC     = $(CUDA_PATH)/bin/nvcc -ccbin $(CLANG)
	#    NVCC = nvcc
else                     #If building on a Linux system
	CUDA_INC = -I/usr/local/cuda-5.5/include
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)  #If 64 bit
		CUDA_LIB = -L/usr/local/cuda-5.5/lib64
	else
		CUDA_LIB = -L/usr/local/cuda-5.5/lib
	endif
	NVCC = nvcc
endif

BUILDDIR =
OBJDIR   = $(BUILDDIR)obj/
SRCDIR   = src/

INCLUDE = -I include/ -I /usr/local/hdf5/include
LIB     = -L /usr/local/hdf5/lib

all: build

build: CUDADSMC

CUDADSMC: main.o setUp.o moveAtoms.o  cudaHelpers.o hdf5Helpers.o
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 -o $(addprefix $(BUILDDIR), $@ ) $(INCLUDE) $(addprefix $(OBJDIR), $^ ) $(LIB) $(CUDA_LIB) -lcurand -lcublas -lhdf5 -lhdf5_hl

main.o: $(addprefix $(SRCDIR), main.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

setUp.o: $(addprefix $(SRCDIR), setUp.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

moveAtoms.o: $(addprefix $(SRCDIR), moveAtoms.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

#collisions.o: $(addprefix $(SRCDIR), collisions.cu )
#	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

cudaHelpers.o: $(addprefix $(SRCDIR), cudaHelpers.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

hdf5Helpers.o: $(addprefix $(SRCDIR), hdf5Helpers.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

buildTest: test

test: mainTest.o setUp.o statTests.o cudaHelpers.o
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 -o $(addprefix $(BUILDDIR), $@ ) $(INCLUDE) $(addprefix $(OBJDIR), $^ ) $(LIB) $(CUDA_LIB) -lcurand -lcublas -lhdf5

mainTest.o: $(addprefix $(SRCDIR), mainTest.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

statTests.o: $(addprefix $(SRCDIR), statTests.cu )
	$(NVCC) -m 64 -pg -arch=compute_30 -code=sm_30,compute_30 $(INCLUDE) -o $(addprefix $(OBJDIR), $@ ) -c $?

run:
	{ time ./$(BUILDDIR)CUDADSMC ; } | tee output.txt
#	/usr/local/hdf5/bin/h5import "./src/main.cu" -c "./hdfimportconfig/h5import-main.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "./src/evolveSpin.cu" -c "./hdfimportconfig/h5import-evolveSpin.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "Makefile" -c "./hdfimportconfig/h5import-makefile.conf" -o "mmDataMCWF.h5"
	echo "" >> output.txt
	echo "----------------------------------------" >> output.txt
	echo "" >> output.txt

	tput bel
clean:
	rm -rf $(OBJDIR)*.o $(BUILDDIR)CUDADSMC