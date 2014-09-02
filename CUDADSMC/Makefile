#  Makefile.sh
#  CUDADSMC
#
#  Created by Christopher Watkins on 1/08/2014.
#  Copyright (c) 2014 WIJ. All rights reserved.

# Colour coding the makfile :-)
NO_COLOR=\033[0m
OK_COLOR=\033[0;33m
ERROR_COLOR=\x1b[31;01m
WARN_COLOR=\x1b[33;01m

OK_STRING=$(OK_COLOR)[OK]$(NO_COLOR)
ERROR_STRING=$(ERROR_COLOR)[ERRORS]$(NO_COLOR)
WARN_STRING=$(WARN_COLOR)[WARNINGS]$(NO_COLOR)

# Common binaries
GCC   ?= g++
CLANG ?= /usr/bin/clang

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin) #If building on an OSX system
	CUDA_PATH= /Developer/NVIDIA/CUDA-6.5
	CUDA_INC = -I/Developer/NVIDIA/CUDA-6.5/include
	CUDA_LIB = -L/Developer/NVIDIA/CUDA-6.5/lib
	NVCC     = $(CUDA_PATH)/bin/nvcc -ccbin $(CLANG)
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

NVCCFLAGS = -m 64 -arch=compute_30 -code=sm_30,compute_30

BUILDDIR = bin/
OBJDIR   = $(BUILDDIR)obj/
SRCDIR   = src/

EXEC = $(BUILDDIR)CUDADSMC
TESTMOTION = $(BUILDDIR)testMotion

INCLUDE = -I include/ -I /usr/local/hdf5/include
LIB     = -L /usr/local/hdf5/lib

all: $(EXEC)

debug: NVCCFLAGS += -g -G
#debug: clean
debug: $(EXEC)

profile: NVCCFLAGS += -pg
profile: $(EXEC)

$(EXEC): $(addprefix $(OBJDIR), main.o setUp.o moveAtoms.o collisions.o cudaHelpers.o hdf5Helpers.o )
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Linker'
	$(NVCC) -o $@ $(INCLUDE) $^ $(LIB) $(CUDA_LIB) -lcurand -lcublas -lhdf5 -lhdf5_hl
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '

$(OBJDIR)%.o: $(SRCDIR)%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $?
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '

testMotion: $(TESTMOTION)

$(TESTMOTION): $(addprefix $(OBJDIR), moveTest.o setUp.o moveAtoms.o collisions.o cudaHelpers.o hdf5Helpers.o )
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Linker'
	$(NVCC) -o $@ $(INCLUDE) $^ $(LIB) $(CUDA_LIB) -lcurand -lcublas -lhdf5 -lhdf5_hl
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '
	@echo 'Testing...'
	{ time ./$(BUILDDIR)testMotion ; } | tee motionTest.txt
	@echo "... Test complete $< $(OK_STRING)"
	@echo " " >> motionTest.txt
	@echo "----------------------------------------" >> motionTest.txt
	@echo " " >> motionTest.txt
	tput bel

$(OBJDIR)moveTest.o: $(SRCDIR)moveTest.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $?
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '

run:
	{ time ./$(BUILDDIR)CUDADSMC ; } | tee output.txt
#	/usr/local/hdf5/bin/h5import "./src/main.cu" -c "./hdfimportconfig/h5import-main.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "./src/evolveSpin.cu" -c "./hdfimportconfig/h5import-evolveSpin.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "Makefile" -c "./hdfimportconfig/h5import-makefile.conf" -o "mmDataMCWF.h5"
	@echo "" >> output.txt
	@echo "----------------------------------------" >> output.txt
	@echo "" >> output.txt
	tput bel

memcheck:
	{ time cuda-memcheck ./$(BUILDDIR)CUDADSMC ; } | tee output.txt
	@echo "" >> output.txt
	@echo "----------------------------------------" >> output.txt
	@echo "" >> output.txt
	tput bel

clean:
	rm -rf $(OBJDIR)*.o $(BUILDDIR)CUDADSMC