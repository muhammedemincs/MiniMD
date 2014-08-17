# Multiple-machine Makefile

# Default Options

SVN = no
OMP = yes
CUDA = no

SHELL = /bin/sh

# Files

ifeq ($(SVN), yes)
	KOKKOSPATH = /usr/local/Trilinos/packages/kokkos/core/src
else
	KOKKOSPATH = $(shell cd kokkos; pwd;)
endif

HWLOCPATH = ./

PWD = `pwd`
SRC =	ljs.cpp input.cpp integrate.cpp atom.cpp force_lj.cpp neighbor.cpp \
	thermo.cpp comm.cpp timer.cpp output.cpp setup.cpp force_eam.cpp  
KOKKOS_SRC = $(shell cd $(KOKKOSPATH)/impl; ls *.cpp;)
KOKKOS_CPY = $(KOKKOSPATH)/impl/*.cpp

ifeq ($(OMP),yes)
KOKKOS_SRC += $(shell cd $(KOKKOSPATH)/OpenMP; ls *.cpp;)
KOKKOS_CPY += $(KOKKOSPATH)/OpenMP/*.cpp
else
KOKKOS_SRC += $(shell cd $(KOKKOSPATH)/Threads; ls *.cpp;)
KOKKOS_CPY += $(KOKKOSPATH)/Threads/*.cpp
endif

ifeq ($(CUDA),yes)
KOKKOS_CUDASRC += $(shell cd $(KOKKOSPATH)/Cuda; ls *.cu;)
KOKKOS_CPY += $(KOKKOSPATH)/Cuda/*.cu
endif

INC =	ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h \
	threadData.h variant.h types.h force_eam.h force_lj.h
	
  

# Definitions

ROOT =	miniMD
EXE =	$(ROOT)_$@
OBJ =	$(SRC:.cpp=.o) $(KOKKOS_SRC:.cpp=.o) $(KOKKOS_CUDASRC:.cu=.o)

# Help

help:
	@echo 'Type "make target {Options}" where target is one of:'
	@echo '      openmpi     (using OpenMPI)'
	@echo '      cuda        (using NVCC)'
	@echo '      intel       (using IntelMPI)'
	@echo '      titan       (using CrayCompiler for XK7)'
	@echo '      cray        (using CrayCompiler for CPUs)'
	@echo '      cray_intel  (using Cray Wrapper for Intel compiler for CPUs)'
	@echo 'Options (not all of which apply for each target):'
	@echo '      KOKKOSPATH=path (Path to Kokkos core source directory [set default in Makefile])'
	@echo '      CUDAPATH=path   (Path to Cuda installation [set default in Makefile.cuda])'
	@echo '      HWLOCPATH=path  (Optional path to HWLOC directory [set default in Makefile])'
	@echo '      SVN=yes         (needed when compiling using the mantevo repository)'
	@echo '      CUDA=yes        (use CUDA)'
	@echo '      CUDA_ARCH=sm_xx (use architectur target sm_xx [default sm_35])'
	@echo '      KNC=yes         (compile for XeonPhi)'
	@echo '      OMP=yes         (use OpenMP  [DEFAULT])'
	@echo '      SP=yes          (compile for single precision [32bit floats])'
	@echo '      DEBUG=yes       (compile debug mode)'
	@echo '      HWLOC=yes       (use Hardware Locality library for thread pinning)'
	@echo '      AVX=yes         (compile for avx  [DEFAULT])'
	@echo '      SIMD=yes        (use "pragma simd"  [DEFAULT])'
	@echo '      RED_PREC=yes    (use reduced precision math intrinsics)'
	@echo '      ANSI_ALIAS=yes  (compile with ansi-alias flag)'
	@echo '      GSUNROLL=yes    (compile with flags for XeonPhi which unroll gather/scatter operations [DEFAULT])'
	@echo '      LIBRT=yes       (use librt for timing [more precise])'
	@echo '      PAD=[3,4]       (pad data to 3 or 4 elements [default 3 = no padding])'
	@echo '(Good) Examples:'
	@echo '    Modern CPUs (Intel Sandy Bridge, AMD Interlagos or newer):'
	@echo '      make openmpi -j 32'
	@echo '    NVIDIA Kepler:'
	@echo '      make cuda -j 32 CUDA=yes'
	@echo '    Intel Xeon Phi:'
	@echo '      make intel -j 32 KNC=yes ANSI_ALIAS=yes SIMD=no'
	@echo '    Modern CPUs using pthreads:'	
	@echo '      make openmpi -j 32 OMP=no HWLOC=yes'
	
# Targets

openmpi:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@	
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)
#	@if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

cray:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@     
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)

cray_intel:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@     
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)

titan:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@	
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

cuda:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@echo $(OBJ)
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@	
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

intel:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@echo '' > Obj_$@/KokkosCore_config.h
	@cp -p $(SRC) $(INC) Obj_$@
	@cp -p $(KOKKOS_CPY) Obj_$@	
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" "KOKKOSPATH = $(KOKKOSPATH)" "HWLOCPATH = $(HWLOCPATH)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

# Clean
clean:
	rm -r Obj_*
	
clean_openmpi:
	rm -r Obj_openmpi

clean_cuda:
	rm -r Obj_cuda

clean_intel:
	rm -r Obj_intel

clean_titan:
	rm -r Obj_titan

clean_cray:
	rm -r Obj_cray
	
clean_cray_intel:
	rm -r Obj_cray_intel

# Test

scope=0
input=lj
halfneigh=0
path=""
test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path} 
  
