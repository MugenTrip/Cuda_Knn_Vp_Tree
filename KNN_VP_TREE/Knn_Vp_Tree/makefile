SHELL := /bin/bash

IDIR =./inc

ODIR=./lib

SDIR=./src

_OBJ = vptree.o knn.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_SRC = vptree.cu knn.cu 
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

_DEPS = vptree.h knn.h cuda_stack.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_FINAL = vptree.a knn.a
FINAL = $(patsubst %,$(ODIR)/%,$(_FINAL))

# define the C/C++ compiler to use,default here is clang 
#-lm -L$(NETLIB_LAPACK_ROOT)/lib64 -I$(NETLIB_LAPACK_ROOT)/include 
CC = nvcc
CFLAGS = -I$(IDIR) 

all: 
	$(CC) -c src/vptree.cu -o lib/vptree.o $(CFLAGS)
	$(CC) -c src/knn.cu -o lib/knn.o $(CFLAGS)
	ar rcs lib/vptree.a lib/vptree.o 
	ar rcs lib/knn.a lib/knn.o