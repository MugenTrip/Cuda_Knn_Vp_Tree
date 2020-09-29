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
CC = gcc
CFLAGS = -I$(IDIR) -lm 

all: 
	$(CC) -c src/vptree.c -o lib/vptree.o $(CFLAGS)
	$(CC) -c src/knn_recursive.c -o lib/knn_recursive.o $(CFLAGS)
	$(CC) -c src/knn_stack.c -o lib/knn_stack.o $(CFLAGS)
	ar rcs lib/vptree.a lib/vptree.o 
	ar rcs lib/knn_recursive.a lib/knn_recursive.o
	ar rcs lib/knn_stack.a lib/knn_stack.o