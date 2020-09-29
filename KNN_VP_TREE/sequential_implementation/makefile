# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc

all: test_knn_stack test_knn_recursive

test_knn_stack:
	cd serial_knn_vp_tree; make lib; cd ..
	cd serial_knn_vp_tree; cp lib/*.a inc/*.h ../; cd ..
	$(CC) tester.c knn_stack.a vptree.a -lm -o $@
	#"											  "
	#"											  "
	#"											  "
	#"********************************************"
	#"	Knn search over a Vantage         
	#"	Point Tree using a stack		  
	#"	Arguments: 	N=3000	 D=10  K=25	      
	#"	Run ./test_knn_stack with your	      
	#"	arguments				  
	#"********************************************"
	./test_knn_stack 3000 10 25
	#
	#
	#

test_knn_recursive:
	cd serial_knn_vp_tree; make lib; cd ..
	cd serial_knn_vp_tree; cp lib/*.a inc/*.h ../; cd ..
	$(CC) tester.c knn_recursive.a vptree.a -lm -o $@
	#"											  "
	#"											  "
	#"											  "
	#"********************************************"
	#"	Recursive Knn search over a Vantage         
	#"	Point Tree 				  
	#"	Arguments: 	N=3000	 D=10  K=25	      
	#"	Run ./test_knn_recursive with your	      
	#"	arguments						  
	#"********************************************"
	./test_knn_recursive 3000 10 25
	#
	#
	#