# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = nvcc

all: test_vp_tree test_knn

test_vp_tree:
	cd Knn_Vp_Tree; make lib; cd ..
	cd Knn_Vp_Tree; cp lib/*.a inc/*.h ../; cd ..
	$(CC) vp_tree_tester.cu vptree.a -o $@
	#"											  "
	#"											  "
	#"											  "
	#"********************************************"
	#"	Vantage_point Tree Construction	  
	#"	Arguments: 	N=3000	 D=10		  
	#"	Run ./test_vp_tree with your	  
	#"	arguments 				  
	#"********************************************"
	./test_vp_tree 3000 10
	#
	#
	#

test_knn:
	cd Knn_Vp_Tree; make lib; cd ..
	cd Knn_Vp_Tree; cp lib/*.a inc/*.h ../; cd ..
	$(CC) knn_tester.cu knn.a vptree.a -o $@
	#"											  "
	#"											  "
	#"											  "
	#"********************************************"
	#"	Knn search over a Vantage         
	#"	Point Tree 				  
	#"	Arguments: 	N=100	 D=15  K=5	      
	#"	Run ./test_knn with your	      
	#"	arguments 				  
	#"********************************************"
	./test_knn 100 15 5
	#
	#
	#