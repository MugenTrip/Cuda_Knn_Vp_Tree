# Cuda_Knn_Vp_Tree
A naive implementation of a vantage point tree using CUDA and an additional knn search over it using CUDA too.

Parallel implementation execution:

Run the makefile in folder Cuda_Knn_Vp_Tree/KNN_VP_TREE/Knn_Vp_Tree/
Then run the make file in folder Cuda_Knn_Vp_Tree/KNN_VP_TREE/
You can test the tree construction running the executable test_vp_tree with 2 arguments(1: Number of elements 2: Dimensions)
You can test the knn search running the executable test_knn with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)

Sequential implementation:

Run the makefile in folder Cuda_Knn_Vp_Tree/KNN_VP_TREE/sequential_implementation/serial_knn_vp_tree/
Then run the make file in folder Cuda_Knn_Vp_Tree/KNN_VP_TREE/sequential_implementation/
You can test the recursive knn search running the executable test_knn_recursive with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)
You can test the knn search using the stack running the executable test_knn_stack with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)
