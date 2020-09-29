# Cuda_Knn_Vp_Tree
A naive implementation of a vantage point tree using CUDA and an additional knn search over it using CUDA too.

**Parallel implementation execution:**

Run the makefile in folder **Cuda_Knn_Vp_Tree/KNN_VP_TREE/Knn_Vp_Tree/**<br/>
Then run the makefile in folder **Cuda_Knn_Vp_Tree/KNN_VP_TREE/**<br/>
You can test the tree construction running the executable **./test_vp_tree** with 2 arguments(1: Number of elements 2: Dimensions)<br/>
You can test the knn search running the executable **./test_knn** with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)<br/>

**Sequential implementation:**

Run the makefile in folder **Cuda_Knn_Vp_Tree/KNN_VP_TREE/sequential_implementation/serial_knn_vp_tree/**<br/>
Then run the makefile in folder **Cuda_Knn_Vp_Tree/KNN_VP_TREE/sequential_implementation/**<br/>
You can test the recursive knn search running the executable **./test_knn_recursive** with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)<br/>
You can test the knn search using the stack running the executable **./test_knn_stack** with 3 arguments(1: Number of elements 2: Dimensions 3:Number of nearest neighbors)<br/>
