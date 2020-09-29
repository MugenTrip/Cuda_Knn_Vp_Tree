#include <curand.h>
#include <cuda_runtime_api.h>

#include <time.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#include <algorithm>
#include "knn.h"
#include "vptree.h"
#include "cuda_stack.h"

using namespace std;

inline cudaError_t cudacheck(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
/* Function to sort an array using insertion sort*/
__device__ void insertionSort(double *arr, int *idx ,int n) 
{ 
    int i, idx_key, j;
    double key; 
    for (i = 1; i < n; i++) { 
        key = arr[i];
        idx_key = idx[i]; 
        j = i - 1; 
  
        /* Move elements of arr[0..i-1], that are 
          greater than key, to one position ahead 
          of their current position */
        while (j >= 0 && arr[j] > key) { 
            arr[j + 1] = arr[j];
            idx [j + 1] = idx[j]; 
            j = j - 1; 
        } 
        arr[j + 1] = key;
        idx[j+1] = idx_key; 
    } 
}
__global__ void vp_tree_search(double *cuda_share_points,double* cuda_distances, int *cuda_idx, tree_array *cuda_tree,int index ,int n , int m , int k , int d);
__device__ void update(double *cuda_distances,int *cuda_idx ,double dist , int id , double *radius  ,int k ,int block_idx);
__device__ double calculate_dist(double *a, double *b , int d);

knnresult kNN(double* X,double* Y,int n,int m,int d,int k)
{
	struct timeval startwtime,endwtime;
	double seq_time;

	knnresult result;
	result = init_knnresult(m,k);

	double *d_shared_points;
	cudacheck(cudaMalloc((void **) &d_shared_points , n*d*sizeof(double)));
    cudacheck(cudaMemcpy(d_shared_points,X,sizeof(double)*n*d,cudaMemcpyHostToDevice));

	int *d_idx;
	cudacheck(cudaMalloc((void **) &d_idx , m*k*sizeof(int)));

	double *d_dist;
	cudacheck(cudaMalloc((void **) &d_dist,sizeof(double)*k*m));
	cudacheck(cudaMemcpy(d_dist,result.ndist,sizeof(double)*k*m,cudaMemcpyHostToDevice));

	tree_array *dynamic_vp_tree;
	int itterations = ((int)log2f(n))+1;
	int num = pow(2,itterations);
	cudacheck( cudaMalloc( (void **) &dynamic_vp_tree,sizeof(tree_array)*num));

	printf("Building vantage point tree...\n");
	gettimeofday(&startwtime,NULL);

	buildvp_cuda(X, dynamic_vp_tree ,n, d);
	
	cudacheck(cudaPeekAtLastError());
	cudacheck(cudaDeviceSynchronize());
	
	gettimeofday(&endwtime,NULL);
	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
   	printf("Building tree time = %f \n", seq_time);
	


   	printf("Start searching the vantage point tree\n");

	gettimeofday(&startwtime,NULL);

	cudacheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));

	vp_tree_search<<<512,1>>>(d_shared_points,d_dist,d_idx, dynamic_vp_tree , 0 , num-1 , m , k , d);
	cudacheck(cudaPeekAtLastError());
	cudacheck(cudaDeviceSynchronize());

	gettimeofday(&endwtime,NULL);
	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
   	printf("GPU Knn time = %f \n", seq_time);
	
	cudacheck(cudaMemcpy(result.ndist, d_dist ,sizeof(double)*k*m,cudaMemcpyDeviceToHost));
	cudacheck(cudaMemcpy(result.nidx,d_idx,sizeof(int)*k*m,cudaMemcpyDeviceToHost));

	return result;
}

knnresult init_knnresult(int m_arg , int k_arg )
{
	knnresult result;
	result.nidx = (int *) malloc(sizeof(int)*m_arg*k_arg);
	result.ndist = (double *) malloc(sizeof(double)*m_arg*k_arg);
	for (int i = 0; i < m_arg*k_arg; i++)
	{
		result.ndist[i] = INFINITY;
	}
	result.m =m_arg;
	result.k=k_arg;
	return result;
}

__global__ void vp_tree_search(double *cuda_share_points,double* cuda_distances, int *cuda_idx, tree_array *cuda_tree,int index ,int n , int m , int k , int d)
{
	for (int i = 0; i < m/gridDim.x+1; i++)
	{
		int m_id = blockIdx.x+i*gridDim.x;
		if (m_id<m)
		{
			int tree_index = index;
			//Maybe it will needed more than 1000 stack size for large dataset.
			Stack* stack = createStack(1000);
			//printf("stack created in every block\n"); 
			push(stack,cuda_tree+index);
	
			tree_array *node;
			node = (tree_array*)malloc(sizeof(tree_array));

			double *radius = (double*)malloc(sizeof(double));
			*radius = INFINITY;

			while(!isEmpty(stack))
			{
				if(!pop(stack,node))
					printf("Error:Stack is is empty\n");
				if(node->valid){
					double dist;
					dist = calculate_dist(cuda_share_points+m_id*d,cuda_share_points+(node->idx)*d,d);
					//store point and distance
					if (dist < *radius)
						update(cuda_distances, cuda_idx ,dist, node->idx,radius,k,m_id);
					//check inside
					if (dist <= node->median + *radius){
						tree_index = 2*node->array_idx + 1;
						if(tree_index<n)
							push(stack ,cuda_tree+tree_index );			
					}
					//check outside
					if (dist > node->median - *radius){
						tree_index = (2*node->array_idx + 2); 
						if(tree_index<n)
							push(stack , cuda_tree+tree_index);
					}
				}
			}
			free(node);
			free(radius);
			deleteStack(stack);
		}
	}
}

__device__ void update(double *cuda_distances,int *cuda_idx ,double dist , int id , double *radius  ,int k ,int block_idx)
{
	double temp_rad;	
	cuda_distances[block_idx*k+(k-1)] = dist;
	cuda_idx[block_idx*k+(k-1)] = id;
	insertionSort(cuda_distances+block_idx*k,cuda_idx+block_idx*k,k);
	temp_rad = cuda_distances[block_idx*k+(k-1)];
	
	*radius = temp_rad;
}

__device__ double calculate_dist(double *a, double *b , int d)
{
	double dist;
	double sum = 0;
	for (int i = 0; i < d; i++)
	{
		sum+= pow(a[i]-b[i],2);
	}
	dist = sqrt(sum);
	return dist;
}