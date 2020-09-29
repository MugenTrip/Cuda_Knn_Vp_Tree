#include <time.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "vptree.h"

int *idx_host;
__device__ tree_array *vp_tree;

vptree* newnode(double *point ,int index ,double median , int d)
{
	vptree *node = (vptree*) malloc(sizeof(vptree));
	node->idx = index;
	node->vantage_point = (double *) malloc(sizeof(double)*d);
	for (int i = 0; i < d; i++)
	{
		node->vantage_point[i] = point[index*d+i];
	}
	node->median_value = median;
	node->outer = node->inner = NULL ;
	return node;
}

vptree * getInner(vptree * T)
{
	return T->inner;
}

vptree * getOuter(vptree * T)
{
	return T->outer;
}

double getMD(vptree * T)
{
	return T->median_value;
}

double * getVP(vptree * T)
{
	return T->vantage_point;
}

int getIDX(vptree * T)
{
	return T->idx;
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int* set_start_end(int* milestone,int level);
int* set_sorting_priority(int *milestone,int itterations,int n);
__global__ void block_calculate_distances(double *shared_points_cuda,double* distances_cuda, int *idx_cuda , int* cuda_milestone,int d ,int power_of_level);
__global__ void set_median(double* distances_cuda,int* cuda_milestone,int power_of_level);
__device__ void calculate_distance_gpu(double *shared_points_cuda,double* distances_cuda,int *idx_cuda ,double *vantage_point,int start , int  end , int d );
//__global__ void update(double *distances_cuda,int *cuda_milestone,int *temp,int power_of_level);
vptree *build_tree(tree_array *host,vptree *node ,double *data,int index,int d, int n);

vptree *buildvp(double *X,int n, int d)
{
	struct timeval startwtime,endwtime;
	double seq_time;

	tree_array *dynamic_vp_tree;
	int itterations = ((int)log2f(n))+1;
	int num = pow(2,itterations);
	checkCuda( cudaMalloc( (void **) &dynamic_vp_tree,sizeof(tree_array)* num));

	gettimeofday(&startwtime,NULL);

	buildvp_cuda(X, dynamic_vp_tree ,n, d);

	gettimeofday(&endwtime,NULL);
	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
	printf("GPU TREE CONSTRUCTION:%lf \n",seq_time );
	
	tree_array *host_tree = (tree_array*) malloc(sizeof(tree_array)*num);
	checkCuda(cudaMemcpy(host_tree,dynamic_vp_tree,sizeof(tree_array)*num,cudaMemcpyDeviceToHost));
	vptree *root;
	root = build_tree(host_tree,root ,X,0,d,num);
 
	free(host_tree);
	cudaFree(dynamic_vp_tree);
	return root;
}

/*The conctruction of the vantage point tree in the gpu. It stores it in tree_array stucture.
It constructs the tree level by level, calculating the parameters of each node in parallel.*/
void buildvp_cuda(double *X, tree_array *cuda_tree ,int n, int d)
{	
	struct timeval startwtime,endwtime;
	double seq_time;
	
 	int itterations = ((int)log2f(n))+1;
 	int num = pow(2,itterations);
 	printf("\n");
 	printf("N=%d and Itterations:%d \n", n , itterations );

	checkCuda(cudaMallocHost((void **) &idx_host , n*sizeof(int)));
  

	/*********** Data initialization*******************/
   	printf("\n");

	gettimeofday(&startwtime,NULL);
	//Seting IDs
	for (int i=0;i<n;i++){
    	idx_host[i]=i;
	}
	
	//Cuda array to store the points
	double *dynamic_shared_points_cuda;
	checkCuda(cudaMalloc((void **) &dynamic_shared_points_cuda , n*d*sizeof(double)));
    checkCuda(cudaMemcpy(dynamic_shared_points_cuda,X,sizeof(double)*n*d,cudaMemcpyHostToDevice));
	
	//Cuda array to store the distances
	int *dynamic_idx_cuda;
	checkCuda(cudaMalloc((void **) &dynamic_idx_cuda , n*sizeof(int)));
    checkCuda(cudaMemcpy(dynamic_idx_cuda,idx_host,sizeof(int)*n,cudaMemcpyHostToDevice));

    //Cuda array to store the distances
	double *dynamic_dist_cuda;
	checkCuda(cudaMalloc((void **) &dynamic_dist_cuda,sizeof(double)*n));

	//Connect the pointer with the global variable
	checkCuda(cudaMemcpyToSymbol(vp_tree, &cuda_tree, sizeof(tree_array*) ,0,cudaMemcpyHostToDevice));
	
	//Milestone array stores the start and the end index of each segment in which we cut our points in every level
	int milestone_size = (int) 4*pow(2,(itterations));
	int *milestone = (int *)malloc(sizeof(int)*milestone_size);
	*milestone=0;
	*(milestone+1)=n; 
	
	//Cuda variable for milestone
	int *cuda_milestone;
	checkCuda(cudaMalloc((void **) &cuda_milestone , milestone_size*sizeof(int)));
	
	int *temp_milestone;
	checkCuda(cudaMalloc((void **) &temp_milestone , milestone_size*sizeof(int)));

	//Group array stores the group of each point.With this way we can sort each group seperatly.
	//I got this idea based on the following link.
	//https://stackoverflow.com/questions/28150098/how-to-use-thrust-to-sort-the-rows-of-a-matrix?fbclid=IwAR2wAiz9aaGKAMZ5gnAuJKa81dP7qn-CsnA-w932911qj2cBClTh-88CND8
	int *group = (int*) malloc(sizeof(int)*n);
	
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1024*1024*1024));

	gettimeofday(&endwtime,NULL);
	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
   	printf("Setting up variables = %f \n", seq_time);
	
	double distances_time =0.0;  	//time to measure the total distances calculation
	double median_time =0.0; 	 	//time to measure the total median selection
	printf("\n");
	printf("Starting....\n");
	printf("\n");
	//Main procedure: Build the tree level by level. In each level calculate the parameters of each node in parallel.
	for (int level = 0; level < itterations; level++)
	{
		int it_num = (int) pow(2,level); 	//max nodes in each level
		
		//Get time to measure distances calculation
		gettimeofday(&startwtime,NULL);
		
		//Calculating the start and the end of each segment based on the previous ones
		milestone = set_start_end(milestone,level);
		checkCuda(cudaMemcpy(cuda_milestone,milestone,sizeof(int)*milestone_size,cudaMemcpyHostToDevice));
		//checkCuda(cudaMemcpy(milestone,cuda_milestone,sizeof(int)*milestone_size,cudaMemcpyDeviceToHost));

		/*Caclculate distances of every node in parallel*/
		block_calculate_distances<<<512,1024,sizeof(double)*d>>>( dynamic_shared_points_cuda,dynamic_dist_cuda, dynamic_idx_cuda ,cuda_milestone,d , it_num);
		checkCuda(cudaGetLastError());
		checkCuda(cudaDeviceSynchronize());

		//Measure the time elapsed and accumulate it in the total time
		gettimeofday(&endwtime,NULL);
		seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
		distances_time +=seq_time;

		//Get time to measure medians selection
		gettimeofday(&startwtime,NULL);

		//Seperate each group of points
		group = set_sorting_priority(milestone,pow(2,level),n);

		/********Vectors initiation***********/
		thrust::host_vector<int> h_segments(group,group+n);		//Host groups
  		thrust::device_vector<int> d_segments = h_segments;		//Device groups

  		thrust::device_vector<double> d_result(dynamic_dist_cuda,dynamic_dist_cuda+n);		//Device vector of distances
  		thrust::device_vector<int> d_result_idx(dynamic_idx_cuda,dynamic_idx_cuda+n);		//Device vector of IDs

  		//Sorting distances and update the tuple = {groups , ids} and the resort the groups and update the tuple = {distances , ids}
  		thrust::stable_sort_by_key(d_result.begin(), d_result.end(), thrust::make_zip_iterator(thrust::make_tuple(d_segments.begin(),d_result_idx.begin())) );
 	 	cudaDeviceSynchronize();
 	 	thrust::stable_sort_by_key(d_segments.begin(), d_segments.end(), thrust::make_zip_iterator(thrust::make_tuple(d_result.begin(),d_result_idx.begin()))  );
  		cudaDeviceSynchronize();

  		//Save back the distances and the ids
  		thrust::copy(d_result_idx.begin(),d_result_idx.end(),dynamic_idx_cuda);
  		thrust::copy(d_result.begin(),d_result.end(),dynamic_dist_cuda);
		cudaDeviceSynchronize();
		
		//Setting the median value of each node in parallel.
		set_median<<<1,1024>>>(dynamic_dist_cuda,cuda_milestone,it_num);
		cudaDeviceSynchronize();


		gettimeofday(&endwtime,NULL);
		seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
		median_time +=seq_time;
	}

	printf("Total distance calculation time: %lf \n", distances_time);
	printf("Picking(sorting) and setting median: %lf \n", median_time);

	free(milestone);
	cudaFree(dynamic_dist_cuda);
	cudaFree(dynamic_idx_cuda);
	cudaFree(dynamic_shared_points_cuda);
	cudaFree(cuda_milestone);
}

//Move the true from device to host and transform it to vpree struct
vptree *build_tree(tree_array *host,vptree *node ,double *data,int index,int d, int n)
{
	if (index>n-1)
		return NULL;
	else{
		if (host[index].valid==false)
			return NULL;
		else{
			if (index%2==0)
			{
				if (host[index].idx==host[index-1].idx || host[index].idx==host[(index-2)/2].idx)
				{
					host[index].valid=false;
					return NULL;
				}
			}
			else
			{
				if (host[index].idx==host[(index-1)/2].idx)
				{
					host[index].valid=false;
					return NULL;
				}
			}
			node = newnode(data,host[index].idx, host[index].median , d);
			node->outer = build_tree(host, node->outer, data,2*index+2,d,  n);
			node->inner = build_tree(host  ,node->inner , data,2*index+1,d ,   n);
			return node;
		}
	}
}

//Find the start and the end of each segment
int* set_start_end(int* milestone,int level)
{
	int h = pow(2,level);
	int *temp=(int*)malloc(sizeof(int)*2*h);
	int start,end,n;
	if(level==0)
	{
		temp[0]=0;
		temp[1]=milestone[1]-1;
	}
	else
	{
		for (int i = 0; i < pow(2,level-1); i++)
		{
			start = milestone[2*i];
			end = milestone[2*i+1];
			n = end-start+1;
			//printf("start: %d end: %d n: %d\n", start , end , n );
		
			if(end-start==0)
			{
				temp[4*i] = start;
				temp[4*i+1] = end;
				temp[4*i+2] = start;
				temp[4*i+3] = end;
			}
			//In this occasion we have two points in the array. One of them will
			//be chosen as a vantage point,so it is going to be created a complete binary
			//which means there are gonna be nodes with only a right child node
			else if(end-start==1)
			{
				temp[4*i] = start;
				temp[4*i+1] = end-1;
				temp[4*i+2] = start;
				temp[4*i+3] = end-1;	
			}
			//In this a occasion we are having three points and the tree gonna be splitted in a perfect binary tree 
			else if(end-start==2)
			{
				temp[4*i] = start;
				temp[4*i+1] = end-2;
				temp[4*i+2] = start+1;
				temp[4*i+3] = end-1;
			}
			//General occasion
			else
			{
				temp[4*i] = start;
				if ((n-1)%2==0)
				{
					temp[4*i+1] = start+(n-1)/2-1;
					temp[4*i+2] = start+(n-1)/2;
				}
				else
				{
					temp[4*i+1] = start+(n-1)/2;
					temp[4*i+2] = start+(n-1)/2+1;
				}
				temp[4*i+3] = end-1;
 			}	
		}
	}

	for (int j = 0; j < 2*pow(2,level); j++)
	{
		milestone[j]=temp[j];
	}
	free(temp);
	return milestone;
}

//Set an ascending number to each diffirent segment
int* set_sorting_priority(int *milestone,int itterations,int n)
{
	int flag=0;
	int *array = (int*)malloc(sizeof(int)*n);
	for (int i = 0; i < itterations; i++)
	{
		int start = milestone[2*i];
		int end = milestone[2*i+1];
		if (i==0)
		{
			for (int j = start; j <= end; j++)
			{
				array[j] = flag;
				if (start==end)
				{
					flag++;
				}
				else if(j==end)
				{
					flag++;
					array[j]=flag;
					flag++;
				}
			}
		}
		else
		{
			int milestone_end =  milestone[2*i-1];
			if (start-milestone_end>1)
			{
				for (int j = milestone_end; j < start; j++)
				{
					array[j] = flag;
					flag++;
				}
			}
			for (int j = start; j <= end; j++)
			{
				array[j] = flag;
				if (start==end)
				{
					flag++;
				}
				else if(j==end)
				{
					flag++;
					array[j]=flag;
					flag++;
				}
			}
		}
		if(i==itterations-1)
		{
			for (int j = end; j < n; j++)
			{
				array[j]=flag;
				flag++;	
			}
		}
	}	
	return array;
}

//Function that set the parameters and calculate the distances for each node in parallel.
__global__ void block_calculate_distances(double* shared_points_cuda,double* distances_cuda, int *idx_cuda ,int* cuda_milestone,int d , int power_of_level)
{
	__shared__ int start;
	__shared__ int end;
	__shared__ int index;
	__shared__ int dim;
	//extern __shared__ double vp[];
	
	for (int i = 0; i < power_of_level/gridDim.x+1; ++i)
	{	
		int id = blockIdx.x+i*gridDim.x;
		if (id<power_of_level)
		{
			if (threadIdx.x==0)
			{
				dim = d;
				start =  cuda_milestone[2*id];
				end = cuda_milestone[2*id+1];
				index=power_of_level-1+id;
				/*for (int i = 0; i < dim; i++)
				{
					vp[i] = shared_points_cuda[idx_cuda[end]*dim+i];
				}*/
				vp_tree[index].idx = idx_cuda[end];
				vp_tree[index].array_idx=index;
				vp_tree[index].valid=true;
				if(index>0){
					if (index%2==0){
						if (vp_tree[index].idx==vp_tree[index-1].idx || vp_tree[index].idx==vp_tree[(index-2)/2].idx)
							vp_tree[index].valid=false;
					}
					else{
						if (vp_tree[index].idx==vp_tree[(index-1)/2].idx)
							vp_tree[index].valid=false;
					}
				}
			}
			__syncthreads();	

			calculate_distance_gpu(shared_points_cuda,distances_cuda,idx_cuda,shared_points_cuda+idx_cuda[end]*dim,start,end,dim);
			//__syncthreads();
		}
	}
}

//Device Function to calculate the distances
__device__ void calculate_distance_gpu(double* shared_points_cuda,double* distances_cuda, int *idx_cuda ,double *vantage_point,int start , int  end , int d )
{
	int n=end-start+1;
	for(int i = 0; i < n/blockDim.x+1;i++){
		int index = (start + threadIdx.x) + i*blockDim.x;
		double sum = 0;
		if (index>=start && index<=end)
		{
			for(int j=0 ; j<d ;j++){
				sum+= pow(shared_points_cuda[idx_cuda[index]*d+j] - vantage_point[j],2);
			}
			distances_cuda[index] = sqrt(sum);
		}
	}
}

//Setting the median value to each node in parallel.
__global__ void set_median(double* distances_cuda,int* cuda_milestone , int power_of_level)
{
	for (int i = 0; i < power_of_level/blockDim.x+1; i++)
	{
		int idx;
		idx=threadIdx.x+i*blockDim.x;
		if (idx<power_of_level){
			
			int index;
			index=power_of_level-1+idx;
			
			int start,end;
			start =  cuda_milestone[2*idx];
			end = cuda_milestone[2*idx+1];

			int median_index;
			if (end-start>0)
			{
				if(end-start>1)
				{
					if (end-start==2)
						median_index = start;
					else if((end-start)%2==0)
						median_index = start+(end-start)/2-1;
					else if((end-start)%2!=0)
						median_index = start+(end-start)/2;
				}	
				else if ((end-start)==1)
					median_index=start;

				vp_tree[index].median = distances_cuda[median_index];
			}
		}
	}
	__syncthreads();
}

/*Function to fix the problem of the same points.Unfortunatly it can't fix the problem of the same distance of two diffirent points.That's why it's not included*/
/*__global__ void update(double *distances_cuda,int *cuda_milestone,int *temp,int power_of_level)
{

	for (int i = 0; i < power_of_level/blockDim.x+1; i++)
	{
		int index = threadIdx.x+i*blockDim.x;
		if (index<power_of_level)
		{
			int start,end;
			start =  cuda_milestone[2*i];
			end = cuda_milestone[2*i+1];

			int median_index;
			if (end-start>0)
			{
			if(end-start>1)
				{
					if (end-start==2)
						median_index = start;
					else if((end-start)%2==0)
						median_index = start+(end-start)/2-1;
					else if((end-start)%2!=0)
						median_index = start+(end-start)/2;
				}	
				else if ((end-start)==1)
					median_index=start;
			}

			int count=median_index;

			if(end-start==0)
			{
				temp[4*index+0] = start;
				temp[4*index+1] = end;
				temp[4*index+2] = start;
				temp[4*index+3] = end;
			}
			else if(end-start==1)
			{
				temp[4*index+0] = start;
				temp[4*index+1] = end-1;
				temp[4*index+2] = start;
				temp[4*index+3] = end-1;	
			}
			else{
				while(distances_cuda[median_index]==distances_cuda[count+1])
				{
					count++;
				}
				if(end-start==2)
				{
					if (count==median_index )
					{
						temp[4*index+0] = start;
						temp[4*index+1] = end-2;
						temp[4*index+2] = start+1;
						temp[4*index+3] = end-1;
					}
					else
					{
						temp[4*index+0] = start;
						temp[4*index+1] = count;
						temp[4*index+2] = end;
						temp[4*index+3] = end;
					}
				}
				else{	
					temp[4*index+0] = start;
					temp[4*index+1] = count;
					temp[4*index+2] = count +1;
					temp[4*index+3] = end -1;
				}
			}
		}
	}

	__syncthreads();

	for (int i = 0; i < power_of_level/(blockDim.x)+1; i++)
	{
		int index = threadIdx.x + i*blockDim.x;
		if (index<power_of_level)
		{
			cuda_milestone[4*index]=temp[4*index+0];
			cuda_milestone[4*index+1]=temp[4*index+1];
			cuda_milestone[4*index+2]=temp[4*index+2];
			cuda_milestone[4*index+3]=temp[4*index+3];
		}	
	}
}*/
