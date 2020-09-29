#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#ifdef NAN
/* NAN is supported */
#endif
#include <unistd.h>
#include <stdbool.h>
#include "knn.h"
#include "vptree.h"
#include "stack.h"

void vp_knn(double *distances, int *idx ,vptree *node,double *point,int d ,int k,int n);
double calculate_dist(double *a, double *b , int d);
void update(double *distances,int *idx , int id , double dist, double *radius ,int k );
void swap_int(int* a, int* b);
void swap_double(double* a, double* b);
int partition_q(double *arr, int *idx ,int low, int high);
void quickSort(double *arr, int *idx, int low, int high);
int vptree_stack_search(double *distances, int *idx ,vptree *node,double *point,double *radius,int d ,int k,int n);



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


knnresult kNN(double* X,double* Y,int n,int m,int d,int k)
{
	struct timeval startwtime,endwtime;
	double seq_time;
	
	knnresult result;
	result = init_knnresult(m,k);

	vptree *root=buildvp(X,n,d);

	gettimeofday(&startwtime,NULL);

	for (int i = 0; i < m; i++)
		vp_knn(result.ndist+i*k , result.nidx+i*k,root,Y+i*d,d,k,n);

	gettimeofday(&endwtime,NULL);
	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
   	printf("Serial Knn time = %f \n", seq_time);

	return result;
}

void vp_knn(double *distances, int *idx ,vptree *node,double *point,int d ,int k,int n)
{
	double *radius = (double *)malloc(sizeof(double));
	*radius = INFINITY;
	int res = vptree_stack_search(distances,idx,node,point,radius, d,k,n);
}

int vptree_stack_search(double *distances, int *idx ,vptree *root,double *point,double *radius,int d ,int k,int n)
{
	//Maybe it will needed more than 1000 stack size for large dataset.
	Stack* stack = createStack(1000,d); 
	push(stack,root,d);
	int count=0;
	
	vptree *node;
	node = (vptree*)malloc(sizeof(vptree));
	node->vantage_point = (double*)malloc(sizeof(double)*d);

	while(!isEmpty(stack))
	{
		if(!pop(stack,node,d))
			printf("Error:Stack is is empty\n");
		double dist;
		dist = calculate_dist(point,node->vantage_point,d);
		//store point and distance
		if (dist < *radius)
			update(distances,idx,node->idx,dist,radius,k);
		//check inside
		if(node->inner!=NULL){
			if (dist <= node->median_value + *radius)
				push(stack , node->inner , d);			
		}
		if (node->outer!=NULL)
		{
			//check outside
			if (dist > node->median_value - *radius)
				push(stack , node->outer , d);
		}
	}
	return 0;
}

double calculate_dist(double *a, double *b , int d)
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

void update(double *distances,int *idx , int id , double dist, double *radius ,int k)
{
	distances[k-1] = dist;
	idx[k-1] = id;
	quickSort(distances,idx,0,k-1);
	*radius = distances[k-1]; 
}
  
// A utility function to swap two elements 
void swap_int(int* a, int* b) 
{ 
    int t = *a; 
    *a = *b; 
    *b = t; 
}

void swap_double(double* a, double* b) 
{ 
    double t = *a; 
    *a = *b; 
    *b = t; 
}  
  
int partition_q (double *arr, int *idx ,int low, int high) 
{ 
    double pivot = arr[high];    // pivot 
    int i = (low - 1);  // Index of smaller element 
  
    for (int j = low; j <= high- 1; j++) 
    { 
        // If current element is smaller than the pivot 
        if (arr[j] < pivot) 
        { 
            i++;    // increment index of smaller element 
            swap_double(&arr[i], &arr[j]);
            swap_int(&idx[i], &idx[j]); 
        } 
    } 
    swap_double(&arr[i + 1], &arr[high]); 
    swap_int(&idx[i + 1], &idx[high]); 
    return (i + 1); 
} 
  
void quickSort(double *arr, int *idx, int low, int high) 
{ 
    if (low < high) 
    { 
        /* pi is partition_qing index, arr[p] is now 
           at right place */
        int pi = partition_q(arr, idx ,low, high); 
  
        // Separately sort elements before 
        // partition_q and after partition_q 
        quickSort(arr, idx,low, pi - 1); 
        quickSort(arr, idx ,pi + 1, high); 
    } 
} 
