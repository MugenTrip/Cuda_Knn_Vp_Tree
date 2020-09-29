#ifndef KNN_H
#define KNN_H


// Definition of the kNN result 
typedef struct knnresult
{
	int* nidx; //!< Indices (0-based) of nearest neighbors [m-by-k]
	double* ndist;	//!< Distance of nearest neighbors[m-by-k]
	int m;	//!< Number of query points[scalar]
	int k;	//!< Number of nearest neighbors[scalar]
} knnresult;

knnresult kNN(double* X,double* Y,int n,int m,int d,int k);

knnresult init_knnresult(int m_arg , int k_arg );


#endif
