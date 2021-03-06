#ifndef VPTREE_H
#define VPTREE_H
// type definition of vptree
// ========== LIST OF ACCESSORS
//! Build vantage-point tree given input dataset X
/*!
\param X Input data points, stored as [n-by-d] array
\param n Number of data points (rows of X)
\param d Number of dimensions (columns of X)
\return The vantage-point tree
*/
typedef struct vptree vptree;

struct vptree
{
	int idx;
	double *vantage_point;
	double median_value;
	struct vptree *outer;
	struct vptree *inner;
};

typedef struct tree_array tree_array;

struct tree_array
{
	double median;
	int idx;
	bool valid;
};


void buildvp_cuda(double *X, tree_array *cuda_tree ,int n, int d);


vptree * buildvp(double *X, int n, int d);
//! Return vantage-point subtree with points inside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getInner(struct vptree * T);
//! Return vantage-point subtree with points outside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getOuter(struct vptree * T);
//! Return median of distances to vantage point
/*!
\param node A vantage-point tree
\return The median distance
*/
double getMD(struct vptree * T);
//! Return the coordinates of the vantage point
/*!
\param node A vantage-point tree
\return The coordinates [d-dimensional vector]
*/
double * getVP(struct vptree * T);
//! Return the index of the vantage point
/*!
\param node A vantage-point tree
\return The index to the input vector of data points
*/
int getIDX(struct vptree * T);

vptree* newnode(double *point ,int index ,int n , int d);


#endif
