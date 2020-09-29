/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include "knn.h"
#include "vptree.h"
#include "tester_helper.h"

using namespace std;

int main(int argc, char** argv)
{
  if(argc!=4){
    printf("\nNeed 3 argument. N:number of elements D:dimensions of elements K:number of nearest neighbors");
    exit(0);
  }

  int n=atoi(argv[1]);//data
  int d=atoi(argv[2]);//dimensions
  int k=atoi(argv[3]);//number of dimensions

  struct timeval startwtime, endwtime;
  double seq_time;

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
    corpus[i] = ( (double) (rand()) ) / 100000000;

  gettimeofday (&startwtime, NULL);
  knnresult knnres; 
  knnres = kNN(corpus, corpus, n, n, d, k );
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
  printf("Overall clock time = %f %d elements \n", seq_time , n);

  int isValidC = validateResult( knnres, corpus, corpus, n, n, d, k, COLMAJOR );

  int isValidR = validateResult( knnres, corpus, corpus, n, n, d, k, ROWMAJOR );
  
  printf("Tester validation: %s NEIGHBORS\n",
         STR_CORRECT_WRONG[isValidC||isValidR]);

  free( corpus );
  return 0;
  
}
