
/**********************************************************************
 * Quick Sort Using MPI
 *
 * Author : Panagiotis Stamatakopoulos
 *
 * Parallel Programming, Uppsala University
 **********************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "time.c"
#define root 0
#define len 100000000


/*Functions declarations*/
int partition(double *Array, int left, int right, int pivotIndex);
void quick_sort(double *Array, int left ,int right);
int MedianValue(double *Array, int left, int right);
int CheckIt(double *Array, int size);
void merge (double *A, int a, double *B,int start, int b, double *C);
double rand_normal(double mean, double stddev);


int main(int argc, char *argv[]) {

  unsigned long localLength;
  int  numProcs, rank , i,k, step, color, chunk, *rbuf, *displs;
  double start_time,finish_time,elapsed_time;
  double start_time2,finish_time2,elapsed_time2;
  double localPivot,globalPivot;
  double *BigArray,*LocalArray,*L[2],*RecvBuffer;

  MPI_Request Rrequest,Srequest;
  MPI_Status status;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int logP = log2((double)numProcs);

  /*Variables needed for topology and group creation*/
  int coords[2],pos[2],reorder=1,ndim=2;
  int **dims,periods[2]={0,0};
  int subrank[logP],myid[logP];

  dims = (int **)malloc(logP*sizeof(int *));
  for (i=0;i<logP;i++)
      dims[i]= (int *)malloc(2*sizeof(int));

  MPI_Comm proc_grid[logP],Communicator[logP];

  int newp = numProcs;

  /*Create communicators for each set of processes*/
 
  for (i=0;i<logP;i++){

    dims[i][0] = newp;
    dims[i][1] = 0;
    
    /* Create a virtual 2D-grid topology */
    MPI_Dims_create(numProcs,ndim,dims[i]);
    MPI_Cart_create(MPI_COMM_WORLD,ndim,dims[i],periods,reorder,&proc_grid[i]);
    MPI_Comm_rank(proc_grid[i],&myid[i]);

    color = floor((double)(rank / newp));/*Separate processes into groups*/
    
    /* Create a communicator for each row */
    MPI_Cart_coords(proc_grid[i],myid[i],ndim,coords);
    MPI_Comm_split(proc_grid[i],(int)color,coords[0],&Communicator[i]);
    MPI_Comm_rank(Communicator[i],&subrank[i]);

    newp /= 2;
    
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int choise = 0;

  if (rank == root){
    if (argc < 2)
    {
      printf("\nLength = 100.000.000 and data will be unified\n");
    }else {
      printf("\nLength = 100.000.000 and data will be normal\n");
      choise = 1;}
  }

  if (rank == root){

    /*Just a check*/
   printf("\nNumber of processes %d with log2 = %d\n",numProcs,logP); 

   BigArray = malloc(len*sizeof(MPI_DOUBLE));
    if (BigArray == NULL) {
        printf("ERROR-> BigArray malloc FAILED\n");
        exit(-1);
    }
   rbuf     = malloc(numProcs*sizeof(MPI_INT));
   displs   = malloc(numProcs*sizeof(MPI_INT));
   // Generate random numbers
   printf("Initializing data...\n");
   if (choise){
     for (i=0;i<len;i++){
       BigArray[i]= rand_normal(10.0, 3.0)/100.0 ;
       //printf("%f \n",BigArray[i]);
     }
   }else{
     for (i=0;i<len;i++){
       BigArray[i]= drand48();
       //printf("%f \n",BigArray[i]);
     }
   }
   start_time = MPI_Wtime();
   start_time2 = MPI_Wtime();
   printf("Sorting has begun please stand by...\n\n");
  }


  /*Step 1 : Divide data equally and sort locally*/

  /*Calculate the chunk size*/
  chunk = len / numProcs;
  
  L[0] = (double*)malloc((chunk+chunk/10)*sizeof(MPI_DOUBLE));
    if (L[0] == NULL) {
        printf("ERROR-> L[0] malloc FAILED\n");
        exit(-1);
    }
  L[1] = (double*)malloc((chunk+chunk/10)*sizeof(MPI_DOUBLE));
    if (L[1] == NULL) {
        printf("ERROR-> L[1] malloc FAILED\n");
        exit(-1);
    }

  RecvBuffer  = malloc(chunk*sizeof(MPI_DOUBLE));
    if (RecvBuffer == NULL) {
        printf("ERROR-> RecvBuffer malloc FAILED\n");
        exit(-1);
    }
    
  localLength = chunk;

  int x = 0;
  LocalArray = L[x];

  /*Distribute the data*/
  MPI_Scatter(BigArray, chunk, MPI_DOUBLE,
                  LocalArray, chunk, MPI_DOUBLE,
                  root, MPI_COMM_WORLD);

  /*Sort in Parallel*/
  quick_sort(LocalArray, 0 , localLength-1);

  int jj = 0;
  

  for (step = numProcs ; step > 1 ; step /= 2){ /*log2(P) steps*/

    /*Step 2 : Select pivot and B cast within processor set*/

    /*Select the median value*/
    //localPivot = LocalArray[localLength/2];
    /*B cast  the Medians*/
    //MPI_Allreduce(&localPivot, &globalPivot, 1, MPI_DOUBLE, MPI_SUM, Communicator[jj]);
    /*Get the Mean of the medians*/
    //globalPivot /= (double)step;

    /*Select the median value*/
    localPivot = LocalArray[localLength/2];

    /*Have one process to select the pivot*/
    if (subrank[jj] == root){globalPivot = localPivot;}
    
    /*B cast  the Median*/
    MPI_Bcast( &globalPivot, 1, MPI_DOUBLE, 0,  Communicator[jj] );

    //printf ("Rank[%d] ---- globalPivot2[%f] ------ Step %d\n",rank,globalPivot,step);

    /*Step 3 : Divide data according to pivot*/

    /*Find where the larger data start*/

    int larger = 0;
    int nitems = 0;
    if (globalPivot < localPivot) {
      for (i=localLength/2;i>=0;i--){
        if (LocalArray[i] < globalPivot){
          larger = i+1;
          nitems = localLength - larger;
          break;
        }
      }
    }else{
      for (i=localLength/2;i<localLength;i++){
        if (LocalArray[i] >= globalPivot){
          larger = i;
          nitems = localLength - larger;
          break;
        }
      } 
    }
    

    if ( rank % step < step/2 ){  

      //printf("Rank [%d] Sending %d out of %d \n",rank,nitems,localLength);
      MPI_Isend(LocalArray + larger, nitems, MPI_DOUBLE,
                  rank + step/2, 123, MPI_COMM_WORLD, &Srequest);
      
      /*How many items we have left*/            
      if (larger != 0) {localLength = larger;} 
      
      MPI_Recv( RecvBuffer, chunk, MPI_DOUBLE,
                  rank + step/2, 321, MPI_COMM_WORLD, &status);

      /*Check how many items we received*/
      MPI_Get_count(&status, MPI_DOUBLE, &nitems);

      /*Merge the received with what we had*/
      x++;
      x = x % 2;
      merge(RecvBuffer, nitems, LocalArray, 0, localLength, L[x]);
      localLength += nitems;
      

      MPI_Wait(&Srequest, &status);  //Wait for the send to finish
      LocalArray = L[x];

    }else {

      MPI_Isend(LocalArray, larger, MPI_DOUBLE,
                  rank - step/2, 321,MPI_COMM_WORLD, &Srequest);
      
      MPI_Recv( RecvBuffer, chunk, MPI_DOUBLE,
                  rank - step/2, 123, MPI_COMM_WORLD, &status);

      /*Check how many items we received*/
      MPI_Get_count(&status, MPI_DOUBLE, &nitems);
      
      x++;
      x = x % 2;

      /*Merge the received with what we had*/
      merge(RecvBuffer, nitems, LocalArray, larger, localLength, L[x]);

      /*How many items we have left*/
      localLength += nitems - larger;

      MPI_Wait(&Srequest, &status);  //Wait for the send to finish
      LocalArray = L[x];
      
    }
    jj++; /*Move to next communicator*/
  }

  /* First learn how many items each process has*/
  MPI_Gather( &localLength, 1, MPI_INT, rbuf, 1, MPI_INT, 
                                    root, MPI_COMM_WORLD); 

  int displsHelper = 0;
  if (rank == root) {
    for (i=0; i<numProcs; ++i) { 
      displs[i] = displsHelper;
      displsHelper += rbuf[i];
    } 
    finish_time2 = MPI_Wtime();
  }

  /*Collect data back to the root*/
  MPI_Gatherv( LocalArray, localLength, MPI_DOUBLE, BigArray, rbuf,
                             displs, MPI_DOUBLE,root, MPI_COMM_WORLD); 

  if (rank == root ) 
  {   
   finish_time = MPI_Wtime();
   elapsed_time = finish_time - start_time;
   elapsed_time2 = elapsed_time - (finish_time2 - start_time2);
   printf("Completed.\nComputation time: %f ",elapsed_time);
   printf("\nData collection time: %f \n\n",elapsed_time2);
   CheckIt(BigArray,len);
   free(BigArray);
  }

  free(L[0]);
  free(LocalArray);
  free(RecvBuffer);


  /* Clean up MPI  */
  for (i=0;i<logP;i++){
      MPI_Comm_free(&Communicator[i]);
      MPI_Comm_free(&proc_grid[i]);
  }
  MPI_Finalize(); /* Shut down */
  return 0;
}

/*This function is the serial version of quick sort as described
  in http://en.wikipedia.org/wiki/Quicksort */
void quick_sort(double *Array, int left ,int right)
{
  int pivotIndex,pivotNewIndex;
    if (left < right)
    {
        pivotIndex = MedianValue(Array, left, right);
        pivotNewIndex = partition(Array, left, right, pivotIndex);
        quick_sort(Array, left, pivotNewIndex - 1);
        quick_sort(Array, pivotNewIndex + 1, right);
    }
}
/*This function is the partition function as described
  in http://en.wikipedia.org/wiki/Quicksort */
int partition(double *Array, int left, int right, int pivotIndex)
{
  int i,storeIndex;
  double temp,pivotValue = Array[pivotIndex];
  /*Move Pivot number to the end*/
  temp = Array[right];
  Array[right] = Array[pivotIndex];
  Array[pivotIndex] = temp;
  storeIndex = left;
  for (i = left; i < right; i++)
  {
    if (Array[i] <= pivotValue)
    {
      /*Swap Array[i] and Array[left..right]*/
      temp = Array[i];
      Array[i] = Array[storeIndex];
      Array[storeIndex] = temp;
      storeIndex++;
    }
  }
  /* Move pivot to its final place*/
  temp = Array[right];
  Array[right] = Array[storeIndex];
  Array[storeIndex] = temp;
  return storeIndex;
}
/*Function to merge two sorted array together into a new
  sorted array*/
void merge (double *A, int a, double *B, int start, int b, double *C)
{
  int i,j,k;
  i = 0;
  j = start;
  k = 0;
  while (i < a && j < b) {
    if (A[i] <= B[j]) {
    /* copy A[i] to C[k] and move the pointer i and k forward */
    C[k] = A[i];
    i++;
    k++;
    }
    else {
      /* copy B[j] to C[k] and move the pointer j and k forward */
      C[k] = B[j];
      j++;
      k++;
    }
  }
  /* move the remaining elements in A into C */
  while (i < a) {
    C[k]= A[i];
    i++;
    k++;
  }
  /* move the remaining elements in B into C */
  while (j < b)  {
    C[k]= B[j];
    j++;
    k++;
  }
}
/*This function will find the pivot number by taking three
  numbers and comparing them together*/
int MedianValue(double *Array, int left, int right)
{
  double A,B,C;

  A = Array[left];
  B = Array[left + (right-left)/2];
  C = Array[right];

  if ( A > B )
  {
    if ( C > A )
    {
      return left;
    }else{
      if ( C < B )
      {
        return left + (right-left)/2;
      }else{
        return right;
    }
  }
  }else{
    if ( C > B )
    {
      return left + (right-left)/2;
    }else{
      if ( C < A )
      {
        return left;
      }else{
        return right;
      }
    }
  }

}
/*This function normal distributed data as taken from
  http://en.literateprograms.org/Box-Muller_transform_(C) */
double rand_normal(double mean, double stddev) {
  static double n2 = 0.0;
  static int n2_cached = 0;
  if (!n2_cached) {
    double x, y, r;
    do {
        x = 2.0*rand()/RAND_MAX - 1;
        y = 2.0*rand()/RAND_MAX - 1;
        r = x*x + y*y;
    } while (r == 0.0 || r > 1.0);
    {
    double d = sqrt(-2.0*log(r)/r);
    double n1 = x*d;
    n2 = y*d;
    double result = n1*stddev + mean;
    n2_cached = 1;
    return result;
    }
  } else {
      n2_cached = 0;
      return n2*stddev + mean;
  }
}
/*Just a function to validate the correctness of
  the program*/
int CheckIt(double *Array, int size)
{
  int i;
  for (i = 1; i < size; i ++)
  {
    if (Array[i] < Array[i-1])
    {
      printf("at loc %d, %f < %f \n", i, Array[i], Array[i-1]);
      return 0;
    }
  }
  return 1;
}



