#include <mpi.h>
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <algorithm>
using namespace std;

void merge(float a[], int m, float b[], int n, float sorted[]) {
  int i, j, k;

  j = k = 0;

  for (i = 0; i < m;) {
    if (j < m && k < n) {
      if (a[j] < b[k]) {
        sorted[i] = a[j];
        j++;
      }
      else {
        sorted[i] = b[k];
        k++;
      }
      i++;
    }
  }
}

void merge1(float a[], int m, float b[], int n, float sorted[]) {
  int i, j, k;

  j = k = 0;

  for (i = 0; i < m + n;) {
    if (j < m && k < n) {
      if (a[j] < b[k]) {
        sorted[i] = a[j];
        j++;
      }
      else {
        sorted[i] = b[k];
        k++;
      }
      i++;
    }
    else if (j == m) {
      for (; i < m + n;) {
        sorted[i] = b[k];
        k++;
        i++;
      }
    }
    else {
      for (; i < m + n;) {
        sorted[i] = a[j];
        j++;
        i++;
      }
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  int n = atoi(argv[1]);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_File f, output;
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
  MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output);
  int num = n/size;
  int remain = n - num*(size-1);
  float *data = new float[num];
  float *data1 = new float[remain];
  if (rank!=size-1){
    MPI_File_read_at(f, sizeof(float) * (rank*num), data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
    sort(data, data+num);
  }
  else {
    MPI_File_read_at(f, sizeof(float) * (rank*num), data1, remain, MPI_FLOAT, MPI_STATUS_IGNORE);
    sort(data1, data1+remain);
  }

  float *comp = new float[num];
  float *comp1 = new float[remain];
  float *mer = new float[num*2];
  float *mer1 = new float[num+remain];

  int k = size/2+1;
  if(size==1){
    sort(data1, data1+remain);
    if (rank==0) MPI_File_write(output, data1, n, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  else{
    while(k--) {
      if (rank%2==1) {
        if (rank==size-1){
          MPI_Sendrecv(data1, remain, MPI_FLOAT, rank-1, 0, comp, num, MPI_FLOAT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge1(data1, remain, comp, num, mer1);
          for (int i=0; i<remain; i++) data1[i] = mer1[num+i];
        }
        else {
          MPI_Sendrecv(data, num, MPI_FLOAT, rank-1, 0, comp, num, MPI_FLOAT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge1(data, num, comp, num, mer);
          for (int i=0; i<num; i++) data[i] = mer[num+i];
        }
      }
      else if (rank!=size-1){
        if (rank==size-2){
          MPI_Sendrecv(data, num, MPI_FLOAT, rank+1, 0, comp1, remain, MPI_FLOAT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge(data, num, comp1, remain, mer1);
          for (int i=0; i<num; i++) data[i] = mer1[i];
        }
        else {
          MPI_Sendrecv(data, num, MPI_FLOAT, rank+1, 0, comp, num, MPI_FLOAT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge(data, num, comp, num, mer);
          for (int i=0; i<num; i++) data[i] = mer[i];
        }
      }

      if (rank%2==0 && rank!=0) {
        if(rank==size-1){
          MPI_Sendrecv(data1, remain, MPI_FLOAT, rank-1, 0, comp, num, MPI_FLOAT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge1(data1, remain, comp, num, mer1);
          for (int i=0; i<remain; i++) data1[i] = mer1[i+num];
        }
        else{
          MPI_Sendrecv(data, num, MPI_FLOAT, rank-1, 0, comp, num, MPI_FLOAT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge1(data, num, comp, num, mer);
          for (int i=0; i<num; i++) data[i] = mer[i+num];
        }
      }
      else if (rank!=size-1 && rank!=0){
        if (rank==size-2){
          MPI_Sendrecv(data, num, MPI_FLOAT, rank+1, 0, comp1, remain, MPI_FLOAT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge(data, num, comp1, remain, mer1);
          for (int i=0; i<num; i++) data[i] = mer1[i];
        }
        else{
          MPI_Sendrecv(data, num, MPI_FLOAT, rank+1, 0, comp, num, MPI_FLOAT, rank+1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          merge(data, num, comp, num, mer);
          for (int i=0; i<num; i++) data[i] = mer[i];
        }
      }
    }

    if (rank!=size-1) MPI_File_write_at(output, sizeof(float) * (rank*num), data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
    else MPI_File_write_at(output, sizeof(float) * (rank*num), data1, remain, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  // MPI_File_close(&output);
  MPI_Finalize();
}
