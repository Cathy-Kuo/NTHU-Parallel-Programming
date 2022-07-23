#include <mutex>
#include <cstdio>
#include <pthread.h>
#include <stdlib.h>
#include <thread>
#include <iostream>
#include <math.h>
void threadRoutine(int a, int b, int cpus, double*c) {
  double result = 0;
  if (a!=cpus-1){
    for (int i=a*(b/cpus); i<a*(b/cpus)+b/cpus; i++){
      double ik = (double)i/b;
      double ik2 = pow(ik, 2);
      result += (pow((1-ik2),0.5)/b)*4;
    }
  }
  else{
    for (int i=a*(b/cpus); i<b; i++){
      double ik = (float)i/b;
      double ik2 = pow(ik, 2);
      result += (pow((1-ik2),0.5)/b)*4;
    }
  }
  *c = result;
}
int main(int argc, char** argv) {
  cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
  int cpus = CPU_COUNT(&cpuset);
	int slices = atoi(argv[1]);
  double c[cpus];
  double sol=0;
  std::thread th[cpus];
  for (int i=0; i<cpus; i++) {
    th[i] = std::thread(threadRoutine, i, slices, cpus, &c[i]);
   }
   for (int i=0; i<cpus; i++) {
     th[i].join();
   }
   if (cpus>slices) sol=c[cpus-1];
   else {
     for (int i=0; i<cpus; i++) sol+=c[i];
   }

	std::cout << "result: " << sol << std::endl;
}
