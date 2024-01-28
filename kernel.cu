#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<cstdio>
#include<time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
using namespace std;
void generate(int* a, int size, int max, int min) {
	for (unsigned long long int i = 0; size > i; i++) {
		a[i] = rand() % max + 1 + min;
	}
}
__global__ void prefix_sum(int* in, int* out, int tn, int bn) {
	int  tid = threadIdx.x, bid = blockIdx.x, sid = blockIdx.x * tn + tid, p = 1, q = 0;
	//printf("in[%d]:%d\n ", sid, in[sid]);
	//printf("\n");
	extern __shared__ int add[];
	add[tid] = in[sid];
	for (int z = 1; tn > z; z *= 2) {
		p = 1 - p; //0 1 0 1
		q = 1 - p; //1 0 1 0
		if (tid >= z) {
			add[q * tn + tid] = add[p * tn + tid] + add[p * tn + tid - z];
			//printf("add[%d]:%d\n ", sid, add[q * tn + tid]);
			//printf("\n");
		}
		else {
			add[q * tn + tid] = add[p * tn + tid];
			//printf("add[%d]:%d\n ", sid, add[q * tn + tid]);
			//printf("\n");
		}
		__syncthreads();
	}
	//printf("out[%d]:%d\n ", sid, out[sid]);
	//printf("\n");
	out[sid] = add[q * tn + tid];
	out[sid] += out[bid * tn - 1];
	__syncthreads();
}

bool test(int a[], int b[], int max) {
	return 0;
}
__global__ void cpuprefixsum(int* array, int data) {
	for (int i = 1; data > i; i++) {
		array[i] = array[i - 1] + array[i];
	}
}
main() {
	srand(time(NULL));
	while (1) {
		int data; int maxr = 10, minr = 0, n, tn, ctr[1] = { 1 };
		clock_t start_t, stop_t;
		double time;
		cudaEvent_t start, stop;
		float htd, dth, kernel, cpukernel;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		printf("輸入要連加的整數數量(2的n次方):");
		cin >> n;
		data = pow(2, n);
		tn = (data < 1024) ? data : 1024;
		//計算幾格block
		int bn = (data % tn == 0) ? data / tn : data / tn + 1;
		//獲取陣列
		int* array = new int[data];
		//產生亂數
		generate(array, data, maxr, minr);
		//顯示初始陣列
		/*for (int i = 0; data > i; i++) {
			printf(" %d", array[i]);
		}*/
		//printf("\n");
		//獲取gpu陣列
		int* gpuarray, * result = new int[data], * cpukernelarray;
		cudaMalloc((void**)&gpuarray, data * sizeof(int));
		cudaMalloc((void**)&cpukernelarray, data * sizeof(int));
		cudaMemcpy(cpukernelarray, array, sizeof(int) * data, cudaMemcpyHostToDevice);
		//CPU作法丟GPU跑
		cudaEventRecord(start, 0);
		//cpuprefixsum << <1, 1 >> > (cpukernelarray, data);
		cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&cpukernel, start, stop);
		//將cpu陣列轉至gpu陣列
		cudaEventRecord(start, 0);
		cudaMemcpy(gpuarray, array, sizeof(int) * data, cudaMemcpyHostToDevice);

		cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&dth, start, stop);
		//printf("x:%d", x);
		cudaEventRecord(start, 0);
		prefix_sum << <bn, tn, tn * 2 * sizeof(int) >> > (gpuarray, gpuarray, tn, bn);
		cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&kernel, start, stop);
		cudaEventRecord(start, 0);
		cudaMemcpy(result, gpuarray, sizeof(int) * data, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&htd, start, stop);
		//CPU prefix_sum
		start_t = clock();
		for (int i = 1; data > i; i++) {
			array[i] = array[i - 1] + array[i];
		}
		stop_t = clock();
		time = double(stop_t - start_t) / CLOCKS_PER_SEC;
		//檢驗
		if (test(array, result, data)) {
			printf("incorrect\n");
		}
		else {
			printf("correct\n");
		}
		printf("CPU: %f\n", time);
		printf("CPU to GPU: %f\n", htd / 1000);
		printf("GPU: %f\n", kernel / 1000);
		printf("GPU to CPU: %f\n", dth / 1000);
		//printf("CPU in kernel: %f\n", cpukernel / 1000);
		//輸出result
		/*for (int i = 0; data > i; i++) {
			printf("[%d] %d\n", i, result[i]);
		}*/

	}
}
