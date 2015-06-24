
#pragma OPENCL EXTENSION cl_khr_fp64 : enable 

#include "svm.h"

global void* malloc(size_t size, global uchar *heap, global uint *next)
{
	uint index = atomic_add(next, size);
	return heap + index;
}


int SVMPredict2(struct svm_model* model, float *descript_vector, double *prob_est, int size)// prob_est is a pointer to array
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type = svm_get_svm_type(model);
	int nr_class = svm_get_nr_class(model);
	double *prob_estimates = NULL;
	int j;

	int i = 0;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

	int n = 1;
	while (n < size)
		n *= 2;
	//struct svm_node* x = (struct svm_node *) malloc(n*sizeof(struct svm_node));
	struct svm_node x[10000];
	for (int i = 0; i < size; i++)
	{
		x[i].index = i + 1;
		x[i].value = descript_vector[i];
	}
	x[size].index = -1;
	if (prob_est != NULL)
		predict_label = svm_predict_probability(model, x, prob_est);
	else
		predict_label = svm_predict(model, x);

	if (prob_est != NULL)
		*prob_est = -1;
	free(x);
	return predict_label;
}
int add1(int a){
	return a + 1;
}

__kernel void svmProcess(
	__global float* descriptors, __global double* scores, 
	__global int* x, __global int* y, __global int* width, 
	__global int* height, float zoom, int size,
	__constant const struct svm_model* restrict model){
	

	// Get global position in X direction
	int row = get_global_id(0);
	// Get global position in Y direction
	int col = get_global_id(1);

	int WIDTH = get_global_size(0);

	float descript_vector[1764];
	int descriptorSize = 0;
	int i = (row + col*WIDTH )*1764;
	int end = (row+(col+1)*WIDTH)*1764;
	for( ; i <= end; i++){
		if(i >= size){
			break;
		}
		descript_vector[descriptorSize++] = descriptors[i];
	}
	int xx = add1(5);
	double score[10];
	struct svm_model model_tmp = *model;
	//mem_cpy(model, model_tmp, sizeof(struct svm_model));
	//int SVMPredict2(struct svm_model* model, float *descript_vector, double *prob_est, int size);
	int result = SVMPredict2(&model_tmp, descript_vector, score, descriptorSize);

	int index = row*WIDTH + col;
	if (result == 1)	//	输出结果如何并行处理？？？	每个item一个存放位置，copy到host后如果！NULL就将该数组元素push_back
	{
		scores[index] = score[0];
		x[index] = (int)row / zoom;
		y[index] = (int)col / zoom;
		width[index] = (int)64 / zoom;
		height[index] = (int)64 / zoom;
	}
	else{
		scores[index] = '\0';
		x[index] = '\0';
		y[index] = '\0';
		width[index] = '\0';
		height[index] = '\0';
	}
	//free(model_tmp);
}