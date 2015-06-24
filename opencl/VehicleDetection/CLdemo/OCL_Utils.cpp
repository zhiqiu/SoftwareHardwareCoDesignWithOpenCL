
#include "OCL_Utils.h"
#include <stdlib.h>

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
	struct svm_node* x = (struct svm_node *) malloc(n*sizeof(struct svm_node));

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