
#ifndef __OCL_UTILS_H
#define __OCL_UTILS_H

#include "svm.h"

int SVMPredict2(struct svm_model* model, float *descript_vector, double *prob_est, int size);
int add1(int a);
#endif