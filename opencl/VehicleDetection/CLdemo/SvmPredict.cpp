
#define _OPENCL

#include <iostream>
#include <memory>
#include "svm.h"
#include "SvmPredict.h"
#ifdef _OPENCL
#include "OpenCLToolsPredict.h"
#endif
#include "Matrix.h"

namespace core{
    namespace util{
        namespace prediction{
            namespace svm {
                using namespace std;
                using namespace core::util;
#ifdef _OPENCL
                using namespace core::opencl::libsvm;
#endif

                SvmPredict::SvmPredict() {
                    model = 0;
                }

                SvmPredict::~SvmPredict() {
                    if (model)
                        svm_free_and_destroy_model(&model);
                }

                void SvmPredict::loadModel(){
					string modelFile = "vehicle_detect.model";
                    model = svm_load_model(modelFile.c_str());
                    if (model == NULL) {
						cout << "Empty Model...." << endl;
                    }
                }

                unsigned char* SvmPredict::predict(const Matrix<float>* imagePixelsParameters, const int& pixCount, const int& parameterCount) {
                    if (model == NULL) {
						cout << "Empty Model..." << endl;
                    }
                    unsigned char* ret = 0;
					if (imagePixelsParameters == NULL){
						cout << "Empty Image Pixels Parameters..." << endl;
						return 0;
					}
#ifndef _OPENCL                                
					ret = new  unsigned char[pixCount];
                    for (int i = 0; i < pixCount; i++) {
                        if (i % 1000 == 0)
                            cout << "Pix no: " << i << endl;
                        const float* x = (*imagePixelsParameters)[i];
                        svm_node *nodes = new svm_node[parameterCount + 1];
                        for (int j = 0; j < parameterCount; j++) {
                            (nodes)[j].index = j + 1;
                            (nodes)[j].value = x[j];
                        }
                        (nodes)[parameterCount].index = -1;
                        (nodes)[parameterCount].value = 0.;
                        double val = svm_predict(model, nodes);
						ret[i] = (unsigned char)round(val);
                    }
#else
					/*
                    if (OpenCLToolsPredict::getInstancePtr()->hasInitialized() == false) {
						cout << "getInstancePtr()->hasInitialized() == false" << endl;
                    }*/
                    ret = OpenCLToolsPredict::getInstancePtr()->predict(model, imagePixelsParameters);
#endif
                    return ret;
                }

                bool SvmPredict::hasLoadedModel() {
                    return model != 0;
                }
            }
        }
    }
}