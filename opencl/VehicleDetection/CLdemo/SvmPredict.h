#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include "Singleton.h"
#include "IPrediction.h"
#include "Matrix.h"
//#include "core/util/rtti/ObjectFactory.h"

struct svm_model;

namespace core{
    namespace util{
        namespace prediction{
            namespace svm{

				class SvmPredict : public IPrediction, public Singleton<SvmPredict> {
					friend class Singleton<SvmPredict>;
                private:
                    svm_model* model;
                public:
                    SvmPredict();
                public:
                    ~SvmPredict();
                    void loadModel();
					unsigned char* predict(const Matrix<float>* imagePixelsParameters,
                                            const int& pixCount, const int& parameterCount);
                    bool hasLoadedModel();
                };

            }
        }
    }
}

#endif