#ifndef __IPREDICTION_H__
#define __IPREDICTION_H__

#include "Matrix.h"
#include <string>

namespace core{
    namespace util{
        namespace prediction{
         
            class IPrediction{                
            private:                
            protected:
            public:                
                virtual void loadModel() = 0;
                virtual unsigned char* predict( const core::util::Matrix<float>* imagePixelsParameters, 
                                const int& pixCount, const int& parameterCount) = 0;
                virtual bool hasLoadedModel() = 0;
            };
            
        }
    }
}

#endif
