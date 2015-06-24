/* 
 * File:   RAIIS.h
 * Author: marko
 *
 * Created on June 1, 2014, 8:42 PM
 */

#ifndef RAIIS_H
#define	RAIIS_H

#include <cv.h>
#include <highgui.h>
#include <pthread.h>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "MemTracker.h"

#pragma comment(lib,"pthreadVC2.lib")

namespace core{
    namespace util{
        namespace raii{
            
            class ImageRaii{
            private:
                IplImage* image;
                ImageRaii(){
                    image = 0;
                }
            protected:
            public:
                ImageRaii(IplImage* img){
                    image = img;
                }
                ~ImageRaii(){
                    if (image != 0)
                        cvReleaseImage(&image);
                }
            };
            
            class MutexRaii{
            private:
                pthread_mutex_t* mutex;
                MutexRaii(){
                    mutex = 0;
                }
            protected:
            public:
                MutexRaii(pthread_mutex_t* mtx){
                    mutex = mtx;
                    pthread_mutex_lock(mutex);
                }
                
                ~MutexRaii(){
                    if (mutex != 0)
                        pthread_mutex_unlock(mutex);
                }
            };
            
            class FileRaii{
            private:
                std::fstream* file;
                FileRaii(){}
            protected:
            public:
                FileRaii(std::fstream* file){
                    this->file = file;
                }
                
                ~FileRaii(){
                    if (file){
                        file->flush();
                        file->close();
                    }
                }
            };
            
            /**allocate vector with MemMenager*/
            template<typename T> class VectorRaii{
            private:
                T* vector;
                
                VectorRaii(){
                    vector = 0;
                }                                
            protected:
            public:                                
                VectorRaii(T* vec){
                    vector = vec;
                }
                
                ~VectorRaii(){
                    if (vector)                        
                        delete [](vector);
                }                                
            };
            
            /**allocate vector and elements with MemMenager*/
            template<typename T> class MatrixRaii{
            private:
                T** matrix;
                int dim;
                
                MatrixRaii(){
                    matrix = 0;
                }                                
            protected:
            public:
                MatrixRaii(T** mat, int dimension){
                    matrix = mat;
                    dim = dimension;
                }
                
                ~MatrixRaii(){
                    if (matrix != 0){
                        for (int i = 0; i < dim; i++){                            
                            delete [](matrix[i]);
                        }
                        delete [](matrix);
                    }
                }
            };            
            
        }
    }
}

#endif	/* RAIIS_H */

