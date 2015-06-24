/* 
 * File:   Singleton.h
 * Author: marko
 *
 * Created on May 31, 2014, 6:34 PM
 */

#ifndef SINGLETON_H
#define	SINGLETON_H
#include <iostream>
#include "RAIIS.h"
//#include "typedefs.h"
using namespace std;
namespace core{
    namespace util{
        
        /**
         * template singleton class
         */
        template <class T> class Singleton{
        private:
            static T* instancePtr;            
        protected:
            Singleton();            
            virtual ~Singleton();
        public:
            /**
             * 
             * @return instance of T class
             */
            static T* getInstancePtr();
            /**
             * maybe someone will call it sometime !!??
             */
            static void destroy();
        };
        
        template<class T> T* Singleton<T>::instancePtr = 0;
        
        template<class T> Singleton<T>::Singleton(){            
        }
        
        template<class T> Singleton<T>::~Singleton(){            
        }
        
        template<class T> T* Singleton<T>::getInstancePtr(){            
            pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
            raii::MutexRaii autoLock(&mutex);
            static T* instancePtrTmp = 0;
            if (instancePtrTmp == 0 || instancePtr == 0){
                instancePtrTmp = New T();
                if (instancePtrTmp == 0){
                    cout << "Init singleton" <<endl;
                    //throw exc;
                }
                else{
                    instancePtr = instancePtrTmp;
                }
            }
            return instancePtr;
        }
        
        template<class T> void Singleton<T>::destroy(){           
            pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
            raii::MutexRaii autoLock(&mutex);
            if (instancePtr != 0){
                delete [](instancePtr);                
            }
            instancePtr = 0;
        }

    }
}

#endif	/* SINGLETON_H */

