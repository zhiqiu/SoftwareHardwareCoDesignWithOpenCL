#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace core{
    namespace util{
        
        template <typename T> class Matrix{
        private:
            T* vector;
            int width, height;
            bool changed;
            Matrix();            
        protected:
        public:
            class Vector{
                friend class Matrix;
            private:
                T* pointerToVec;
                int width;
            protected:
                void setWidth(int val){
                    width = val;                    
                }
                void setVec(T* vec){
                    pointerToVec = vec;
                }
            public:
                Vector& operator=(T* vec){
                    memcpy(pointerToVec, vec, sizeof(T) * width);
                    return *this;
                }
                
                T& operator[](int idx){
                    return pointerToVec[idx];
                }
            };
            Matrix(T* const* mat, int width, int height);
            Matrix(const T* vec, int width, int height);
            Matrix(int width, int height);
            virtual ~Matrix();
            void swap(int i, int j);
            const bool changedValues();
            //operators                                   
            const T* operator[](int idx) const{
                return (vector + (idx * width));
            }
            
            const T* operator()(int idx){
                return (vector + (idx * width));
            }
            
            Vector& operator[](int idx){
                return v[idx];
            }
            
            operator const T*() const{
                return vector;
            }
            
            int getWidth() const{
                return width;
            }
            
            int getHeight() const{
                return height;
            }
            
            T* getVec() const{
                return vector;
            }        
        private:
            Vector* v;
        };
        
        template <typename T> Matrix<T>::Matrix(){
            vector = 0;
            changed = true;
            v = 0;
        }
        
        template <typename T> Matrix<T>::Matrix(T* const* mat, int width, int height){
            this->width = width;
            this->height = height;
            vector = new T[height * width];
            v = new Vector[height];
            for (int i = 0; i < height; i++){
                memcpy(vector + i * width, mat[i], width * sizeof(T));
                v[i].setWidth(width);
                v[i].setVec(vector + i * width);
            }
            changed = true;
        }
        
        template <typename T> Matrix<T>::Matrix(const T* vec, int width, int height){
            this->width = width;
            this->height = height;
            vector = new T[height * width];
            v = new Vector[height];
            memcpy(vector, vec, width * height * sizeof(T));
            for (int i = 0; i < height; i++){
                v[i].setWidth(width);
                v[i].setVec(vector + i * width);
            }
        }
        
        template <typename T> Matrix<T>::Matrix(int width, int height){
            this->width = width;
            this->height = height;
            vector = new T[height * width];
            v = new Vector[height];
            for (int i = 0; i < height; i++){
                v[i].setWidth(width);
                v[i].setVec(vector + i * width);
            }
        }
        
        template <typename T> Matrix<T>::~Matrix(){
            if (vector)
                delete [] vector;
            if (v)
                delete [] v;
        }
        
        template <typename T> void Matrix<T>::swap(int i, int j){
            T* tmp = new T[width];
            memcpy(tmp, vector + (i * width), width * sizeof(T));
            memcpy(vector + (i * width), vector + (j * width), width * sizeof(T));
            memcpy(vector + (j * width), tmp, width * sizeof(T));
            delete []Arr(tmp);
            changed = true;
        }
        
        template <typename T> const bool Matrix<T>::changedValues(){
            bool val = changed;
            changed = false;
            return val;
        }
        
    }
}

#endif
