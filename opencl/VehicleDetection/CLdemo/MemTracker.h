#ifndef __MEM_MENAGER_H__
#define __MEM_MENAGER_H__

#ifdef _DEBUG

#include <set>
#include <string>


namespace core{
    namespace util{
        
        struct MemTrackerStruct{
            void* ptr;
            std::string file;
            int lineNum;                        
            
            MemTrackerStruct(){
                ptr = 0;
                lineNum = -1;
                file = "";
            }
            
            MemTrackerStruct(const MemTrackerStruct& other){
                ptr = other.ptr;
                file = other.file;
                lineNum = other.lineNum;
            }
            
            bool operator==(const MemTrackerStruct &other) const;            
            bool operator<(const MemTrackerStruct& rhs) const;
            MemTrackerStruct& operator=(MemTrackerStruct other);
            std::string toString();
        };
        
        class MemTracker{            
        private:
            static std::set<MemTrackerStruct> allocatedByManager;
        protected:
        public:            
            static void add(MemTrackerStruct ptr) ;
            static void remove(void* ptr);
            static void remove(const void* ptr) ;
            static std::string getUnfreed();
        };                
        
    }
}

#endif

#if defined(_DEBUG)
    inline void * operator new (std::size_t size, const char* file, int line){
        if (void * p = ::operator new (size, std::nothrow)){
            core::util::MemTrackerStruct mtStruct;
            mtStruct.ptr = p;
            mtStruct.file = file;
            mtStruct.lineNum = line;
            core::util::MemTracker::add(mtStruct);
            return p ;
        }
        throw 0;
    }
    inline void* operator new[] (std::size_t size, const char* file, int line){
        if (void * p = ::operator new[] (size, std::nothrow)){
            core::util::MemTrackerStruct mtStruct;
            mtStruct.ptr = p;
            mtStruct.file = file;
            mtStruct.lineNum = line;
            core::util::MemTracker::add(mtStruct);
            return p ;
        }
        throw 0;
    }
    
    inline void operator delete(void * p, const char* file, int line){        
    }
    
    inline void operator delete[](void * p, const char* file, int line){        
    }
    #define New new(__FILE__, __LINE__)
    #define Delete(P) core::util::MemTracker::remove(P); delete P;
    #define DeleteArr(P) core::util::MemTracker::remove(P); delete[] P;
#else
    #define New new
    #define Delete(P) delete P;
    #define DeleteArr(P) delete[] P;
#endif

template<typename T> struct MemTrackerDeleter {
    void operator()(T* b) {
        T* ptr = const_cast<T*> (b);
        Delete(ptr);
    }
};

#define UNIQUE_PTR(TYPE) unique_ptr< TYPE, MemTrackerDeleter< TYPE > >

#endif