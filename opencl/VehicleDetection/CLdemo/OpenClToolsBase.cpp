#define _OPENCL
#ifdef _OPENCL
#include "OpenClToolsBase.h"
#include <iostream>
#include "RAIIS.h"
#include "MemTracker.h"

#pragma warning( disable : 4996 )
#pragma comment (lib, "OpenCL.lib")

#define MAX_DEVICES 100
#define MAX_SRC_SIZE 5242800
#define MAX_PLATFORMS 100

namespace core{
    namespace opencl{
        
        using namespace std;
        using namespace core::util::raii;
        using namespace core::util;
        
        OpenClBase::OpenClBase(){
        }
        
        OpenClBase::~OpenClBase(){
            cleanUp();
        }
        
        void OpenClBase::initVars(){
            initialized     = false;
            kernel          = 0;
            workGroupSize   = 0;
            kernelCount     = 0;
            program         = 0;
            context         = 0;
            command_queue   = 0;
        }
        
        void OpenClBase::cleanUp(){            
            if (program)
                clReleaseProgram(program);
            if (context)
                clReleaseContext(context);
            if (command_queue)
                clReleaseCommandQueue(command_queue);
            
            if (kernel){
                for (int i = 0; i < kernelCount; i++) {
                    if (kernel[i]){
                        clReleaseKernel(kernel[i]);
                        err_check(err, "OpenClBase::cleanUp clReleaseKernel");
                    }
                }
                delete [](kernel);
            }
            if (workGroupSize){
                delete [](workGroupSize);
            }
        }
        
        bool OpenClBase::hasInitialized(){
            return initialized;
        }
        
        void OpenClBase::err_check(int err, string err_code)  {
            if (err != CL_SUCCESS) {
                cout << "Error: " << err_code << "(" << err << ")" << endl;
                if (err == CL_BUILD_PROGRAM_FAILURE) {
                    // Determine the size of the log
                    size_t log_size;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                    // Allocate memory for the log
                    char* log = new char[log_size];
                    VectorRaii<char> vraii(log);
                    // Get the log
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                    // Print the log
                    cout << log << endl;
                    
                }

            }
        }
        
        size_t OpenClBase::shrRoundUp(size_t localSize, size_t allSize) {
            if (allSize % localSize == 0) {
                return allSize;
            }
            int coef = allSize / localSize;
            return ((coef + 1) * localSize);
        }
        
        void OpenClBase::loadProgramFile(const string& programFileName){
           // string usePrecompiledStr = Config::getInstancePtr()->getPropertyValue("general.openCL.UsePrecompiledKernels");
			string usePrecompiledStr = "false";
			bool usePrecompiled = usePrecompiledStr.compare("true") == 0;
            if (usePrecompiled){
                bool succ = loadProgramFromBinary(programFileName);
                if (succ){
                    return;
                }
            }
            loadProgramFileFromSource(programFileName);
            if (usePrecompiled){
                char* fp = saveProgramBinary(programFileName);
                if (fp != 0){
                    remove(fp);
                    delete [](fp);
                }
            }
        }
        
        void OpenClBase::loadProgramFileFromSource(const string& programFileName){
            fstream kernelFile;
            string file = programFileName + ".cl";
            kernelFile.open(file.c_str(), ifstream::in);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = new char[MAX_SRC_SIZE];
                if (buffer) {
                    VectorRaii<char> vraiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program = clCreateProgramWithSource(context, 1, (const char **) &buffer, &readBytes, &err);
                        err_check(err, programFileName + " clCreateProgramWithSource");
                        cout << "Build program: " << programFileName << " started" << endl;
                        err = clBuildProgram(program, 1, &device, 0, NULL, NULL);
                        err_check(err, programFileName + " clBuildProgram");
                        cout << "Build program: " << programFileName << " finished" << endl;
                    }                    
                }               
            } 
        }
        
        string OpenClBase::getBinaryFile(const string& programFileName){
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            char deviceName[256];
            err = clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, 0);            
            err_check(err, programFileName + " Get device name, save binary kernel");            
            string file = deviceName;
            file += "_" + programFileName + ".ptx";
            return file;
        }
        
        bool OpenClBase::loadProgramFromBinary(const string& kernelFileName){
            fstream kernelFile;            
            string file = getBinaryFile(kernelFileName);
            kernelFile.open(file.c_str(), ifstream::in | ifstream::binary);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = new char[MAX_SRC_SIZE];
                if (buffer) {
                    VectorRaii<char> vRaiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program = clCreateProgramWithBinary(context, 1, &device, &readBytes, (const uchar**)&buffer, 0, &err);
                        
                        err_check(err, kernelFileName + " clCreateProgramWithBinary");
                        
             
                        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
                        
						 err_check(err, kernelFileName + " clBuildProgram");
            
                    }                    
                } else {                    
                    return false;
                }                
            } else {
                return false;
            }
            return true;
        }
        
        char* getCStrCopy(string str){
            char* ret = new char[str.size() + 1];
            ret[str.size()] = '\0';
            strcpy(ret, str.c_str());
            return ret;
        }
        
        char* OpenClBase::saveProgramBinary(const string& kernelFileName){            
            fstream kernel;
            string programFile = getBinaryFile(kernelFileName);
            kernel.open(programFile.c_str(), ofstream::out | ofstream::binary);            
            if (kernel.is_open()){
                FileRaii fRaii(&kernel);
                cl_uint nb_devices;
                size_t retBytes;
                err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &nb_devices, &retBytes);
               
                    err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo");
            
         
                
                size_t* binarySize = 0;
                binarySize = new size_t[nb_devices];
                if (binarySize == 0){
                    
                    //cout << exc.what() << endl;
                    return getCStrCopy(programFile);
                }
                VectorRaii<size_t> bsRaii(binarySize);
                err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * nb_devices, binarySize, 0);
        
                    err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo2");

                uchar**  buffer = 0;
                buffer = new uchar*[nb_devices];
                if (buffer != 0){                    
                    for (unsigned int i = 0; i < nb_devices; i++){
                        buffer[i] = new uchar[binarySize[i]]; 
                    }
                    MatrixRaii<uchar> mRaii(buffer, nb_devices);
                    
                    size_t read;
                    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(uchar*)*nb_devices, buffer, &read);
                
                        err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo3");
        
                    //because I know that is on one device
                    kernel.write((const char*)buffer[0], binarySize[0]);
                }
                   
            }
           
            return 0;
        }
        
        void OpenClBase::createWorkGroupSizes() {
            workGroupSize = new size_t[kernelCount];
            for (int i = 0; i < kernelCount; i++) {
                if (kernel[i]){
                    err = clGetKernelWorkGroupInfo(kernel[i], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(workGroupSize[i]), &(workGroupSize[i]), NULL);
                    err_check(err, "penClBase::createWorkGroupSizes clGetKernelWorkGroupInfo");
					if (err == CL_SUCCESS){
						cout << "Finished Create Work Group Size...." << endl;
					}
                }
            }
        }
		
        vector<string> OpenClBase::getKernelNamesForClass(){
			/*
            vector<string> retVec;
            Config* conf = Config::getInstancePtr();
            string className = getClassName();
            string keyStr = "general.classes." + className + ".kernels";
            string k1 = keyStr + ".kernelCount";            
            string kerCountStr = conf->getPropertyValue(k1);
            int count = atoi(kerCountStr.c_str());
            for (int i = 0; i < count; i++){
                char index[4];
                sprintf(index, "%d", i);
                k1 = keyStr + ".kernelNo" + index;
                string kernelName = conf->getPropertyValue(k1);
                retVec.push_back(kernelName);
            }
            return retVec;
			*/
			vector<string> retVec;
			return retVec;
        }
		
		
        vector<string> OpenClBase::getProgramFilesForClass(){
            /*vector<string> retVec;
            Config* conf = Config::getInstancePtr();
            string className = getClassName();
            string keyStr = "general.classes." + className + ".programs.programFile";
            string programFile = conf->getPropertyValue(keyStr);
            retVec.push_back(programFile);
            keyStr = "general.classes." + className + ".programs.rootDir";
            dirToOpenclprogramFiles = conf->getPropertyValue(keyStr);
            return retVec;
			*/
			vector<string> retVec;
			return retVec;
        }
        
        void OpenClBase::createKernels(){
			//vector<string> kernelNames = getKernelNamesForClass();
			vector<string> kernelNames;
			kernelNames.push_back("predict");
			kernelCount = kernelNames.size();
            kernel = new cl_kernel[kernelCount];
            for (int i = 0; i < kernelCount; i++){
                kernel[i] = clCreateKernel(program, kernelNames[i].c_str(), &err);
                err_check(err, "OpenClBase::createKernels clCreateKernel: " + kernelNames[i]);
				if (err == CL_SUCCESS){
					cout << "Create Kernel " << kernelNames[i] << " Succeed...." << endl;
				}
            }            
        }
        
        void OpenClBase::init(unsigned int platformID, unsigned int deviceID, bool listOnly) {
            char info[256];
            cl_platform_id platform[MAX_PLATFORMS];
            cl_uint num_platforms;                        
            
            err = clGetPlatformIDs(MAX_PLATFORMS, platform, &num_platforms);
            err_check(err, "OpenclTools::init clGetPlatformIDs");
            cout << "Found " << num_platforms << " platforms." << endl;                        
            cout << "=============" << endl;
            for (unsigned int i = 0; i < num_platforms; i++) {
                cl_device_id devices[MAX_DEVICES];
                cl_uint num_devices;
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 256, info, 0);
                err_check(err, "OpenclTools::init clGetPlatformInfo");
                cout << "Platform name: " << info << endl;

#if defined _AMD
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
                    err_check(err, "OpenclTools::init clGetDeviceIDs");
                    cout << "Found " << num_devices << " devices" << endl;

                    for (unsigned int j = 0; j < num_devices; j++) {
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, info, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_NAME");
                        cl_device_type type;
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_TYPE");
                        string typeStr = "DEVICE_OTHER";
                        if (type == CL_DEVICE_TYPE_CPU)
                            typeStr = "DEVICE_CPU";
                        else if (type == CL_DEVICE_TYPE_GPU)
                            typeStr = "DEVICE_GPU";
                        cl_ulong maxAllocSize;
                        err = clGetDeviceInfo(devices[j],  CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxAllocSize, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE");
                        cout << "Device " << j << " name: " << info << " type: " << typeStr << " max alloc: " << maxAllocSize << endl;
                    }
                        

                cout << "=============" << endl;
            }
			//err = clGetPlatformInfo(platform[platformID], CL_PLATFORM_NAME, 256, info, 0);
			//cout << "use platform: " << info << endl;

            if (listOnly)
                return;
            
            if (platformID >= num_platforms){
                cout << "Fail Init platform" <<endl;
                ;
            }
            
            cl_device_id devices[MAX_DEVICES];
            cl_uint num_devices;
#if defined _AMD
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
            err_check(err, "OpenClBase::init clGetDeviceIDs2");
            if (deviceID >= num_devices){
               cout <<  "OpenClBase::Fail Init devices" <<endl;
                //throw exc;
            }
            device = devices[deviceID];
            
			err = clGetPlatformInfo(platform[platformID], CL_PLATFORM_NAME, 256, info, 0);
			cout << endl << "--- use platform: " << info << endl;
			err = clGetDeviceInfo(devices[deviceID], CL_DEVICE_NAME, 256, info, 0);
			cout << "--- use device: " << info << endl;
			cout << endl;

            context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
            err_check(err, "OpenClBase::init clCreateContext");
			if (err == CL_SUCCESS){
				cout << "Create Context Succeed...." << endl;
			}
            command_queue = clCreateCommandQueue(context, device, NULL, &err);
            err_check(err, "OpenClBase::init clCreateCommandQueue");
			if (err == CL_SUCCESS){
				cout << "Create Command Queue Succeed...." << endl;
			}

            cl_bool sup;
            size_t rsize;
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof (sup), &sup, &rsize);
            if (sup != CL_TRUE) {
                cout <<"OpenClBase::init Check for image support"<< endl;
                //throw exception;
            }
			else{
				cout << "CL_DEVICE_IMAGE_SUPPORT: Yes" << endl;
			}
            //image processing section
            //vector<string> programFile = getProgramFilesForClass();
			vector<string> programFile;
			programFile.push_back("libSvmPredict");
			
            loadProgramFile(programFile[0]);

            createKernels();

            createWorkGroupSizes();

            initialized = true;            
        }
        
    }
}
#endif
