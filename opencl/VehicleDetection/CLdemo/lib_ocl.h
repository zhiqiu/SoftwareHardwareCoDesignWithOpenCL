#ifndef __LIB_OCL_H
#define	__LIB_OCL_H

#include <CL/cl.h>

#pragma warning( disable : 4996 )
#pragma comment (lib, "OpenCL.lib")

#pragma OPENCL EXTENSION cl_intel_fp64 : enable

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_INFO_SIZE (0x10000)

cl_device_id getOneDevice();

cl_kernel loadKernel(const char* fileName, const char* kernelName, cl_device_id device_id, cl_context context, cl_program &program);

cl_ulong getStartEndTime(cl_event ev);

#endif	// end of __LIB_OCL_H