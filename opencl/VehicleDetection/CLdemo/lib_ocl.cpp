//
#include "lib_ocl.h"
#include <stdio.h>
#include <stdlib.h>

cl_device_id getOneDevice(){
	// Get Platform and Devices infos.
	cl_uint num_platforms;
	cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);

	if (num_platforms <= 0){
		fprintf(stderr, "No Platform.\n");
		exit(-1);
	}

	cl_platform_id platform_id[3];
	err |= clGetPlatformIDs(3, platform_id, NULL);
	if (err != CL_SUCCESS){
		fprintf(stderr, "Failed to Get Platform.\n");
		exit(1);
	}

	cl_device_id device_id;
	err = clGetDeviceIDs(platform_id[2], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS){
		fprintf(stderr, "Failed to Get Device.\n");
		exit(1);
	}
	/*	支持浮点数扩展...
	char ext_data[4096];
	clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(ext_data), ext_data, NULL);

	printf("EXTENSIONS: %s\n\n", ext_data);*/
	return device_id;
}

cl_kernel loadKernel(const char* fileName, const char* kernelName, cl_device_id device_id, cl_context context, cl_program &program){
	//	Load the source code containing the kernel.
	FILE *fp;
	char *source_str = new char[MAX_SOURCE_SIZE];
	size_t source_size;

	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(-1);
	}
	source_size = fread(source_str, sizeof(char), MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//	Create Kernel Program from the source.
	cl_int err;
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);

	// Build Kernel Program
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr, "clBuild failed:%d\n", err);
		char info_buf[MAX_INFO_SIZE];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, MAX_INFO_SIZE, info_buf, NULL);
		fprintf(stderr, "\n%s\n", info_buf);
		exit(-1);
	}
	else{
		char info_buf[MAX_INFO_SIZE];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, MAX_INFO_SIZE, info_buf, NULL);
		printf("Kernel Build Success\n%s\n", info_buf);
	}
	// Create OpenCL Kernel
	cl_kernel kernel = clCreateKernel(program, kernelName, &err);
}

cl_ulong getStartEndTimeNs(cl_event ev){
	//	计算kerenl执行时间
	cl_ulong startTime, endTime;

	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);

	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);

	return (endTime - startTime);
}