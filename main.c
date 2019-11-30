//------------------------------------------------------------------------------
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//
//
// New Purpose:  Calculate surface area of 75 millions of cuboids
//               Show 100 first items of the result (both from OpenCL and sequential code)
//               Compare the speed difference with sequential code
//
// HISTORY:    Updated by Ezzat Chamudi, November 2019
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

extern double wtime(void);       // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------

const int LENGTH = 1024 * 1024 * 75;

const char *OpenCL_code = "\n" \
"__kernel void cuboid_area(                                          \n" \
"   __global int* a,                                                    \n" \
"   __global int* b,                                                    \n" \
"   __global int* c,                                                    \n" \
"   __global int* result)                                               \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   result[i] = 2 * ((a[i] * b[i]) + (b[i] * c[i]) +  (a[i] * c[i]));   \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    int          err;

    int*       source_a = (int*) calloc(LENGTH, sizeof(int));
    int*       source_b = (int*) calloc(LENGTH, sizeof(int));
    int*       source_c = (int*) calloc(LENGTH, sizeof(int));

    int*       result_opencl = (int*) calloc(LENGTH, sizeof(int));
    int*       result_sequential = (int*) calloc(LENGTH, sizeof(int));

    size_t global;

    cl_device_id     device_id = NULL;
    cl_context       context;
    cl_command_queue commands;
    cl_program       program;
    cl_kernel        kernel_cuboid_area;

    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;

    cl_mem d_result;

    // Fill vectors a and b with random integer values
    for(int i = 0; i < LENGTH; i++) {
        source_a[i] = (rand() % 9) + 1;
        source_b[i] = (rand() % 9) + 1;
        source_c[i] = (rand() % 9) + 1;
    }

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Get number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platform ids
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Finding a device");

    // Get device type
    long int device_type;
    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    if(device_type  == CL_DEVICE_TYPE_GPU)
       printf("Device type: GPU \n");
    else if (device_type == CL_DEVICE_TYPE_CPU)
       printf("Device type: CPU \n");
    else
       printf("Device type: Not CPU nor GPU \n");

    
    // Get total compute units
    int comp_units;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }
    printf("Total compute units: %d compute units \n", comp_units);
    
    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & OpenCL_code, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    kernel_cuboid_area = clCreateKernel(program, "cuboid_area", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * LENGTH, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * LENGTH, NULL, &err);
    checkError(err, "Creating buffer d_b");

    d_c  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * LENGTH, NULL, &err);
    checkError(err, "Creating buffer d_c");

    d_result  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH, NULL, &err);
    checkError(err, "Creating buffer d_result");

    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * LENGTH, source_a, 0, NULL, NULL);
    checkError(err, "Copying source_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * LENGTH, source_b, 0, NULL, NULL);
    checkError(err, "Copying source_b to device at d_b");

    err = clEnqueueWriteBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * LENGTH, source_c, 0, NULL, NULL);
    checkError(err, "Copying source_c to device at d_c");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel_cuboid_area, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel_cuboid_area, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel_cuboid_area, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel_cuboid_area, 3, sizeof(cl_mem), &d_result);
    checkError(err, "Setting kernel arguments");

    double cl_time = wtime();

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = LENGTH;
    err = clEnqueueNDRangeKernel(commands, kernel_cuboid_area, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    cl_time = wtime() - cl_time;
    printf("\nThe OpenCL kernel ran in %lf seconds\n", cl_time);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_result, CL_TRUE, 0, sizeof(float) * LENGTH, result_opencl, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }
    
    // Sequential testing;
    double seq_time = wtime();
    for (int i = 0; i < LENGTH; i++) {
        result_sequential[i] =  2 * ((source_a[i] * source_b[i]) + (source_b[i] * source_c[i]) +  (source_a[i] * source_c[i]));
    }
    seq_time = wtime() - seq_time;
    printf("The sequential code ran in %lf seconds\n\n", seq_time);
    
    double ratio = seq_time / cl_time;

    printf("The sequential time is %lfX of the OpenCL time\n\n", ratio);

    // Print Result
    for(int i = 0; i < 100; i++) {
        printf("a=%d\tb=%d\tc=%d\t\topencl=%d\t\tseq=%d\n",
               source_a[i],
               source_b[i],
               source_c[i],
               result_opencl[i],
               result_sequential[i]
               );
    }
    printf("... %d more items\n", LENGTH - 100);
        
    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel_cuboid_area);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(source_a);
    free(source_b);
    free(source_c);
    free(result_opencl);
    free(result_sequential);

    printf("\n");

    return 0;
}

