#include <stdio.h>

// World Coordinates
__device__ void get_world_coords(int screenx, int screeny, double& worldx, double& worldy,
    double scalex, double scaley, double offsetx, double offsety
){
    worldx = screenx / scalex + offsetx;
    worldy = screeny / scaley + offsety;
}


// This does the julia iteration count
__global__ void get_julia_iters(int* iteration_count, int _WIDTH, int _HEIGHT, int MAX_ITERS, double scalex, double scaley, double offsetx, double offsety)
{

    // get the current thread's x and y values
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;


    // transform into world coords
    double za, zb;
    get_world_coords(tx, ty, za, zb, scalex, scaley, offsetx, offsety);
    double ca = -0.8, cb = 0.156;

    // loop over
    int iters = 0;
    for (; iters < MAX_ITERS; ++iters) {

        double tempa = za * za - zb * zb + ca;
        double tempb = 2 * za * zb + cb;
        za = tempa; zb = tempb;
        if (za*za + zb*zb >= 4) break;
    }
    // add the number of iterations to the array
    iteration_count[ty * _WIDTH + tx] = iters;
}

// see comments above for julia
__global__ void get_mandelbrot_iters(int* iteration_count, int _WIDTH, int _HEIGHT, int MAX_ITERS, double scalex, double scaley, double offsetx, double offsety)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    double ca, cb;
    get_world_coords(tx, ty, ca, cb, scalex, scaley, offsetx, offsety);

    int iters = 0;
    double za = 0, zb = 0;
    for (; iters < MAX_ITERS; ++iters) {

        double tempa = za * za - zb * zb + ca;
        double tempb = 2 * za * zb + cb;
        za = tempa; zb = tempb;
        if (za*za + zb*zb >= 4) break;
    }
    
    iteration_count[ty * _WIDTH + tx] = iters;
}