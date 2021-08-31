# Mandlebrot and Julia sets in Parallel
This project was inspired by [this](https://www.youtube.com/watch?v=PBvLs88hvJ8) great video. Some of the AVX code was taken from the corresponding [Github Repository](https://github.com/OneLoneCoder/olcPixelGameEngine/blob/master/Videos/OneLoneCoder_PGE_Mandelbrot.cpp).

The program displays the Mandelbrot / Julia set using [SFML](https://www.sfml-dev.org/) and speeds up the computation using OpenMP, as well as using AVX-256 instructions.

## Usage
Simply type `make` to compile. You might need to use `CXX=... make` to use a non standard compiler (as is necessary on mac to get OpenMP to work). If your CPU does not support AVX-256, then you might need to remove the `-DUSE_AVX` flag in the `makefile`.

Then to run the program, you can simply type `./bin/main I`, where `I` is either 0 or 1, which will show the Mandelbrot or Julia set.
## Example
![Julia Set](images/julia.png)
![Mandelbrot Set](images/mandelbrot.png)

