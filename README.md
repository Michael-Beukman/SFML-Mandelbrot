# Mandlebrot and Julia sets in Parallel
This project was inspired by [this](https://www.youtube.com/watch?v=PBvLs88hvJ8) video.

It displays the Mandelbrot / Julia set using [SFMl](https://www.sfml-dev.org/) and speeds up the computation using OpenMP.

## Usage
Simply type `make` to compile. You might need to use `CXX=... make` to use a non standard compiler (as is necessary on mac to get OpenMP to work).

Then to run the program, you can simply type `./bin/main I`, where `I` is either 0 or 1, which will show the Mandelbrot or Julia set.

