#define WIDTH 1600
#define HEIGHT 1600
#define USE_OMP
// #define DEBUG
int MAX_ITERS = 128;
int WHICH_SET = 0;
#include <immintrin.h>
#include <omp.h>

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <chrono>
#include <cmath>
#include <string>

#include "complex.h"
int current_microseconds() {
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() * 1000;
}

struct Timer {
    int start;
    int end;
    std::string name;
#ifdef DEBUG
    Timer(std::string _name) : name(_name), start(current_microseconds()) {}
    ~Timer() {
        end = current_microseconds();
        printf("%s took %lf us\n", name.c_str(), (double)(end - start) / 1e6);
    }
#else
    Timer(std::string _name) {}
#endif
};

typedef sf::Vector2<double> vec2;
typedef sf::Vector2<int> vec2i;

struct Application {
    std::vector<int> iteration_count;
    vec2 scale = {400, 400}, offset = {-WIDTH / 2, -HEIGHT / 2};
    Application() : iteration_count(HEIGHT * WIDTH, 0) {
        offset.x /= scale.x;
        offset.y /= scale.y;
    }

    vec2 screen_to_world(const vec2& screen) {
        return {
            screen.x / scale.x + offset.x,
            screen.y / scale.y + offset.y};
    }

    vec2 world_to_screen(const vec2& world) {
        return {
            ((world.x - offset.x) * scale.x),
            ((world.y - offset.y) * scale.y)};
    }

    void update_vec() {
#ifdef USE_AVX
        if (WHICH_SET == 0)
            do_intrinsics();
        else
            do_intrinsics_julia();
        return;
#endif

// This is the normal, non-avx things
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int vec_index = 0; vec_index < WIDTH * HEIGHT; ++vec_index) {
            int x = vec_index % WIDTH;
            int y = vec_index / HEIGHT;
            vec2 screen_pos = {(double)x, (double)y};
            vec2 world_pos = screen_to_world(screen_pos);
            // now do things
            int iters;
            if (WHICH_SET == 0)
                iters = get_iters(world_pos);
            else
                iters = get_iters_julia(world_pos);
            iteration_count[vec_index] = iters;
        }
    }

    int get_iters(vec2 world_pos) {
        int iters = 0;
        Complex z = {0, 0};
        Complex c = {world_pos.x, world_pos.y};
        for (; iters < MAX_ITERS; ++iters) {
            z = z.square() + c;
            if (z.norm_sq() >= 4) break;
        }

        return iters;
    }

    int get_iters_julia(vec2 world_pos) {
        int iters = 0;
        Complex z = {world_pos.x, world_pos.y};
        Complex c = {-0.8, 0.156};
        for (; iters < MAX_ITERS; ++iters) {
            z = z.square() + c;
            if (z.norm_sq() >= 4) break;
        }
        return iters;
    }
#ifdef USE_AVX
    void do_intrinsics() {
#pragma omp parallel
        {
            // how many threads to split up in
            int num_threads = omp_get_num_threads();


            // the scale in the x and y directions
            __m256d _xscale = _mm256_set1_pd(1 / scale.x);
            __m256d _yscale = _mm256_set1_pd(1 / scale.y);
            
            // Some variables
            __m256d zr, zi, cr, ci, temp_zr, temp_zi;
            __m256d zr2, zi2, norm;
            __m256d four, two;
            __m256d x, y, _mask1;
            __m256i _one, _c, _n, _iterations, _mask2;
            __m256d onetwothreefour = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);

            // set 4 and 2
            four = _mm256_set1_pd(4.0);
            two = _mm256_set1_pd(2.0);
            _one = _mm256_set1_epi64x(1);

            // What is the top left corner of the screen in world space?
            vec2 test = {0, 0};
            double offset_x_in_world = screen_to_world(test).x;
            double offset_y_in_world = screen_to_world(test).y;
            
            // x = [0, 1, 2, 3] * scale + offset. The 0, 1, 2, 3 is to offset each element of our vector, as well as starting 
            // with the first 4 pixels.
            x = _mm256_add_pd(_mm256_mul_pd(onetwothreefour, _xscale), _mm256_set1_pd(offset_x_in_world));
            // How many iterations are there?
            _iterations = _mm256_set1_epi64x(MAX_ITERS);
            
            // similarly for y, set it as the offset, and the + (bracket) serves to split the screen into rows for each thread.
            y = _mm256_set1_pd(offset_y_in_world + (1 / scale.y * HEIGHT / num_threads * omp_get_thread_num()));
            // Where should we start and end with our loop
            int start = omp_get_thread_num() * HEIGHT / num_threads * WIDTH;
            int end = (omp_get_thread_num() + 1) * HEIGHT / num_threads * WIDTH;

            for (int i = start; i < end; i += 4) {
                if (i % (WIDTH) == 0 && i != 0) {
                    // new row, so reset things
                    // x is again the first four pixels in world space.
                    x = _mm256_add_pd(_mm256_mul_pd(onetwothreefour, _xscale), _mm256_set1_pd(offset_x_in_world));
                    // add one to y
                    y = _mm256_add_pd(y, _yscale);
                }
                // now we do the maths
                // c = x + yi
                ci = y;
                cr = x;
                zr = _mm256_setzero_pd();
                zi = _mm256_setzero_pd();
                // the iteration count.
                _n = _mm256_setzero_si256();
            repeat:
                // get zr^2
                zr2 = _mm256_mul_pd(zr, zr);
                // get zi^2
                zi2 = _mm256_mul_pd(zi, zi);

                // new_zr = (zr^2 - zi^2) + cr
                // new_zi = 2 * (zr * zi) + ci
                temp_zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
                temp_zi = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(zr, zi), two), ci);
                
                // update
                zr = temp_zr;
                zi = temp_zi;
                // get the norm
                norm = _mm256_add_pd(zr2, zi2);
                
                // Not totally sure what this does, compares first
                _mask1 = _mm256_cmp_pd(norm, four, _CMP_LT_OQ);
                _mask2 = _mm256_cmpgt_epi64(_iterations, _n);
                // cast to integer
                _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));
                _c = _mm256_and_si256(_one, _mask2);  // Zero out ones where n < iterations
                _n = _mm256_add_epi64(_n, _c);        // n++ Increase all n
                if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                    goto repeat;
                
                // then update the iteration count
                iteration_count[i + 0] = int(_n[3]);
                iteration_count[i + 1] = int(_n[2]);
                iteration_count[i + 2] = int(_n[1]);
                iteration_count[i + 3] = int(_n[0]);
                // update x
                x = _mm256_add_pd(x, _mm256_mul_pd(_xscale, four));
            }
        }
    }


    void do_intrinsics_julia() {
#pragma omp parallel
        {
            // how many threads to split up in
            int num_threads = omp_get_num_threads();


            // the scale in the x and y directions
            __m256d _xscale = _mm256_set1_pd(1 / scale.x);
            __m256d _yscale = _mm256_set1_pd(1 / scale.y);
            
            // Some variables
            __m256d zr, zi, cr, ci, temp_zr, temp_zi;
            __m256d zr2, zi2, norm;
            __m256d four, two;
            __m256d x, y, _mask1;
            __m256i _one, _c, _n, _iterations, _mask2;
            __m256d onetwothreefour = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);

            cr = _mm256_set1_pd(-0.8);
            ci = _mm256_set1_pd(0.156);
            // set 4 and 2
            four = _mm256_set1_pd(4.0);
            two = _mm256_set1_pd(2.0);
            _one = _mm256_set1_epi64x(1);

            // What is the top left corner of the screen in world space?
            vec2 test = {0, 0};
            double offset_x_in_world = screen_to_world(test).x;
            double offset_y_in_world = screen_to_world(test).y;
            
            // x = [0, 1, 2, 3] * scale + offset. The 0, 1, 2, 3 is to offset each element of our vector, as well as starting 
            // with the first 4 pixels.
            x = _mm256_add_pd(_mm256_mul_pd(onetwothreefour, _xscale), _mm256_set1_pd(offset_x_in_world));
            // How many iterations are there?
            _iterations = _mm256_set1_epi64x(MAX_ITERS);
            
            // similarly for y, set it as the offset, and the + (bracket) serves to split the screen into rows for each thread.
            y = _mm256_set1_pd(offset_y_in_world + (1 / scale.y * HEIGHT / num_threads * omp_get_thread_num()));
            // Where should we start and end with our loop
            int start = omp_get_thread_num() * HEIGHT / num_threads * WIDTH;
            int end = (omp_get_thread_num() + 1) * HEIGHT / num_threads * WIDTH;

            for (int i = start; i < end; i += 4) {
                if (i % (WIDTH) == 0 && i != 0) {
                    // new row, so reset things
                    // x is again the first four pixels in world space.
                    x = _mm256_add_pd(_mm256_mul_pd(onetwothreefour, _xscale), _mm256_set1_pd(offset_x_in_world));
                    // add one to y
                    y = _mm256_add_pd(y, _yscale);
                }
                // now we do the maths
                zr = x;
                zi = y;
                // the iteration count.
                _n = _mm256_setzero_si256();
            repeat:
                // get zr^2
                zr2 = _mm256_mul_pd(zr, zr);
                // get zi^2
                zi2 = _mm256_mul_pd(zi, zi);

                // new_zr = (zr^2 - zi^2) + cr
                // new_zi = 2 * (zr * zi) + ci
                temp_zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
                temp_zi = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(zr, zi), two), ci);
                
                // update
                zr = temp_zr;
                zi = temp_zi;
                // get the norm
                norm = _mm256_add_pd(zr2, zi2);
                
                // Not totally sure what this does, compares first
                _mask1 = _mm256_cmp_pd(norm, four, _CMP_LT_OQ);
                _mask2 = _mm256_cmpgt_epi64(_iterations, _n);
                // cast to integer
                _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));
                _c = _mm256_and_si256(_one, _mask2);  // Zero out ones where n < iterations
                _n = _mm256_add_epi64(_n, _c);        // n++ Increase all n
                if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                    goto repeat;
                
                // then update the iteration count
                iteration_count[i + 0] = int(_n[3]);
                iteration_count[i + 1] = int(_n[2]);
                iteration_count[i + 2] = int(_n[1]);
                iteration_count[i + 3] = int(_n[0]);
                // update x
                x = _mm256_add_pd(x, _mm256_mul_pd(_xscale, four));
            }
        }
    }
#endif
};

int main(int argc, char** argv) {
    if (argc == 2) {
        WHICH_SET = atoi(argv[1]);
    }
    printf("Running with set = %d\n", WHICH_SET);
    const int size = 2;

    const int WIDTH_IMAGE = WIDTH * size;
    const int HEIGHT_IMAGE = HEIGHT * size;

    Application app;
    sf::RenderWindow window;
    sf::Font font;
    if (!font.loadFromFile("src/arial.ttf")) {
        printf("FONT ERROR\n");
        return 1;
    }
    sf::Text text;
    text.setCharacterSize(48);
    text.setFont(font);

    window.create(sf::VideoMode(WIDTH_IMAGE, HEIGHT_IMAGE), std::string("SFML works!"));

    sf::Event event;

    vec2 start_pan;
    bool is_holding_down = false;

    sf::Sprite sprite;
    sf::Texture tex;

    tex.create(WIDTH_IMAGE, HEIGHT_IMAGE);
    sprite.setTexture(tex);
    std::vector<sf::Uint8> pixels(WIDTH_IMAGE * HEIGHT_IMAGE * 4);
    int time_now = current_microseconds();
    while (window.isOpen()) {
        Timer T("Entire Loop");
        sf::Vector2i _mouse_pos = sf::Mouse::getPosition(window);
        while (window.pollEvent(event)) {
            vec2 mouse = {(double)_mouse_pos.x / size, (double)_mouse_pos.y / size};
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::MouseButtonPressed) {
                start_pan.x = mouse.x;
                start_pan.y = mouse.y;
                is_holding_down = true;
            } else if (event.type == sf::Event::MouseButtonReleased) {
                is_holding_down = false;
            }

            if (is_holding_down) {
                auto d = mouse - start_pan;
                d.x /= app.scale.x;
                d.y /= app.scale.y;
                app.offset = app.offset - (d);
                start_pan.x = mouse.x;
                start_pan.y = mouse.y;
            }

            if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheel.delta > 0) {
                    app.scale *= 1.01;
                } else {
                    app.scale *= 0.99;
                }
            }

            vec2 mouse_in_world_before_zoom = app.screen_to_world(mouse);

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Key::Q) {
                    app.scale *= 1.1;
                } else if (event.key.code == sf::Keyboard::Key::A) {
                    app.scale *= 0.9;
                } else if (event.key.code == sf::Keyboard::Key::N) {
                    MAX_ITERS += 32;
                } else if (event.key.code == sf::Keyboard::Key::M) {
                    MAX_ITERS = std::max(32, MAX_ITERS - 32);
                }
            }
            vec2 mouse_in_world_after_zoom = app.screen_to_world(mouse);
            auto diff = mouse_in_world_before_zoom - mouse_in_world_after_zoom;
            app.offset += diff;
        }

        window.clear();
        {
            Timer t("Loop Print");
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int x = 0; x < WIDTH; ++x) {
                for (int y = 0; y < HEIGHT; ++y) {
                    int index_other = (y)*WIDTH + x;
                    float a = 0.1;
                    float n = (float)app.iteration_count[index_other];

                    float r = 0.5f * sin(a * n) + 0.5f;
                    float g = 0.5f * sin(a * n + 2.094f) + 0.5f;
                    float b = 0.5f * sin(a * n + 4.188f) + 0.5f;
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            int index = (y * size + i) * WIDTH_IMAGE + (x * size + j);
                            pixels[index * 4 + 0] = (int)(r * 255);
                            pixels[index * 4 + 1] = (int)(g * 255);
                            pixels[index * 4 + 2] = (int)(b * 255);
                            pixels[index * 4 + 3] = 255;
                        }
                    }
                }
            }
        }
        double seconds_to_generate;
        {
            Timer t("Update vec");
            int s = current_microseconds();
            app.update_vec();
            int e = current_microseconds();
            seconds_to_generate = (double)(e - s) / 1e6;
        }
        {
            Timer t("Update tex");
            tex.update(pixels.data());
        }
        window.draw(sprite);
        int new_time = current_microseconds();
        window.setTitle("FPS: " + std::to_string(1.0 / ((new_time - time_now) / (float)1e6)));
        window.draw(text);
        // print stats
        text.setString("Scale: " + std::to_string(app.scale.x) + " log10 = " + std::to_string(log10(app.scale.x)) + "\tZoom in and out using Q and A" +
                       "\nOffset: " + std::to_string(app.offset.x) + "," + std::to_string(app.offset.y) + "\tPan using the mouse" +
                       "\nMaximum iterations: " + std::to_string(MAX_ITERS) + "\tIncrease / Decrease using N and M" +
                       "\nTime taken to generate the iterations: " + std::to_string(seconds_to_generate)

        );
        window.display();
        time_now = current_microseconds();
    }

    return 0;
}
