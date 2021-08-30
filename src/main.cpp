#define WIDTH 800
#define HEIGHT 800
#define USE_OMP
// #define DEBUG
int MAX_ITERS = 128;
int WHICH_SET = 0;
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
    vec2 scale = {100, 100}, offset = {-WIDTH / 2, -HEIGHT / 2};
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
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int vec_index = 0; vec_index < WIDTH * HEIGHT; ++vec_index) {
            int x = vec_index % WIDTH;
            int y = vec_index / HEIGHT;
            vec2 screen_pos = {(float)x, (float)y};
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
};

int main(int argc, char** argv) {
    if (argc == 2) {
        WHICH_SET = atoi(argv[1]);
    }
    printf("Running with set = %d\n", WHICH_SET);
    const int size = 4;

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
        vec2 mouse = {(float)_mouse_pos.x / size, (float)_mouse_pos.y / size};
        while (window.pollEvent(event)) {
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
        text.setString("Scale: " + std::to_string(app.scale.x) + "\tZoom in and out using Q and A"+
                       "\nOffset: " + std::to_string(app.offset.x) + "," + std::to_string(app.offset.y) + "\tPan using the mouse" +
                       "\nMaximum iterations: " + std::to_string(MAX_ITERS) + "Increase / Decrease using N and M" + 
                       "\nTime taken to generate the iterations: " + std::to_string(seconds_to_generate)

        );
        window.display();
        time_now = current_microseconds();
    }

    return 0;
}
