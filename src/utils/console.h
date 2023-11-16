
#pragma once


namespace cpuft {

enum class Color {
    NONE   = 0,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    PURPLE,

    MAX
};

class Console {
public:
    Console(bool enabled=true) : _enabled(enabled) {}

    const char* get_color(Color c);
    const char* get_color(int c) {
        return get_color(Color(c % int(Color::MAX)));
    }

    const char* get_endtag();
private:
    bool _enabled = true;
};

} // namespace cpuft
