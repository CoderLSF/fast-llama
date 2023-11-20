
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

    const char* get_color(Color c) const noexcept ;
    const char* get_color(int c) const noexcept {
        return get_color(Color(c % int(Color::MAX)));
    }

    const char* red() const noexcept {
        return get_color(Color::RED);
    }
    const char* green() const noexcept {
        return get_color(Color::GREEN);
    }
    const char* yellow() const noexcept {
        return get_color(Color::YELLOW);
    }
    const char* blue() const noexcept {
        return get_color(Color::BLUE);
    }
    const char* purple() const noexcept {
        return get_color(Color::PURPLE);
    }

    const char* get_endtag() const noexcept;
    const char* endtag() const noexcept { // alias
        return get_endtag();
    }
private:
    bool _enabled = true;
};

} // namespace cpuft
