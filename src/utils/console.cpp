
#include "console.h"

namespace cpuft {

constexpr const char* g_bash_colors[] = {
    "",         // NONE
    "\033[31m", // RED
    "\033[32m", // GREEN
    "\033[33m", // YELLOW
    "\033[34m", // BLUE
    "\033[35m", // PURPLE
};

const char* Console::get_color(Color c) const noexcept {
    #ifdef _WIN32
    return "";
    #else
    return _enabled ? g_bash_colors[int(c) % int(Color::MAX)] : "";
    #endif
}

const char* Console::get_endtag() const noexcept {
    #ifdef _WIN32
    return "";
    #else
    return _enabled ? "\x1b[0m" : "";
    #endif
}

}
