
#include "console.h"

namespace cpuft {

constexpr const char* g_bash_colors[] = {
    "",         // NONE
    "\x1b[31m", // RED
    "\x1b[32m", // GREEN
    "\x1b[33m", // YELLOW
    "\x1b[34m", // BLUE
    "\x1b[35m", // PURPLE
};

const char* Console::get_color(Color c) {
    return _enabled ? g_bash_colors[int(c) % int(Color::MAX)] : "";
}

const char* Console::get_endtag() {
    return _enabled ? "\x1b[0m" : "";
}

}
