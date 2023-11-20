#pragma once

#if defined(__linux__) || defined(__APPLE__) 

    #include <cfloat>

#elif defined(_WIN32) || defined(__CYGWIN__)

#ifndef M_PI
constexpr float M_PI = 3.1415926536;
#endif

#ifndef INT_MAX
constexpr int INT_MAX = 0x7FFFFFFF;
#endif
#ifndef INT_MIN
constexpr int INT_MIN = ~0;
#endif

#endif
