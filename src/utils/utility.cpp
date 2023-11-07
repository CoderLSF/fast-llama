/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 @author: Liu Shaofeng
 @date: 2023/10/16
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "float.h"
#include "math.h"

#include <string>
#include <string_view>

#include "utility.h"

namespace cpuft {

std::vector<std::string_view> ssplit(const std::string& s, std::string_view delim) {
    std::vector<std::string_view> out;
    size_t pos = 0; 
    while (pos < s.length()) {
        auto next = s.find(delim, pos);
        if (next == std::string::npos) {
            out.emplace_back(s.c_str() + pos, s.length() - pos);
            break;
        }
        out.emplace_back(s.c_str() + pos, next - pos);
        pos = next + 1;
    }
    return out;
}
int stoi(std::string_view s) {
    int res = 0;
    for (auto c : s) {
        if (c >= '0' && c <= '9') {
            res = res * 10 + int(c - '0');
        }
    }
    return res;
}

float float16_to_float32(uint16_t f16) {
    uint16_t sign = f16 & (1<<15);
    if ((f16 & (sign - 1)) == 0) {
        return 0.;
    }

    union {
        float       val;
        uint32_t    f32;
    } res;

    res.f32 = (uint32_t)(f16 & 0x8000) << 16;
    uint32_t frac = f16 & 1023u;
    uint32_t expo = (f16 >> 10) & 0x1F;
    if (expo == 0) {
        int clz = __builtin_clz(frac) - 21;
        expo = clz + 14;
        res.f32 |= sign << 16 | ((uint32_t)(127 - expo) << 23) | ((frac << (clz+13)) & ((1u << 23) - 1));
    } else if (expo == 31) {
        return frac == 0 ? (res.f32 == 0 ? FLT_MAX : FLT_MIN) : NAN;
    } else {
        expo = 112 + expo;
        res.f32 |= ((uint32_t)(f16 & 0x8000) << 16) | (expo << 23) | (frac << 13);
    }
    return res.f32;
}

} // namespace cpuft
