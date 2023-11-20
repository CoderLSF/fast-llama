/******************************************************************
 @Author: Liu Shaofeng
 @Date: 2023/10/17
 ******************************************************************/

#include "tensor.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <cstdint>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <vector>

#include "utility.h"
#include "base_types.h"
#include "malloc_utility.h"
#include "tf_operators.h"
#include "quant_operators.h"
#include "log.h"

namespace cpuft {

Tensor::Tensor(const Tensor& other) {
    like(other);
    if (MemoryType(other._mtype) == MemoryType::MANAGED) {
        _data = other._data;
        _scales = other._scales;
        return;
    }
    if (other._data != nullptr) [[likely]] {
        if (!reserve_memory()) {
            throw std::runtime_error("Out of memory");
        }
        memcpy(_data, other._data, data_memory_size());
        if (_scales != nullptr) {
            memcpy(_scales, other._scales, scales_memory_size());
        }
    }
}
Tensor& Tensor::operator=(const Tensor& other) {
    if (MemoryType(_mtype) == MemoryType::MANAGED && memory_size() < other.memory_size()) {
        throw std::runtime_error("Insufficient managed memory.");
    }
    if (!is_same_shape(other) || _qtype != other._qtype) {
        if (memory_size() < other.memory_size()) {
            release();
        }
        _columns = other._columns;
        _rows    = other._rows;
        _layers  = other._layers;
        _qtype   = other._qtype;
        _qgsize  = other._qgsize;
    }
    if (other._data == nullptr) {
        return *this;
    }
    if (!reserve_memory()) {
        throw std::runtime_error("out of memory");
    }

    memcpy(_data, other._data, data_memory_size());
    if (_scales != nullptr) {
        memcpy(_scales, other._scales, scales_memory_size());
    }
    return *this;
}

bool Tensor::read_data(std::ifstream& file, ssize_t data_offset, ssize_t scales_offset) noexcept {
    if (!reserve_memory()) {
        tf_log_error("Out of memory for tensor");
        return false;
    }

    if (data_offset >= 0) {
        file.seekg(data_offset, std::ios::beg);
    }
    if (!file.read(_data, data_memory_size())) {
        tf_log_error("Reading tensor data error");
        return false;
    }
    if (is_quantized()) {
        if (scales_offset >= 0) {
            file.seekg(scales_offset, std::ios::beg);
        }
        if (!file.read(reinterpret_cast<char*>(_scales), scales_memory_size())) {
            tf_log_error("Reading tensor scales error");
            return false;
        }
    }

    return true;
}

Tensor Tensor::copy(MemoryType mt) const noexcept {
    if (mt == MemoryType::NONE) {
        mt = MemoryType(_mtype);
    }
    Tensor t(shape(), QuantType(_qtype), _qgsize);
    t._mtype = uint8_t(mt);
    if (MemoryType(_mtype) == MemoryType::MANAGED) {
        t._data = _data;
        t._scales = _scales;
        return t;
    }
    if (t.reserve_memory()) {
        memcpy(t._data, _data, data_memory_size());
        if (t._scales != nullptr && _scales != nullptr) {
            memcpy(t._scales, _scales, scales_memory_size());
        }
    }
    return t;
}

bool Tensor::copy(Tensor& t, const Tensor& s,
            int target_start_column, int source_start_column, int columns) noexcept {
    if (s._data == nullptr || source_start_column >= s._columns || target_start_column >= t._columns) [[unlikely]] {
        tf_log_debug("No need to copy");
        return true;
    }
    if (s._qtype != t._qtype || (s.is_quantized() && (s._qgsize != t._qgsize
            || (source_start_column % s._qgsize) != 0 || (target_start_column % t._qgsize) != 0))) [[unlikely]] {
        tf_log_error("Invalid parameters for Tensor::copy() t.qtype:%-8s s.qtype:%-8s t.group_size:%-2d s.group_size:%-2d target_start_column:%-5d source_start_column:%-5d",
                t.quant_type_name(), s.quant_type_name(), int(t._qgsize), int(s._qgsize), target_start_column, source_start_column);
        return false;
    }
    if (!t.reserve_memory()) [[unlikely]] {
        tf_log_debug("Out of memory for Tensor::copy()");
        return false;
    }
    if (source_start_column == 0 && target_start_column == 0 && columns == s._columns && columns == t._columns) {
        memcpy(t._data, s._data, s.memory_size());
        if (t.is_quantized()) {
            memcpy(t._scales, s._scales, s.scales_memory_size());
        }
        return true;
    }
    if (columns < 0) {
        columns = t._columns;
    }

    int rows = cpuft::min(s.total_rows(), t.total_rows());
    columns = cpuft::min(s._columns - source_start_column, t._columns - target_start_column, int(columns));
    auto src = s._data + s.size_to_offset(source_start_column);
    auto tgt = t._data + t.size_to_offset(target_start_column);
    const int t_row_size = t.size_to_offset(t._columns);
    const int s_row_size = s.size_to_offset(s._columns);
    for (int i = 0; i < rows; ++i) {
        memcpy(tgt + t_row_size * i, src + s_row_size * i, s.size_to_offset(columns));
    }
    if (t.is_quantized()) {
        auto src = s._scales + source_start_column / s._qgsize;
        auto tgt = t._scales + target_start_column / s._qgsize;
        for (int i = 0; i < rows; ++i) {
            memcpy(tgt + t._columns*i/s._qgsize, src + s._columns*i/s._qgsize, sizeof(float)*columns/s._qgsize);
        }
    }
    return true;
}

void Tensor::clip(int start_columns, int end_columns) noexcept {
    if (end_columns < 0) {
        end_columns = _columns;
    }
    if (start_columns >= end_columns) {
        return;
    }
    int cols = end_columns - start_columns;
    if (_data == nullptr) {
        _columns = cols;
        return;
    }

    if (start_columns > 0) {
        memcpy(_data, _data + start_columns, size_to_offset(cols));
        if (_scales != nullptr) {
            memcpy(_scales, _scales + sizeof(float) * cols / _qgsize, sizeof(float) * cols / _qgsize);
        }
    }

    int drsz = size_to_offset(_columns);
    int dcsz = size_to_offset(cols);

    auto p1 = _data;
    auto p2 = _data + size_to_offset(start_columns);
    for (int i = 1, n = total_rows(); i < n; ++i) {
        memcpy(p1 + dcsz * i, p2 + drsz*i, dcsz);
    }
    if (_scales != nullptr) {
        int drsz = _columns / _qgsize;
        int dcsz = cols / _qgsize;
        auto p1 = _scales;
        auto p2 = _scales + start_columns/_qgsize;
        for (int i = 1, n = total_rows(); i < n; ++i) {
            memcpy(&p1[dcsz*i], &p2[drsz*i], sizeof(float)*dcsz);
        }
    }
    _columns = cols;
}

bool Tensor::zero() noexcept {
    if (!reserve_memory()) {
        return false;
    }
    memset(_data, 0, data_memory_size());
    if (is_quantized() && _scales != nullptr) {
        memset(_scales, 0, scales_memory_size());
    }
    return true;
}
bool Tensor::random(float max_value, float min_value) noexcept {
    if (!reserve_memory()) [[unlikely]] {
        return false;
    }
    quant::fill_random(QuantType(_qtype), _data, _scales, size(), _qgsize, min_value, max_value);
    return true;
}

void Tensor::print(std::string_view prefix, bool show_attributes, bool pretty_print, int display_num, int precision, int column_start, int column_end) const noexcept {
    if (prefix.size() > 0) {
        std::cout << std::setw(15) << prefix;
    }
    std::cout << display_string(show_attributes, pretty_print, display_num, precision, column_start, column_end) << std::endl;
}

static std::string build_summary(Tensor t, size_t display_num, int precision) {
    size_t size = t.size();
    if (display_num > size) {
        display_num = size;
    }
    if (display_num == 0) [[unlikely]] {
        return size > 0 ? "[...]" : "[]";
    }

    std::ostringstream oss;
    auto outf = [&oss, precision](float v) {
        if (precision >= 0) {
            oss << std::setw(precision+4) << std::setprecision(precision) << std::fixed << v;
            return;
        }

        auto abs = fabsf(v);
        oss << std::setw(10);
        if (abs == 0.) {
            oss << std::setprecision(0) << std::fixed;
        } else if (abs >= 1e3 && abs < 1e8) {
            oss << std::setprecision(0) << std::fixed;
        } else if (abs > 1e8 || abs <= 1e-4) {
            oss << std::setprecision(3) << std::scientific;
        } else if (abs > 1e-4 && abs < 10.0) {
            oss << std::setprecision(6) << std::fixed;
        } else {
            oss << std::setprecision(4) << std::fixed;
        }
        oss << v;
    };
    oss << "[";
    if (size > display_num) {
        for (size_t i = 0; i < display_num-2; ++i) {
            outf(t.get_value(i));
            oss << ", ";
        }
        oss << "..., ";
        outf(t.get_value(size-2));
        oss << ", ";
    } else {
        for (size_t i = 0; i < size-1; ++i) {
            outf(t.get_value(i));
            oss << ", ";
        }
    }
    outf(t.get_value(size-1));
    oss << "]";
    return oss.str();
}

std::string Tensor::display_string(bool show_attributes, bool pretty_print, int display_num, int precision, int column_start, int column_end) const noexcept {
    if (display_num < 4) {
        display_num = 4;
    }
    if (column_end <= 0) {
        column_end = _columns;
    }

    auto n = size();
    std::ostringstream oss;
    if (show_attributes) {
        if (pretty_print) {
            oss << "\nquantization: " << quant_type_name() << "\n";
            if (is_quantized()) {
                oss << "  group size: " << _qgsize << "\n";
            }
            oss << "       shape: " << shape().serialize() << "\n";
            oss << " memory type: " << memory_type_name() << "\n";
            oss << "        data: ";
        } else {
            oss << "quant_type:" << quant_type_name() << ", mem_type:" << memory_type_name()
                << ", shape:" << shape().serialize() << ", data:";
        }
    } else {
        oss << "shape:" << shape().serialize() << ", data:";
    }
    if (_data == nullptr || n < 1) {
        oss << "NULL";
    } else {
        auto nrows = total_rows();
        if (nrows > 1) {
            oss << "[\n";
        }
        auto hrows = nrows <= 8 ? nrows : 6;
        for (int i = 0; i < hrows && i < nrows; ++i) {
            if (i > 0) {
                oss << "\n";
            }
            if (nrows > 1) {
                oss << std::setw(4) << i << ":";
            }
            oss << build_summary(row_slice(i).slice(column_start, column_end), display_num, precision);
        }
        if (hrows < nrows) {
            oss << "\n     ...\n";
            oss << std::setw(4) << nrows-1 << ":" << build_summary(row_slice(nrows-1).slice(column_start, column_end), display_num, precision);
        }
        if (nrows > 1) {
            oss << "\n]";
        }
    }
    return oss.str();
}

void Tensor::release() noexcept {
    if (_data == nullptr) [[unlikely]] {
        return;
    }
    free(MemoryType(_mtype), _data, memory_size());
    _data = nullptr;
    _scales = nullptr;
}

std::string TensorShape::serialize() const noexcept {
    std::ostringstream oss;
    oss << "(" << std::setw(5) << columns;
    if (rows > 0) {
        oss << ", " << std::setw(5) << rows;
    }
    if (layers > 0) {
        oss << ", " << std::setw(5) << layers;
    }
    oss << ")";
    return oss.str();
}

size_t Tensor::manage(void* data, float* scales) noexcept {
    release();
    _mtype = uint8_t(MemoryType::MANAGED);
    _data  = reinterpret_cast<char*>(data);
    if (scales != nullptr) {
        _scales = scales;
    } else {
        _scales = reinterpret_cast<float*>(_data + data_memory_size());
    }
    return memory_size();
}

char* Tensor::alloc(MemoryType mt, size_t size) noexcept {
    if (mt == MemoryType::NUMA) {
        mt = MemoryType::NORMAL;
        return (char*)numa_malloc(size);
    }
    if (mt == MemoryType::NORMAL) {
        return (char*)aligned_malloc(size, 64);
    } else if (mt == MemoryType::GPU) {
        fprintf(stderr, "GPU Memory is not supported yet.");
        return nullptr;
    }
    return nullptr;
}

void Tensor::free(MemoryType mt, char* ptr, size_t size) noexcept {
    if (mt == MemoryType::NORMAL || mt == MemoryType::NONE) {
        aligned_free(ptr);
    } else if (mt == MemoryType::NUMA) {
        numa_free(ptr, size);
    }
}

bool Tensor::set_memory_type(MemoryType mt) noexcept {
    if (_mtype == uint8_t(mt)) {
        return true;
    } else if (_data == nullptr) {
        _mtype = uint8_t(mt);
        return true;
    }

    auto mem_size = memory_size();
    auto p = alloc(MemoryType(_mtype), mem_size);
    if (p == nullptr) {
        fprintf(stderr, "OOM! Out of memory for the tensor.");
        return false;
    }

    memcpy(p, _data, mem_size);
    _data = p;
    update_scales_pointer();
    return true;
}

bool Tensor::reserve_memory(MemoryType mt) noexcept {
    if (_mtype == uint8_t(MemoryType::NONE)) {
        _mtype = uint8_t(MemoryType::NORMAL);
    } else if (_data != nullptr) {
        if (mt == MemoryType::NONE || mt == MemoryType(_mtype)) {
            return true;
        }
        return set_memory_type(mt);
    }
    if (mt != MemoryType::NONE) {
        _mtype = uint8_t(mt);
    }
    if (_mtype == uint8_t(MemoryType::MANAGED)) {
        return false;
    }
#ifdef DISABLE_NUMA
    if (_mtype == uint8_t(MemoryType::NUMA)) {
        _mtype = uint8_t(MemoryType::NORMAL);
    }
#endif

    auto mem_size = memory_size();
    _data = alloc(MemoryType(_mtype), mem_size);
    update_scales_pointer();
    if (_data == nullptr) {
        ft_log_error("OOM! Out of memory for the tensor.");
        return false;
    }
    return true;
}

bool Tensor::quantize(const Tensor& t) noexcept {
    if (_mtype == uint8_t(MemoryType::NONE)) {
        _columns = t._columns;
        _rows    = t._rows;
        _layers  = t._layers;
    } else if (_columns != t._columns || total_rows() != t.total_rows()) {
        ft_log_debug("not the same shape. (%d, %d, %d) vs (%d, %d, %d)",
            _columns, _rows, _layers, t._columns, t._rows, t._layers);
        return false;
    }

    return quantize(std::span{t.float_data(), t.size()});
}

bool Tensor::quantize(std::span<const float> data) noexcept {
    if (size() < 1) {
        _columns = data.size();
        _rows = 0;
        _layers = 0;
    }
    if (data.size() != size()) {
        fprintf(stderr, "Invalid data size for quantize");
        return false;
    }

    if (!is_quantized()) {
        release();
        _qtype = uint8_t(QuantType::INT16);
        _qgsize = QUANT_GROUP_SIZE;
    }
    if (!reserve_memory()) {
        return false;
    }

    quant::quantize(QuantType(_qtype), _data, _scales, data.data(), data.size(), _qgsize);
    return true;
}

bool Tensor::quantize(std::span<const float> data, int layer, int row_offset) noexcept {
    if (data.size() < 1 || layer < 0 || row_offset < 0) {
        return false;
    }
    size_t offset = size_t(_columns) * (_rows * layer + row_offset);
    size_t length = cpuft::min(data.size(), size_t(_rows - row_offset));
    if (length < 1) {
        return false;
    }

    if (!is_quantized()) {
        release();
        _qtype = uint8_t(QuantType::INT16);
        _qgsize = QUANT_GROUP_SIZE;
    }
    if (!reserve_memory()) {
        return false;
    }

    quant::quantize(QuantType(_qtype), get_data(offset), get_scales(offset), data.data(), length, _qgsize);
    return true;
}

bool Tensor::dequantize(const Tensor& b) noexcept {
    if (is_quantized() || !is_same_shape(b) || b._data == nullptr) [[unlikely]] {
        return false;
    }

    if (!reserve_memory()) {
        ft_log_debug("out of memory");
        return false;
    }
    quant::dequantize(QuantType(b._qtype), float_data(), b._data, b._scales, size(), b._qgsize);
    return true;
}

float Tensor::sum_(size_t offset, size_t length) const noexcept {
    if (_data == nullptr) {
        return 0.;
    }
    return quant::sum(QuantType(_qtype), get_data(offset), get_scales(offset), length, _qgsize);
}

float Tensor::sum() const noexcept {
    if (_data == nullptr) {
        return 0.;
    }
    return quant::sum(QuantType(_qtype), _data, _scales, size(), _qgsize);
}
float Tensor::square_sum() const noexcept {
    if (_data == nullptr) {
        return 0.;
    }
    return quant::square_sum(QuantType(_qtype), _data, _scales, size(), _qgsize);
}

float Tensor::max_(size_t offset, size_t length) const noexcept {
    if (_data == nullptr) {
        return FLT_MIN;
    }
    return quant::max(QuantType(_qtype), get_data(offset), get_scales(offset), length, _qgsize);
}

float Tensor::min_(size_t offset, size_t length) const noexcept {
    if (_data == nullptr) {
        return FLT_MAX;
    }
    return quant::min(QuantType(_qtype), get_data(offset), get_scales(offset), length, _qgsize);
}

void Tensor::matmul(const Tensor& b, Tensor& out, int layer) const {
    if (b._qtype != _qtype || b._qgsize != _qgsize) {
        ft_log_error("this.quant_type:%s\tb.quant_type:%s\tthis.group_size:%d\tb.group_size:%d",
            quant_type_name(), b.quant_type_name(), _qgsize, b._qgsize);
        throw std::runtime_error("b should be the same quantization type for matmul()");
    }
    if (out.is_quantized()) {
        throw std::runtime_error("result tensor should not be quantized for matmul()");
    }
    if (_data == nullptr || b._data == nullptr) {
        ft_log_error("Memory is not initialized yet. _data:%p, b._data:%p\n", _data, b._data);
        throw std::runtime_error("Memory is not initialized yet.");
    }

    size_t offset = 0;
    if (_layers > 0) {
        if (layer >= _layers) {
            throw std::runtime_error("layer exceeds for Tensor::matmul()");
        }
        while (layer < 0) layer += _layers;
        offset = layer == 0 ? 0 : size_t(_columns) * _rows * layer;
    }

    if (!out.reserve_memory()) {
        throw std::runtime_error("OOM! Out of memory for saving results of matmul().");
    }

    return quant::matmul(QuantType(_qtype), out.float_data(),
                get_data(offset), get_scales(offset), b.get_data(), b.get_scales(),
                total_rows(), _columns, b.total_rows(), _qgsize);
}

void  Tensor::softmax(int columns) noexcept {
    if (_data == nullptr) [[unlikely]] {
        return;
    }
    char*  qx = _data;
    float* qs = _scales;
    float* px;
    if (is_quantized()) {
        px = new float[_columns];
    } else {
        px = reinterpret_cast<float*>(_data);
    }
    for (int r = 0, n = total_rows(); r < n; ++r) {
        if (is_quantized()) {
            quant::dequantize(QuantType(_qtype), px, qx, qs, columns, _qgsize);
            cpuft::softmax_sisd(px, columns);
            if (columns < _columns) [[unlikely]] {
                memset(px + columns, 0, sizeof(float) * (_columns - columns));
            }
            quant::quantize(QuantType(_qtype),   qx, qs, px, columns, _qgsize);
            qx += size_to_offset(_columns);
            qs += _columns / _qgsize;
        } else {
            cpuft::softmax_sisd(px, columns);
            if (columns < _columns) {
                memset(px + columns, 0, sizeof(float) * (_columns - columns));
            }
            px += _columns;
        }
    }
    if (is_quantized()) {
        delete []px;
    }
}

void  Tensor::rmsnorm(const Tensor& x, const Tensor& w) {
    if (_data == nullptr) [[unlikely]] {
        return;
    }
    if (w._columns != _columns || w._rows > 1) {
        ft_log_error("Invalid parameters for Tensor::rmsnorm() w._columns:%d, _columns:%d, w._rows:%d.",
                w._columns, _columns, w._rows);
        throw std::runtime_error("invalid input for Tensor::rmsnorm()");
    }
    if (is_quantized()) {
        throw std::runtime_error("Tensor::rmsnorm() does not support quantization yet");
    }

    auto po = float_data();
    auto px = x.float_data();
    for (int r = 0, n = total_rows(); r < n; ++r) {
        cpuft::rmsnorm(po, px, w.float_data(), _columns);
        po += _columns;
        px += _columns;
    }
}

void Tensor::silu() {
    if (is_quantized()) [[unlikely]] {
        throw std::runtime_error("Quantized tensor is not supported yet.");
    }

    auto po = float_data();
    for (int i = 0, n = total_rows(); i < n; ++i) {
        cpuft::silu(po + _columns * i, _columns);
    }
}

void Tensor::swiglu(const Tensor& x) {
    if (!is_same_shape(x)) [[unlikely]] {
        throw std::runtime_error("The input tensor should have the same shape");
    }
    if (_qtype != x._qtype || is_quantized()) [[unlikely]] {
        throw std::runtime_error("Quantized tensor is not supported yet.");
    }

    auto po = float_data();
    auto px = x.float_data();
    for (int i = 0, n = total_rows(); i < n; ++i) {
        cpuft::swiglu(po + _columns * i, px + _columns * i, _columns);
    }
}

void Tensor::inner_swiglu() {
    if ((_columns & 1) != 0) [[unlikely]] {
        throw std::runtime_error("columns should be a even number.");
    }
    if (is_quantized()) [[unlikely]] {
        throw std::runtime_error("Quantized tensor is not supported yet.");
    }
    auto px  = float_data();
    for (int i = 0, n = total_rows(); i < n; ++i, px += _columns) {
        cpuft::swiglu(px, px + _columns/2, _columns/2);
    }
}

void Tensor::weighted_sum(std::span<const float> weights, Tensor& out, int layer, float weight_threshold) const {
    if (weights.empty() || (_layers > 0 && layer >= _layers)) [[unlikely]] {
        return;
    }
    if (out.is_quantized()) [[unlikely]] {
        throw std::runtime_error("result tensor should not be quantized for matmul()");
    }
    if (_layers > 0 && layer < 0) [[unlikely]] {
        layer += _layers;
    }
    if (!out.reserve_memory()) [[unlikely]] {
        throw std::runtime_error("OOM! Out of memory for saving results of matmul().");
    }
    return quant::weighted_sum(QuantType(_qtype), out.float_data(), reinterpret_cast<const void*>(_data), _scales, _qgsize, weights.data(), weights.size(), _columns, weight_threshold);
}

void  Tensor::weighted_sum(const Tensor& weights, Tensor& out, int layer, float weight_threshold) const {
    if ((_layers > 0 && layer >= _layers) || out.rows() != weights.rows() || out.columns() != _columns) [[unlikely]] {
        throw std::runtime_error("invalid parameters");
    }
    if (out.is_quantized()) [[unlikely]] {
        throw std::runtime_error("result tensor should not be quantized for weighted_sum()");
    }
    if (_layers > 0 && layer < 0) {
        layer += _layers;
    }
    if (!out.reserve_memory()) [[unlikely]] {
        throw std::runtime_error("OOM! Out of memory for result of weighted_sum().");
    }
    return quant::weighted_sum(QuantType(_qtype), out.float_data(), reinterpret_cast<const void*>(_data), _scales, _qgsize, weights.float_data(), _rows, _columns, out.rows(), weight_threshold);
}

void Tensor::add(float value) noexcept {
    if (_data == nullptr && !reserve_memory()) {
        return;
    }
    return quant::add(QuantType(_qtype), _data, _scales, value, size(), _qgsize);
}

void Tensor::add(const Tensor& b, int this_column_offset) noexcept {
    if (b._data == nullptr || this_column_offset >= _columns) [[unlikely]] {
        return;
    }
    int cols = cpuft::min(_columns - this_column_offset, int(b._columns));
    int rows = cpuft::min(total_rows(), b.total_rows());

    if (_data == nullptr) [[unlikely]] {
        if (reserve_memory()) {
            zero();
            copy_from(b, this_column_offset);
        }
    } else if (!is_quantized() && !b.is_quantized()) {
        if (this_column_offset == 0 && _columns == b._columns) {
            return cpuft::add(float_data(), b.float_data(), _columns * rows);
        } else {
            auto pd1 = float_data() + this_column_offset;
            for (int i = 0; i < rows; ++i) {
                cpuft::add(pd1 + _columns * i, b.float_data(b._columns*i), cols);
            }
        }
    } else if (_qtype != b._qtype || _qgsize != b._qgsize) {
        ft_log_error("Should be same type of quantization for Tensor::add");
    } else if (this_column_offset == 0 && _columns == b._columns) {
        quant::add(QuantType(_qtype), _data, _scales, b._data, b._scales, _columns * rows, _qgsize);
    } else {
        if (this_column_offset % _qgsize != 0 || cols % _qgsize != 0) [[unlikely]] {
            ft_log_error("column offset or columns should be multiple of group size for Tensor::add");
            return;
        }
        auto pd1 = _data + size_to_offset(this_column_offset);
        auto ps1 = _scales + this_column_offset / _qgsize;
        auto pd2 = b._data;
        auto ps2 = b._scales;
        for (int i = 0; i < rows; ++i) {
            quant::add(QuantType(_qtype), pd1, ps1, pd2, ps2, cols, _qgsize);
            pd1 += size_to_offset(_columns);
            ps1 += _columns / _qgsize;
            pd2 += b.size_to_offset(b._columns);
            ps2 += b._columns / _qgsize;
        }
    }
}

Tensor& Tensor::reshape(int columns, int rows, int layers) {
    if (columns == _columns && rows == _rows && layers == _layers) [[unlikely]] {
        return *this;
    }
    size_t sz = size();
    if (columns <= 0) {
        columns = _columns;
    }
    if (rows < 0) {
        if (size_t(columns) == sz) {
            rows = _rows > 0 ? 1 : 0;
        } else {
            rows = int(size_t(_columns) * (_rows <= 1 ? 1 : _rows) / size_t(columns));
        }
    }
    if (layers < 0) {
        if (rows == 0) {
            layers = 0;
        } else if (auto v = size_t(columns * rows); v != sz) {
            layers = int(v);
        } else {
            layers = _layers > 0 ? 1 : 0;
        }
    }
    if (size_t(columns) * (rows <= 1 ? 1 : rows) * (layers <= 1 ? 1 : layers) != sz) {
        throw std::runtime_error("invalid parameter for reshape(). must be the same size.");
    }
    _columns = columns;
    _rows    = rows;
    _layers  = layers;
    return *this;
}

} // namespace cpuft

