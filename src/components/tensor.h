/*************************************************************************************
 @Author: Liu Shaofeng
 @Date:   2023/10/22
**************************************************************************************/

#pragma once

#include <utility>
#include <string>
#include <span>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "base_types.h"
#include "tf_operators.h"
#include "quant_operators.h"
#include "alignmem.hpp"

namespace cpuft {

using quant::QuantType;

// 内存类型
enum class MemoryType {
    NONE = 0,
    NORMAL,     // 用aligned_alloc申请的普通内存
    NUMA,       // numa_alloc申请的内存
    MANAGED,    // 外部传入的托管内存，析构时无需释放
    GPU,        // 显存，暂未支持

    _MAX_,
};

struct TensorShape {
    int columns = 1;
    int rows = 0;
    int layers = 0;

    bool operator==(const TensorShape& other) const noexcept {
        return columns == other.columns && rows == other.rows && layers == other.layers;
    }
    std::string serialize() const noexcept;
};

class Tensor {
    static constexpr int QUANT_GROUP_SIZE = 64;
public:
    /*********************************< constructors and destructors >********************************/
    ~Tensor() noexcept { release(); }
    Tensor() {}
    explicit Tensor(MemoryType mt) : _mtype(uint8_t(mt)) {}
    Tensor(int columns, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) :
            _columns(columns), _rows(0), _layers(0), _qgsize(quant_group_size) {
        _qtype = uint8_t(qt);
    }
    Tensor(int columns, int rows, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) :
            _columns(columns), _rows(rows), _layers(0), _qgsize(quant_group_size) {
        _qtype = uint8_t(qt);
    }
    Tensor(int columns, int rows, int layers, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) :
            _columns(columns), _rows(rows), _layers(layers), _qgsize(quant_group_size) {
        _qtype = uint8_t(qt);
    }
    Tensor(const TensorShape& shape, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) :
            _columns(shape.columns), _rows(shape.rows), _layers(shape.layers), _qgsize(quant_group_size) {
        _qtype = uint8_t(qt);
    }

    Tensor(Tensor&& other) noexcept { move(std::move(other)); }
    Tensor& operator=(Tensor&& other) noexcept { return move(std::move(other)); }
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    inline static Tensor manage(void* data, int columns, int rows, int layers, QuantType qt=QuantType::NONE, float* scales=nullptr, int group_size=QUANT_GROUP_SIZE) noexcept {
        Tensor t(columns, rows, layers, qt, group_size);
        t.manage(data, scales);
        return t;
    }
    inline static Tensor manage(void* data, int columns, int rows, QuantType qt=QuantType::NONE, float* scales=nullptr, int group_size=QUANT_GROUP_SIZE) noexcept {
        return Tensor::manage(data, columns, rows, 0, qt, scales, group_size);
    }
    inline static Tensor manage(void* data, int columns, QuantType qt=QuantType::NONE, float* scales=nullptr, int group_size=QUANT_GROUP_SIZE) noexcept {
        return Tensor::manage(data, columns, 0, 0, qt, scales, group_size);
    }
    inline static Tensor manage(SequentialAllocator& alloc,
            int columns, int rows, int layers, QuantType qt=QuantType::NONE, int group_size=QUANT_GROUP_SIZE) noexcept {
        Tensor t(columns, rows, layers, qt, group_size);
        auto ptr = alloc.alloc(t.memory_size());
        t.manage(ptr);
        return t;
    }
    inline static Tensor manage(SequentialAllocator& alloc,
            int columns, int rows=0, QuantType qt=QuantType::NONE, int group_size=QUANT_GROUP_SIZE) noexcept {
        return Tensor::manage(alloc, columns, rows, 0, qt, group_size);
    }

    void reset(int columns, int rows, int layers, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) noexcept {
        if (_data != nullptr && _mtype != uint8_t(MemoryType::MANAGED)) {
            release();
        }
        _columns = columns;
        _rows    = rows;
        _layers  = layers;
        _qtype   = uint8_t(qt);
        _qgsize  = quant_group_size > 1 ? quant_group_size : QUANT_GROUP_SIZE;
    }
    void reset(int columns, int rows, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) noexcept {
        return reset(columns, rows, 0, qt, quant_group_size);
    }
    void reset(int columns, QuantType qt=QuantType::NONE, int quant_group_size=QUANT_GROUP_SIZE) noexcept {
        return reset(columns, 0, 0, qt, quant_group_size);
    }

    bool read_data(std::ifstream& file, ssize_t data_offset=-1, ssize_t scales_offset=-1) noexcept;

    Tensor slice(int start, int end) const noexcept {
        int layers = 0;
        int rows = 0;
        int cols = _columns;
        size_t offset = 0;
        if (_layers > 0) {
            if (end < 0) {
                end += _layers + 1;
            } else if (end > _layers) {
                end = _layers;
            }
            layers = end - start;
            rows = _rows;
            offset = size_t(_columns) * _rows * start;
        } else if (_rows > 0) {
            if (end < 0) {
                end += _rows + 1;
            } else if (end > _rows) {
                end = _rows;
            }
            rows = end - start;
            offset = size_t(_columns) * start;
        } else {
            if (end < 0) {
                end += _columns + 1;
            } else if (end > _columns) {
                end = _columns;
            }
            cols = end - start;
            offset = start;
        }
        Tensor t(cols, rows, layers, QuantType(_qtype), _qgsize);
        t.manage(get_data(offset), _scales == nullptr ? nullptr : &_scales[offset/_qgsize]);
        return t;
    }
    Tensor slice(int end=-1) const noexcept {
        return slice(0, end);
    }
    Tensor row_slice(int row, int layer=0) const noexcept {
        Tensor t(_columns, QuantType(_qtype), _qgsize);
        size_t offset = size_t(_columns) * (row + size_t(_rows) * layer);
        t.manage(get_data(offset), _scales == nullptr ? nullptr : &_scales[offset/_qgsize]);
        return t;
    }

    Tensor operator[](int i) const noexcept {
        int rows = 0;
        int cols = _columns;
        size_t offset = 0;
        if (_layers > 0) {
            if (i < 0) {
                i += _layers;
            }
            rows = _rows;
            offset = size_t(_columns) * _rows * i;
        } else if (_rows > 0) {
            if (i < 0) {
                i += _rows;
            }
            offset = size_t(_columns) * i;
        } else {
            if (i < 0) {
                i += _columns;
            }
            offset = i;
            cols = 1;
        }
        Tensor t(cols, rows, QuantType(_qtype), _qgsize);
        t.manage(get_data(offset), _scales == nullptr ? nullptr : &_scales[offset/_qgsize]);
        return t;
    }

    void clip(int start_columns, int end_columns) noexcept;
    void clip(int end_columns) noexcept {
        return clip(0, end_columns);
    }

    /*************************************< memory manipulations >*************************************/
    bool set_memory_type(MemoryType mt) noexcept;

    // 释放资源
    void release() noexcept;

    // 托管data指向的数据（需在Tensor对象的生命周期内保证data指向地址的可用性）
    size_t manage(void* data, float* scales=nullptr) noexcept;

    // 申请内存以便准备存放数据。
    // @param mt: MemoryType::NONE means keeping previous configured MemoryType value
    bool reserve_memory(MemoryType mt=MemoryType::NONE) noexcept;

    static bool copy(Tensor& target, const Tensor& source,
            int target_column_offset=0, int source_column_offset=0, int columns=-1) noexcept;
    /**
     * @brief create a new Tensor with the same data
     * @param mt : type of memory for storing the data. MemoryType::NONE means keeping same as itself
     * @return the new Tensor instance
     */
    Tensor copy(MemoryType mt=MemoryType::NONE) const noexcept;
    inline bool copy_to  (Tensor& target, int this_column_offset=0) const noexcept {
        return Tensor::copy(target, *this, 0, this_column_offset);
    }
    inline Tensor& copy_from(const Tensor& source, int this_column_offset=0) noexcept {
        Tensor::copy(*this, source, this_column_offset, 0);
        return *this;
    }

    bool zero() noexcept;
    bool random(float max_value, float min_value) noexcept;

    /*****************************************< quantizations >****************************************/
    bool quantize(std::span<const float> data) noexcept;
    bool quantize(const Tensor& t) noexcept;
    bool quantize(std::span<const float> data, int layer, int row_offset=0) noexcept;
    bool quantize(const Tensor& t, int layer, int row_offset=0) noexcept {
        return quantize(std::span<const float>{t.float_data(), t.size()}, layer, row_offset);
    }
    bool dequantize(const Tensor& t) noexcept;

    /******************************************< operators >*****************************************/
    void matmul(const Tensor& b, Tensor& out, int layer=0) const;
    void softmax(int columns=-1) noexcept;
    void rmsnorm(const Tensor& x, const Tensor& w);
    void silu();
    void swiglu(const Tensor& x);
    // apply swiglu on left half and right half of each row
    void  inner_swiglu();
    void rope(int pos, int head_size) noexcept {
        if (_data != nullptr && !is_quantized()) {
            cpuft::rope(float_data(), size(), pos, head_size);
        }
    }
    void rope(const Tensor& rope_freq_cis, int pos) noexcept {
        if (_data != nullptr && !is_quantized()) {
            cpuft::rope(float_data(), size(), rope_freq_cis.float_data(0, pos), rope_freq_cis.columns());
        }
    }
    void sequence_rope(const Tensor& rope_freq_cis, int pos) noexcept {
        if (_data != nullptr && !is_quantized()) {
            for (int i = 0, n = total_rows(); i < n; ++i) {
                cpuft::rope(float_data(0, i), size(), rope_freq_cis.float_data(0, pos+i), rope_freq_cis.columns());
            }
        }
    }
    void sequence_rope_v2(int pos, int head_size, int context_length) noexcept {
        if (_data != nullptr && !is_quantized()) {
            for (int i = 0, n = total_rows(); i < n; ++i) {
                for (int j = 0; j < _columns; j += head_size) {
                    cpuft::rope_v2(float_data(j, i), float_data(j, i), head_size, context_length, pos + i, 0, 1);
                }
            }
        }
    }

    void  weighted_sum(std::span<const float> weights, Tensor& out, int layer=0, float weight_threshold=0.) const;
    void  weighted_sum(const Tensor& weights, Tensor& out, int layer=0, float weight_threshold=0.) const;
    void  add(const Tensor& b, int this_column_offset=0) noexcept;
    void  add(float value) noexcept;
    void  multiply(float v) noexcept {
        if (_data != nullptr) {
            quant::mul(QuantType(_qtype), _data, v, _scales, size(), _qgsize);
        }
    }
    float sum() const noexcept;
    float sum(int dim3_index) const noexcept {
        auto matrix_size = size_t(_columns) * _rows;
        return sum_(matrix_size * dim3_index, matrix_size);
    }
    float sum(int dim3_index, int dim2_index) const noexcept {
        return sum_(size_t(_columns) * (_rows * dim3_index + dim2_index), _columns);
    }

    float square_sum() const noexcept;

    float max(int layer=-1) const noexcept {
        if (layer < 0) {
            return max_(0, size());
        }
        auto matrix_size = size_t(_columns) * _rows;
        return max_(matrix_size * layer, matrix_size);
    }
    float max(int dim3_index, int dim2_index) const noexcept {
        return max_(size_t(_columns) * (_rows * dim3_index + dim2_index), _columns);
    }
    float min() const noexcept {
        return min_(0, size());
    }
    float min(int dim3_index) const noexcept {
        auto matrix_size = size_t(_columns) * _rows;
        return min_(matrix_size * dim3_index, matrix_size);
    }
    float min(int dim3_index, int dim2_index) const noexcept {
        return min_(size_t(_columns) * (_rows * dim3_index + dim2_index), _columns);
    }
    std::pair<float, float> min_max() const noexcept {
        return {min(), max()};
    }
    void sort() noexcept {
        std::span<float> arr{float_data(), size()};
        std::sort(arr.begin(), arr.end());
    }

    /******************************************< display >*****************************************/
    std::string display_string(bool show_attributes=false, bool pretty_print=false, int display_num=9,
                                int precision=-1, int column_start=0, int column_end=-1) const noexcept; 

    void print(std::string_view prefix="", bool show_attributes=false, bool pretty_print=false, int display_num=9, int precision=-1, int column_start=0, int column_end=-1) const noexcept; 
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.display_string();
        return os;
    }

    /******************************************< attributes >*****************************************/
    QuantType quant_type()       const noexcept { return QuantType(_qtype); }
    uint16_t  quant_group_size() const noexcept { return _qgsize; }

    TensorShape shape() const noexcept {
        return {_columns, _rows, _layers};
    }
    Tensor& reshape(int columns, int rows=-1, int layers=-1);

    int columns()    const noexcept { return _columns; }
    int rows()       const noexcept { return _rows; }
    int layers()     const noexcept { return _layers; }
    int total_rows() const noexcept { return (_rows <= 1 ? 1 : _rows) * (_layers <= 1 ? 1 : _layers); }
    size_t size()    const noexcept {
        if (_layers > 0) {
            return size_t(_columns) * _rows * _layers;
        } else if (_rows > 0) {
            return size_t(_columns) * _rows;
        } else {
            return _columns;
        }
    }

    float* float_data(size_t offset=0) const noexcept {
        return reinterpret_cast<float*>(_data) + offset;
    }
    short* int16_data(size_t offset=0) const noexcept {
        return reinterpret_cast<short*>(_data) + offset;
    }
    int8_t* int8_data(size_t offset=0) const noexcept {
        return reinterpret_cast<int8_t*>(_data) + offset;
    }
    float* float_data(int column, int row, int layer=0) const noexcept {
        return reinterpret_cast<float*>(_data) + (size_t(_columns) * (row + size_t(_rows) * layer) + column);
    }

    float get_value(int layer, int row, int column) const noexcept {
        return get_value(size_t(_columns) * (size_t(_rows) * layer + row) + column);
    }
    float get_value(int row, int column) const noexcept {
        return get_value(size_t(_columns) * row + column);
    }
    float get_value(ssize_t i) const noexcept {
        return _data == nullptr ? 0. : quant::get_value(QuantType(_qtype), _data, _scales, i >= 0 ? i : i + size(), _qgsize);
    }

    operator float() const noexcept {
        return get_value(0);
    }

    QuantType quantization_type() const noexcept {
        return QuantType(_qtype);
    }
    int quantization_group_size() const noexcept {
        return _qgsize;
    }

    char* data() const noexcept {
        return _data;
    }
    float* scales() const noexcept {
        return _scales;
    }
    float get_scale(uint32_t index1, uint32_t index2, uint32_t index3) const noexcept {
        return _scales[((size_t(index3) * _rows + size_t(index2)) * _columns + index1)/ _qgsize];
    }
    float get_scale(size_t i) const noexcept {
        return _scales[i / _qgsize];
    }

    size_t memory_size() const noexcept {
        return data_memory_size() + scales_memory_size();
    }

    bool is_same_shape(const Tensor& other) const noexcept {
        return _columns == other._columns && _rows == other._rows && _layers == other._layers;
    }
    bool is_quantized() const noexcept {
        return QuantType(_qtype) != QuantType::NONE;
    }

    /*
    void shape_like(const Tensor& other) noexcept {
        if (_data != nullptr && _mtype != MemoryType::MANAGED && size() < other.size()) {
            auto data_mem_size = (size_to_offset(other.size()) + 63ul) & ~63ul;
            auto scales_mem_size = _scales == nullptr ? 0 : (sizeof(float) * other.size() / _qgsize + 63ul) & ~63ul;
            auto data = alloc(_mtype, data_mem_size + scales_mem_size);
            if (data == nullptr) {
                return;
            }
            memcpy(data, _data, data_memory_size());
            memset(data + data_memory_size(), 0, data_mem_size - data_memory_size());
            if (_scales != nullptr) {
                memcpy(data + data_mem_size, _scales, scales_memory_size());
            }
            release();
            _data = data;
            _scales = is_quantized() ? reinterpret_cast<float*>(_data + data_mem_size) : nullptr; 
        }
        _columns = other._columns;
        _rows    = other._rows;
        _layers  = other._layers;
    }*/

    inline void like(const Tensor& other) noexcept {
        if (_data != nullptr && (_mtype != other._mtype || memory_size() < other.memory_size())) {
            release();
        }
        _columns = other._columns;
        _rows    = other._rows;
        _layers  = other._layers;
        _mtype  = other._mtype;
        _qtype  = other._qtype;
        _qgsize = other._qgsize;
    }
    static const char* type_to_name(QuantType qt) noexcept {
        switch (qt) {
            case QuantType::NONE:  return "None";
            case QuantType::INT16: return "int16";
            case QuantType::INT8:  return "int8";
            case QuantType::INT4:  return "int4";
            default: break;
        }
        return "unkown";
    }
    static const char* type_to_name(MemoryType mt) noexcept {
        switch (mt) {
            case MemoryType::NONE:      return "NONE";
            case MemoryType::NORMAL:    return "NORMAL";
            case MemoryType::MANAGED:   return "MANAGED";
            case MemoryType::GPU:       return "GPU";
            default: break;
        }
        return "unkown";
    }
    const char* quant_type_name() const noexcept {
        return Tensor::type_to_name(QuantType(_qtype));
    }
    const char* memory_type_name() const noexcept {
        return Tensor::type_to_name(MemoryType(_mtype));
    }

protected:
    inline size_t size_to_offset(size_t z) const noexcept {
        if (QuantType(_qtype) == QuantType::NONE) {
            return z * 4;
        } else if (QuantType(_qtype) == QuantType::INT16) {
            return z * 2;
        } else if (QuantType(_qtype) == QuantType::INT8) {
            return z;
        } else {
            return z / 2;
        }
    }
    void* get_data(size_t offset=0) const noexcept {
        return reinterpret_cast<void*>(&_data[size_to_offset(offset)]);
    }
    float* get_scales(size_t offset=0) const noexcept {
        return &_scales[offset / _qgsize];
    }

    inline size_t data_memory_size() const noexcept {
        return (size_to_offset(size()) + 63ul) & ~63ul;
    }
    inline size_t scales_memory_size() const noexcept {
        return is_quantized() ? ((sizeof(float)*size()/_qgsize + 63ul) & ~63ul) : 0;
    }

    inline void update_scales_pointer() noexcept {
        if (_data == nullptr || _qtype == uint8_t(QuantType::NONE)) {
            _scales = nullptr;
        } else {
            _scales = reinterpret_cast<float*>(_data + data_memory_size());
        }
    }

    static char* alloc(MemoryType mt, size_t size) noexcept;
    static void  free(MemoryType mt, char* ptr, size_t size) noexcept;

    inline Tensor& move(Tensor&& other) noexcept {
        if (_data != nullptr) {
            release();
        }
        if (other._data != nullptr) {
            _data   = other._data;
            other._data   = nullptr;
            _scales = other._scales;
            other._scales = nullptr;
        }
        _columns  = other._columns;
        _rows     = other._rows;
        _layers   = other._layers;
        _mtype    = other._mtype;
        _qtype    = other._qtype;
        _qgsize   = other._qgsize;
        return *this;
    }

    float sum_(size_t offset, size_t length) const noexcept;
    float max_(size_t offset, size_t length) const noexcept;
    float min_(size_t offset, size_t length) const noexcept;

private:
    char*       _data   = nullptr;
    float*      _scales = nullptr; // 量化缩放系数表

    int         _columns = 0;
    int         _rows    = 0; // 为0表示此维度不启用
    int         _layers  = 0; // 为0表示此维度不启用

    uint16_t    _qgsize = QUANT_GROUP_SIZE; // 量化的窗口大小
    uint8_t     _qtype  = uint8_t(QuantType::NONE);
    uint8_t     _mtype  = uint8_t(MemoryType::NORMAL);
};

}
