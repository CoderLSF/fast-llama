
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>

#include <cstring>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <span>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "utility.h"
#include "model_loader.h"
#include "log.h"

namespace cpuft {

static constexpr uint32_t GGUF_FILE_TAG = 0x46554747;

enum class DataType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
};
struct TensorInfo {
    std::string   name;
    int           layer = -1;
    int           n_dims = 0;
    int           shape[3] = {};
    DataType      dtype = DataType::F32;
    int64_t       offset = 0;

    Tensor*       pt = nullptr;
};

enum class GGUFFileType {
    ALL_F32     = 0,
    MOSTLY_F16  = 1, // except 1d tensors
    MOSTLY_Q8_0 = 7, // except 1d tensors
};

enum class GGUFValueType {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,

    _MAX_,
};

inline bool is_integer(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::INT8 || vt == GGUFValueType::UINT8 || vt == GGUFValueType::BOOL
        || vt == GGUFValueType::INT16 || vt == GGUFValueType::UINT16 || vt == GGUFValueType::INT32
        || vt == GGUFValueType::UINT32 || vt == GGUFValueType::INT64 || vt == GGUFValueType::UINT64;
}
inline bool is_float(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::FLOAT32 || vt == GGUFValueType::FLOAT64;
}
inline bool is_string(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::STRING;
}
inline bool is_array(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::ARRAY;
}

template <typename T>
bool read_item(std::ifstream& file, T& v) {
    return !!file.read(reinterpret_cast<char*>(&v), sizeof(v));
}
bool read_item(std::ifstream& file, std::string& v) {
    int64_t len = -1;
    if (!read_item(file, len)) {
        return false;
    }
    v.resize(len, '\0');
    return !!file.read(v.data(), len);
}

struct GGUFValueTypeDescriptor {
    const char*  name;
    int          size;
};
GGUFValueTypeDescriptor s_vtype_descriptors[int(GGUFValueType::_MAX_)] = {
    {"UINT8",   1},  {"INT8",   1},  {"UINT16",  2},  {"INT16",  2},
    {"UINT32",  4},  {"INT32",  4},  {"FLOAT32", 4},  {"BOOL",   1},
    {"STRING", -1},  {"ARRAY", -1},  {"UINT64",  8},  {"INT64",  8},
    {"FLOAT64", 8},
};

int64_t read_integer(std::ifstream& file, GGUFValueType vt, int64_t default_value=-1) {
    int64_t res = default_value;
    if (vt == GGUFValueType::UINT8 || vt == GGUFValueType::UINT16
            || vt == GGUFValueType::UINT32 || vt == GGUFValueType::UINT32
            || vt == GGUFValueType::INT64 || vt == GGUFValueType::UINT64) {
        file.read(reinterpret_cast<char*>(&res), s_vtype_descriptors[int(vt)].size);
    } else if (vt == GGUFValueType::INT8) {
        if (int8_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
            res = v;
        }
    } else if (vt == GGUFValueType::INT16) {
        if (int16_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
            res = v;
        }
    } else if (vt == GGUFValueType::INT32) {
        if (int32_t v; file.read(reinterpret_cast<char*>(&v), s_vtype_descriptors[int(vt)].size)) {
            res = v;
        }
    }
    return res;
}
double read_float(std::ifstream& file, GGUFValueType vt, double default_value=0.) {
    if (vt == GGUFValueType::FLOAT32) {
        float v = 0.;
        if (!file.read(reinterpret_cast<char*>(&v), sizeof(v))) {
            return default_value;
        }
        return v;
    } else if (vt == GGUFValueType::FLOAT64) {
        double v = 0.;
        if (!file.read(reinterpret_cast<char*>(&v), sizeof(v))) {
            return default_value;
        }
        return v;
    } else {
        return default_value;
    }
}
std::string read_string(std::ifstream& file) {
    std::string res;
    read_item(file, res);
    return res;
}

template <typename T>
bool read_array(std::ifstream& file, std::vector<T>& arr) {
    GGUFValueType type =  static_cast<GGUFValueType>(read_integer(file, GGUFValueType::INT32));
    int64_t size = read_integer(file, GGUFValueType::INT64, -1);
    if (size < 1) {
        return size >= 0;
    }
    if (size > (int64_t(1) << 40)) {
        return false;
    }
    if (arr.size() != size_t(size)) {
        arr.resize(size_t(size));
    }
    bool is_fixed = s_vtype_descriptors[int(type)].size > 0;
    if (is_fixed && sizeof(T) == s_vtype_descriptors[int(type)].size) {
        return !!file.read(reinterpret_cast<char*>(arr.data()), sizeof(T) * size);
    }
    if (is_integer(type)) {
        for (int64_t i = 0; i < size; ++i) {
            arr[i] = read_integer(file, type);
        }
    } else if (is_float(type)) {
        for (int64_t i = 0; i < size; ++i) {
            arr[i] = read_float(file, type);
        }
    } else {
        return false;
    }
    return true;
}
bool read_array(std::ifstream& file, std::vector<std::string>& arr) {
    GGUFValueType type =  static_cast<GGUFValueType>(read_integer(file, GGUFValueType::INT32));
    if (type != GGUFValueType::STRING) {
        return false;
    }
    int64_t size = read_integer(file, GGUFValueType::INT64, -1);
    if (size < 1) {
        return size >= 0;
    }
    if (size > (int64_t(1) << 40)) {
        return false;
    }
    if (arr.size() != size_t(size)) {
        arr.resize(size_t(size));
    }
    for (int64_t i = 0; i < size; ++i) {
        arr[i] = read_string(file);
    }
    return true;
}

bool TransformerModel::load_gguf(std::string_view file_path, bool tokenizer_only) noexcept {
    std::ifstream file(file_path.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        tf_log_error("Failed to open gguf file:%s", file_path.data());
        return false;
    }
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    //tf_log_debug("file_size:%lu", file_size);

    struct FileHeader {
        uint32_t magic_number;
        uint32_t version;
        int64_t  ti_data_count;
        int64_t  kv_data_count;
    } fh;
    if (!file.read(reinterpret_cast<char*>(&fh), sizeof(fh))) {
        tf_log_error("Reading gguf file error");
        return false;
    }
    if (fh.magic_number != 0x46554747 || fh.ti_data_count < 1 || fh.kv_data_count < 1) {
        tf_log_error("Not a valid gguf file:%s", file_path.data());
        return false;
    }
    //tf_log_debug("ti_data_count: %ld", fh.ti_data_count);
    //tf_log_debug("kv_data_count: %ld", fh.kv_data_count);

    int rope_dimension_count = 0;

    uint32_t alignment    = 32;
    conf.quant_group_size = 32;
    for (int64_t i = 0; i < fh.kv_data_count; ++i) {
        std::string key = read_string(file);
        GGUFValueType vtype = GGUFValueType(read_integer(file, GGUFValueType::INT32));

        //tf_log_debug("Reading key:\x1b[32m%s\x1b[0m type:\x1b[33m%s\x1b[0m",
        //        key.c_str(), s_vtype_descriptors[int(vtype)].name);
        if (key == "general.architecture" && vtype == GGUFValueType::STRING) {
            std::string val = read_string(file);
            if (val == "llama") {
                conf.arch = ModelArchitecture::LLAMA;
            } else {
                tf_log_error("Unsupported architecture:%s", val.c_str());
                return false;
            }
        } else if (key == "general.name" && vtype == GGUFValueType::STRING) {
            conf.name = read_string(file);
        } else if (key == "general.file_type" && is_integer(vtype)) {
            GGUFFileType ft = GGUFFileType(read_integer(file, GGUFValueType(vtype)));
            switch (ft) {
            case GGUFFileType::ALL_F32:     conf.wtype = WeightType::FP32; break;
            case GGUFFileType::MOSTLY_F16:  conf.wtype = WeightType::FP16; break;
            case GGUFFileType::MOSTLY_Q8_0: conf.wtype = WeightType::Q8_0; break;
            default:
                tf_log_error("Unsupported file type:%d\n", int(ft));
                return false;
            }
        } else if (key == "general.quant_group_size" && is_integer(vtype)) {
            conf.quant_group_size = uint32_t(read_integer(file, vtype));
        } else if (key == "general.alignment" && is_integer(vtype)) {
            alignment = uint32_t(read_integer(file, vtype));
        } else if (key == "llama.context_length" && is_integer(vtype)) {
            conf.max_seq_len = int(read_integer(file, vtype));
        } else if (key == "llama.embedding_length" && is_integer(vtype)) {
            conf.dim = int(read_integer(file, vtype));
        } else if (key == "llama.block_count" && is_integer(vtype)) {
            conf.n_layers = int(read_integer(file, vtype));
        } else if (key == "llama.feed_forward_length" && is_integer(vtype)) {
            conf.hidden_dim = int(read_integer(file, vtype));
        } else if (key == "llama.attention.head_count" && is_integer(vtype)) {
            conf.n_heads = int(read_integer(file, vtype));
        } else if (key == "llama.attention.head_count_kv" && is_integer(vtype)) {
            conf.n_kv_heads = int(read_integer(file, vtype));
        } else if (key == "llama.rope.dimension_count" && is_integer(vtype)) {
            rope_dimension_count = int(read_integer(file, vtype));
        } else if (key == "llama.rope.freq_base" && (is_float(vtype) || is_integer(vtype))) {
            conf.rope_freq_base = is_float(vtype) ? read_float(file, vtype) : read_integer(file, vtype);
        } else if (key == "llama.attention.layer_norm_rms_epsilon" && is_float(vtype)) {
            conf.layer_norm_rms_epsilon = read_float(file, vtype);
        } else if (key == "tokenizer.ggml.model" && is_string(vtype)) {
            tokenizer.set_name(std::move(read_string(file)));
        } else if (key == "tokenizer.ggml.bos_token_id" && is_integer(vtype)) {
            tokenizer.set_bos_token_id(int(read_integer(file, vtype)));
        } else if (key == "tokenizer.ggml.eos_token_id" && is_integer(vtype)) {
            tokenizer.set_eos_token_id(int(read_integer(file, vtype)));
        } else if (key == "tokenizer.ggml.padding_token_id" && is_integer(vtype)) {
            tokenizer.set_pad_token_id(int(read_integer(file, vtype)));
        } else if (key == "tokenizer.ggml.tokens" && is_array(vtype)) {
            std::vector<std::string> arr;
            if (!read_array(file, arr)) {
                tf_log_error("Reading tokens error");
                return false;
            }
            conf.vocab_size = arr.size();
            tokenizer.set_token_texts(arr);
        } else if (key == "tokenizer.ggml.scores" && is_array(vtype)) {
            std::vector<float> arr;
            if (!read_array(file, arr)) {
                tf_log_error("Reading scores error");
                return false;
            }
            tokenizer.set_token_scores(arr);
        } else if (key == "tokenizer.ggml.token_type" && is_array(vtype)) {
            std::vector<int> arr;
            if (!read_array(file, arr)) {
                tf_log_error("Reading scores error");
                return false;
            }
            tokenizer.set_token_types(arr);
            tokenizer.set_vocab_type(VocabType::SPM);
        } else  {
            tf_log_error("Unknown key:\x1b[31m%s\x1b[0m", key.c_str());
            return false;
        }
    }
    conf.head_size = conf.n_heads > 0 ? conf.dim / conf.n_heads : 0;
    conf.kv_dim    = conf.n_kv_heads > 0 ? conf.head_size * conf.n_kv_heads : 0;
    if (conf.n_heads < 1 || conf.n_kv_heads < 1 || conf.dim % conf.n_heads != 0
            || (rope_dimension_count > 0 && rope_dimension_count != conf.head_size)) {
        tf_log_error("Invalid dim:%d or n_heads:%d or n_kv_heads:%d or rope_dimension_count:%d",
            conf.dim, conf.n_heads, conf.n_kv_heads, rope_dimension_count);
        return false;
    }
    if (conf.wtype == WeightType::Q8_0) {
        conf.quant_type = QuantType::INT8;
    } else {
        conf.quant_group_size = 64;
    }

    weights.rope_freq_cis.reset(conf.head_size, conf.max_seq_len);
    if (!weights.build_rope_freq_cis()) {
        tf_log_error("Failed to build rope_freq_cis while loading:%s", file_path.data());
        return false;
    }

    if (!tokenizer_only) {
        if (is_debug) tf_log_debug("Loading tensors ...");

        std::unordered_map<std::string, TensorInfo> tim;
        tim["token_embd.weight"].pt  = &weights.token_embedding_table;
        tim["output_norm.weight"].pt = &weights.out_norm;
        tim["output.weight"].pt      = &weights.classifier;

        for (int i = 0; i < conf.n_layers; ++i) {
            std::string ns = std::string("blk.") + std::to_string(i) + ".";
            tim[ns + "attn_q.weight"]     .pt    = &weights.attn_q;
            tim[ns + "attn_k.weight"]     .pt    = &weights.attn_k;
            tim[ns + "attn_v.weight"]     .pt    = &weights.attn_v;
            tim[ns + "attn_output.weight"].pt    = &weights.attn_o;
            tim[ns + "attn_norm.weight"]  .pt    = &weights.attn_norm;
            tim[ns + "ffn_gate.weight"]   .pt    = &weights.ffn_1;
            tim[ns + "ffn_down.weight"]   .pt    = &weights.ffn_2;
            tim[ns + "ffn_up.weight"]     .pt    = &weights.ffn_3;
            tim[ns + "ffn_norm.weight"]   .pt    = &weights.ffn_norm;

            tim[ns + "attn_q.weight"]     .layer = i;
            tim[ns + "attn_k.weight"]     .layer = i;
            tim[ns + "attn_v.weight"]     .layer = i;
            tim[ns + "attn_output.weight"].layer = i;
            tim[ns + "attn_norm.weight"]  .layer = i;
            tim[ns + "ffn_gate.weight"]   .layer = i;
            tim[ns + "ffn_up.weight"]     .layer = i;
            tim[ns + "ffn_down.weight"]   .layer = i;
            tim[ns + "ffn_norm.weight"]   .layer = i;
        }

        std::vector<TensorInfo*> tis;
        for (int64_t i = 0; i < fh.ti_data_count; ++i) {
            std::string name = read_string(file);
            if (tim.find(name) == tim.end()) {
                tf_log_error("Invalid tensor name:%s", name.c_str());
                return false;
            }

            TensorInfo& ti = tim[name];
            if (ti.n_dims > 0) {
                tf_log_error("Duplicated tensor info for:%s", name.c_str());
                continue;
            }
            ti.n_dims = read_integer(file, GGUFValueType::INT32);
            if (ti.n_dims < 1 || ti.n_dims > int(sizeof(ti.shape) / sizeof(ti.shape[0]))) {
                tf_log_error("Invalid shape of tensor:%s", name.c_str());
                return false;
            }
            ti.name = std::move(name);
            tis.push_back(&ti);

            for (int j = 0; j < ti.n_dims; ++j) {
                ti.shape[j] = read_integer(file, GGUFValueType::INT64);
            }
            ti.shape[ti.n_dims] = ti.layer >= 0 ? conf.n_layers : 0;
            ti.dtype  = DataType(read_integer(file, GGUFValueType::INT32));
            ti.offset = read_integer(file, GGUFValueType::INT64);
            if (ti.layer <= 0) {
                if (ti.dtype == DataType::F32) {
                    ti.pt->reset(ti.shape[0], ti.shape[1], ti.shape[2]);
                } else if (ti.dtype == DataType::Q8_0) {
                    ti.pt->reset(ti.shape[0], ti.shape[1], ti.shape[2], QuantType::INT8, conf.quant_group_size);
                } else {
                    tf_log_error("This data type is not supported yet.");
                    return false;
                }
                if (!ti.pt->reserve_memory()) {
                    tf_log_error("Out of memory for storing tensor data.");
                    return false;
                }
            }
        }

        size_t file_pos = size_t(file.tellg());
        file_pos = (file_pos + alignment - 1) & ~size_t(alignment - 1);

        for (auto pti : tis) {
            auto& ti = *pti;
            size_t offset = file_pos + ti.offset;
            size_t num_items = 1;
            for (int i = 0; i < ti.n_dims; ++i) {
                num_items *= ti.shape[i];
            }
            size_t tensor_size = 0;
            if (ti.dtype == DataType::F32) {
                tensor_size = num_items * 4;
            } else if (ti.dtype == DataType::F16) {
                tensor_size = num_items * 2;
            } else if (ti.dtype == DataType::Q8_0) {
                tensor_size = num_items * 1 + num_items * 2 / conf.quant_group_size;
            } else {
                tf_log_error("Invalid data type:%d", int(ti.dtype));
                return false;
            }

            Tensor t;
            if (ti.layer >= 0) {
                t = (*ti.pt)[ti.layer];
            } else {
                t = ti.pt->slice();
            }

            std::unique_ptr<char[]> raw_data;
            char* pdata = nullptr;
            if (ti.dtype == DataType::F32) {
                pdata = t.data();
            } else {
                raw_data.reset(new char[tensor_size]);
                if (raw_data == nullptr) {
                    tf_log_error("Out of memory for reading tensor data");
                    return false;
                }
                pdata = raw_data.get();
            }
            file.seekg(file_pos + ti.offset);
            if (!file.read(pdata, tensor_size)) {
                tf_log_error("Reading tensor data error, file_pos:%lu/%lu, size:%lu",
                        size_t(file_pos + ti.offset), file_size, tensor_size);
                return false;
            }

            if (ti.dtype == DataType::F32 || ti.dtype == DataType::F16) {
            } else if (ti.dtype == DataType::Q8_0) {
                char*  pd = t.data();
                float* qs = t.scales();
                for (size_t i = 0, n = num_items / conf.quant_group_size; i < n; ++i) {
                    qs[i] = float16_to_float32(*reinterpret_cast<uint16_t*>(pdata));
                    memcpy(pd, pdata+2, conf.quant_group_size);
                    pdata += 2 + conf.quant_group_size;
                    pd += conf.quant_group_size;
                }
            }

            if (is_debug) tf_log_debug("name:%-25s\tlayer:%2d\tdtype:%2d\toffset:%10lu\ttensor_shape:%s",
                ti.name.c_str(), ti.layer, int(ti.dtype), offset, t.shape().serialize().c_str());
            if (offset + tensor_size > file_size) {
                tf_log_error("data exceeds");
            }
        }

        if (is_debug) print_summary();
    }
    return true;
}

bool TransformerModel::is_valid_gguf_header(std::span<const char> file_header) noexcept {
    if (file_header.size() < sizeof(GGUF_FILE_TAG)) {
        return false;
    }
    return *reinterpret_cast<const uint32_t*>(file_header.data()) == GGUF_FILE_TAG;
}

} // namespace cpuft

