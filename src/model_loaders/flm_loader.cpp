
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <span>
#include <filesystem>

#include "log.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "console.h"

namespace cpuft {

enum class BlockType {
    BASE_ITEM     = 0,
    DICT          = 1,
    TENSOR        = 2,
    ARRAY         = 3,
    STRING        = 4,
    STRING_ARRAY  = 5,
    _MAX_,
};

enum class DataType {
    NONE    =  0,
    INT8    =  1,
    INT16   =  2,
    INT32   =  3,
    INT64   =  4,
    UINT8   =  5,
    UINT16  =  6,
    UINT32  =  7,
    UINT64  =  8,
    FLOAT16 = 10, 
    FLOAT32 = 11, 
    FLOAT64 = 12,
    BLOCK   = 15,
    _MAX_,
};

enum class TensorType {
    NONE              =  0,

    TOKEN_EMBD_TABLE  =  1,
    OUTPUT_NORM       =  2,
    CLASSIFIER        =  3,

    LAYER_TYPE        = 16,
    LAYER_INPUT_NORM  = 17,
    LAYER_ATTN_Q      = 18,
    LAYER_ATTN_K      = 19,
    LAYER_ATTN_V      = 20,
    LAYER_ATTN_O      = 21,
    LAYER_MLP_GATE    = 22,
    LAYER_MLP_UP      = 23,
    LAYER_MLP_DOWN    = 24,
    LAYER_POST_NORM   = 25,
};

const char* type_to_name(BlockType dt) {
    switch (dt) {
        case BlockType::BASE_ITEM:   return "base_item";
        case BlockType::DICT:        return "dict";
        case BlockType::TENSOR:      return "tensor";
        case BlockType::ARRAY:       return "array";
        case BlockType::STRING_ARRAY:return "string_array";
        default: return "";
    }
}

const char* type_to_name(DataType dt) {
    switch (dt) {
        case DataType::NONE:   return "None";
        case DataType::INT8:   return "int8";
        case DataType::UINT8:  return "uint8";
        case DataType::INT16:  return "int16";
        case DataType::UINT16: return "uint16";
        case DataType::INT32:  return "int32";
        case DataType::UINT32: return "uint32";
        case DataType::INT64:  return "int64";
        case DataType::UINT64: return "uint64";
        case DataType::FLOAT16:return "float16";
        case DataType::FLOAT32:return "float32";
        case DataType::FLOAT64:return "float64";
        default: return "";
    }
}
const char* type_to_name(TensorType tt) {
    switch (tt) {
        case TensorType::TOKEN_EMBD_TABLE:  return "token_embd_table";
        case TensorType::OUTPUT_NORM:       return "output_norm";
        case TensorType::CLASSIFIER:        return "classifier";
        case TensorType::LAYER_INPUT_NORM:  return "layer_input_norm";
        case TensorType::LAYER_ATTN_Q:      return "layer_attn_q";
        case TensorType::LAYER_ATTN_K:      return "layer_attn_k";
        case TensorType::LAYER_ATTN_V:      return "layer_attn_v";
        case TensorType::LAYER_ATTN_O:      return "layer_attn_o";
        case TensorType::LAYER_POST_NORM:   return "layer_attn_post_norm";
        case TensorType::LAYER_MLP_GATE:    return "layer_mlp_gate";
        case TensorType::LAYER_MLP_UP:      return "layer_mlp_up";
        case TensorType::LAYER_MLP_DOWN:    return "layer_mlp_down";
        default: return "";
    }
}

constexpr uint32_t FLM_FILE_TAG = 0xFA571AEA;
struct FileHeader {
    uint32_t  file_tag;
    struct {
        uint8_t   v1;
        uint8_t   v2;
        uint16_t  v3;
    } version;
};

bool TransformerModel::is_valid_flm_header(std::span<const char> file_header) noexcept {
    if (file_header.size() < sizeof(FLM_FILE_TAG)) {
        return false;
    }
    return *reinterpret_cast<const uint32_t*>(file_header.data()) == FLM_FILE_TAG;
}

struct Block {
    union {
        char  buffer[256];
        struct {
            uint8_t     block_type;
            uint8_t     data_type;
            uint8_t     header_size;
            uint8_t     header_data_size;
        };
        struct {
            uint8_t     header[4];
            union {
                int8_t      i8;
                uint8_t     u8;
                int16_t     i16;
                uint16_t    u16;
                int32_t     i32;
                uint32_t    u32;
                float       f32;
            };
            char        name[8];
        } tb; // tiny block

        struct {
            uint8_t     header[4];
            uint32_t    __pad__;
            union {
                int64_t     i64;
                uint64_t    u64;
                double      f64;
            };
            char        name[8];
        } sb; // small block

        struct {
            uint8_t    header[4];
            uint8_t    name_offset;
            uint8_t    name_size;
            uint16_t   tail_pad_size;
            uint64_t   data_size;
            // Following are only for tensors
            uint32_t   shape[4];
            uint16_t   tensor_type;
            uint16_t   layer_id;
            uint32_t   scales_size;
        } lb; // medium block
    };

    bool read_header(std::ifstream& file) noexcept {
        if (!file.read(buffer, 8)) {
            tf_log_error("Reading block header error");
            return false;
        }
        if (!file.read(buffer+8, header_size - 8)) {
            tf_log_error("Reading block header error");
            return false;
        }
        return true;
    }

    BlockType get_type() const noexcept {
        return BlockType(block_type);
    }
    const char* type_name() const noexcept {
        return type_to_name(get_type());
    }
    DataType get_data_type() const noexcept {
        return DataType(data_type);
    }
    const char* data_type_name() const noexcept {
        return type_to_name(get_data_type());
    }

    const char* get_name() const noexcept {
        if (get_type() == BlockType::BASE_ITEM) {
            return &buffer[header_data_size <= 4 ? 8 : 16];
        } else {
            return &buffer[lb.name_offset];
        }
    }

    size_t get_data_size() const noexcept {
        if (get_type() == BlockType::BASE_ITEM) {
            return header_data_size;
        } else {
            return lb.data_size;
        }
    }
    size_t get_block_size() const noexcept {
        if (get_type() == BlockType::BASE_ITEM) {
            return header_size;
        } else {
            return size_t(header_size) + lb.data_size + lb.tail_pad_size;
        }

    }
    size_t get_item_size() const noexcept {
        switch (get_data_type()) {
        case DataType::INT8:   [[fallthrough]];
        case DataType::UINT8:
            return 1;
        case DataType::INT16:  [[fallthrough]];
        case DataType::UINT16: [[fallthrough]];
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:  [[fallthrough]];
        case DataType::UINT32: [[fallthrough]];
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:  [[fallthrough]];
        case DataType::UINT64: [[fallthrough]];
        case DataType::FLOAT64:
            return 8;
        default:
            return get_data_size();
        }
    }
    size_t get_item_count() const noexcept {
        return get_data_size() / get_item_size();
    }
    int64_t get_int_value() const noexcept {
        return header_data_size <= 4 ? tb.i32 : sb.i64;
    }
    double get_float_value() const noexcept {
        return header_data_size <= 4 ? tb.f32 : sb.f64;
    }
    bool is_string() const noexcept {
        return get_type() == BlockType::STRING;
    }
    bool is_int() const noexcept {
        auto dt = DataType(data_type);
        return dt == DataType::INT8  || dt == DataType::INT16  || dt == DataType::INT32  || dt == DataType::INT64
            || dt == DataType::UINT8 || dt == DataType::UINT16 || dt == DataType::UINT32 || dt == DataType::UINT64;
    }
    bool is_float() const noexcept {
        auto dt = DataType(data_type);
        return dt == DataType::FLOAT16 || dt == DataType::FLOAT32 || dt == DataType::FLOAT32 || dt == DataType::FLOAT64;
    }
    bool is_tensor() const noexcept {
        return get_type() == BlockType::TENSOR;
    }
    TensorType get_tensor_type() const noexcept {
        return TensorType(lb.tensor_type);
    }
    const char* tensor_type_name() const noexcept {
        return type_to_name(get_tensor_type());
    }
    int get_layer_id() const noexcept {
        return int(lb.layer_id);
    }
    bool is_layer_tensor() const noexcept {
        return int(lb.tensor_type) >= int(TensorType::LAYER_TYPE);
    }

    void get_block_info(std::span<char> buf) const noexcept {
        if (!buf.empty()) {
            snprintf(buf.data(), buf.size(), "block:%-50s type:%-10s data_type:%-10s data_size:%-9lu",
                get_name(), type_name(), data_type_name(), get_data_size());
        }
    }
    void get_tensor_info(std::span<char> buf, bool colored=false) const noexcept {
        if (buf.empty()) {
            return;
        }
        buf[0] = '\0';
        if (!is_tensor()) {
            return;
        }

        Console console(colored);

        size_t pos = 0;
        char*  ptr = buf.data();
        if (pos < buf.size()) {
            pos += snprintf(ptr + pos, buf.size() - pos, "tensor:%s%-50s%s type:%s%-22s%s  data_type:%s%-10s%s layer_id:%-5d  shape:(",
                    console.get_color(Color::YELLOW), get_name(), console.get_endtag(),
                    console.get_color(Color::YELLOW), tensor_type_name(), console.get_endtag(),
                    console.get_color(get_data_type() == DataType::INT8 ? Color::GREEN : Color::YELLOW), data_type_name(),  console.get_endtag(),
                    int(lb.layer_id));
        }

        for (int i = 0; pos < buf.size() && i < 4 && lb.shape[i] > 0; ++i) {
            if (i > 0) {
                ptr[pos++] = ',';
            }
            if (pos < buf.size()) {
                pos += snprintf(ptr + pos, buf.size() - pos, "%u", lb.shape[i]);
            }
        }
        if (pos < buf.size()) {
            ptr[pos++] = ')';
        }

        pos = pos < buf.size() ? pos : buf.size() - 1;
        buf[pos] = '\0';
    }
    std::span<const uint32_t> get_tensor_shape() const noexcept {
        if (lb.shape[0] < 1) {
            return {};
        } else if (lb.shape[1] < 1) {
            return {lb.shape, 1};
        } else if (lb.shape[2] < 1) {
            return {lb.shape, 2};
        } else if (lb.shape[3] < 1) {
            return {lb.shape, 3};
        } else {
            return {lb.shape, 4};
        }
    }
};

struct StringArray {
    uint64_t num_items = 0;
    uint64_t data_size = 0;
    std::unique_ptr<char*[]> ptrs;
    std::unique_ptr<char[]>  data;

    bool read(std::ifstream& file) {
        bool ret = true;
        ret = ret && file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        ret = ret && file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if (!ret) {
            return false;
        }
        ptrs.reset(new char*[num_items]);
        data.reset(new char[data_size]);

        uint64_t* offsets = reinterpret_cast<uint64_t*>(ptrs.get());
        ret = ret && file.read(reinterpret_cast<char*>(offsets), sizeof(uint64_t) * num_items);
        ret = ret && file.read(data.get(), data_size);
        if (!ret) {
            return false;
        }
        for (uint64_t i = 0; i < num_items; ++i) {
            ptrs[i] = data.get() + offsets[i];
        }
        return true;
    }
};

static bool read_data(std::ifstream& file, const Block& block, std::span<char> data) {
    auto bt = BlockType(block.block_type);
    if (bt == BlockType::BASE_ITEM) {
        memcpy(data.data(), &block.buffer[block.header_data_size <= 4 ? 8 : 16], block.header_data_size);
        return true;
    }
    if (data.size() < block.lb.data_size) {
        return false;
    }
    if (!file.read(data.data(), block.lb.data_size)) {
        return false;
    }
    if (data.size() > block.lb.data_size) {
        data[block.lb.data_size] = '\0';
    }
    return true;
}

bool load_config(std::ifstream& file, size_t file_pos, size_t data_size, TransformerConfig& config) {
    Block block;
    char  data[256];
    for (size_t end = file_pos + data_size; file_pos < end; file_pos += block.get_block_size()) {
        file.seekg(file_pos, std::ios::beg);
        if (!block.read_header(file)) {
            tf_log_error("Reading config block header error at file_pos:%lu", file_pos);
            return false;
        }
        std::string_view name = block.get_name();
        if (name == "name") {
            read_data(file, block, data);
            config.name = data;
        } else if (name == "model_type") {
            config.arch = ModelArchitecture(block.get_int_value());
        } else if (name == "act_type") {
            config.act_type = ActivationType(block.get_int_value());
        } else if (name == "vocab_size") {
            config.vocab_size = int(block.get_int_value());
        } else if (name == "dim") {
            config.dim = int(block.get_int_value());
        } else if (name == "hidden_dim") {
            config.hidden_dim = int(block.get_int_value());
        } else if (name == "n_heads") {
            config.n_heads = int(block.get_int_value());
        } else if (name == "n_kv_heads") {
            config.n_kv_heads = int(block.get_int_value());
        } else if (name == "n_layers") {
            config.n_layers = int(block.get_int_value());
        } else if (name == "max_length") {
            config.max_seq_len = int(block.get_int_value());
        } else if (name == "rope_theta") {
            config.rope_freq_base = float(block.get_float_value());
        } else if (name == "rms_norm_eps") {
            config.layer_norm_rms_epsilon = block.get_float_value();
        } else if (name == "quant_type") {
            config.quant_type = QuantType(block.get_int_value());
        } else if (name == "quant_group_size") {
            config.quant_group_size = float(block.get_int_value());
        }
    }
    if (config.n_kv_heads < 1) {
        config.n_kv_heads = config.n_heads;
    }
    if (config.n_heads < 1 || (config.dim % config.n_heads) != 0 || config.n_kv_heads > config.n_heads) {
        tf_log_error("Invalid config value");
        return false;
    }
    config.head_size = config.dim / config.n_heads;
    config.kv_dim = config.head_size * config.n_kv_heads;
    config.rope_dimension_count = config.head_size;
    return true;
}

bool load_tokenizer(std::ifstream& file, Tokenizer& tokenizer) {
    struct TokenItem {
        uint32_t    index_text_pos; // offset in text_data
        uint32_t    show_text_pos;  // offset in text_data
        uint32_t    token_type;
        float       score;
    };
    struct TokenizerHeader {
        uint32_t    vocab_type;
        uint32_t    conn_tag_pos;
        int32_t     special_tokens[int(SpecialTokenType::MAX)]; // -1表示为设置
        uint32_t    vocab_size;
        uint32_t    text_data_size;
    } header;

    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        tf_log_error("Reading tokenizer header error");
        return false;
    }

    std::unique_ptr<TokenItem[]> tokens(new TokenItem[header.vocab_size]);
    std::unique_ptr<char[]>  texts(new char[header.text_data_size]);
    
    if (!file.read(reinterpret_cast<char*>(tokens.get()), sizeof(TokenItem) * header.vocab_size)) {
        tf_log_error("Reading tokenizer tokens error");
        return false;
    }
    if (!file.read(texts.get(), header.text_data_size)) {
        tf_log_error("Reading tokenizer texts error");
        return false;
    }

    auto ptext = texts.get();
    std::unique_ptr<Token[]> vocab(new Token[header.vocab_size]);
    for (size_t i = 0; i < header.vocab_size; ++i) {
        auto& token = vocab[i];
        token.index_text = ptext + tokens[i].index_text_pos;
        token.show_text  = ptext + tokens[i].show_text_pos;
        token.type       = TokenType(tokens[i].token_type);
        token.score      = tokens[i].score;
    }
    auto conn_tag = ptext + header.conn_tag_pos;

    tokenizer.set(int(header.vocab_size), std::move(vocab), std::move(texts), conn_tag, header.special_tokens);

    return true;
}

bool load_tensor(std::ifstream& file, const Block& block, const TransformerConfig& c, TransformerWeights& w) noexcept {
    Tensor* tensor = nullptr;
    switch (block.get_tensor_type()) {
        case TensorType::TOKEN_EMBD_TABLE:  tensor = &w.token_embedding_table;  break;
        case TensorType::OUTPUT_NORM:       tensor = &w.out_norm;               break;
        case TensorType::CLASSIFIER:        tensor = &w.classifier;             break;
        case TensorType::LAYER_INPUT_NORM:  tensor = &w.attn_norm;              break;
        case TensorType::LAYER_ATTN_Q:      tensor = &w.attn_q;                 break;
        case TensorType::LAYER_ATTN_K:      tensor = &w.attn_k;                 break;
        case TensorType::LAYER_ATTN_V:      tensor = &w.attn_v;                 break;
        case TensorType::LAYER_ATTN_O:      tensor = &w.attn_o;                 break;
        case TensorType::LAYER_MLP_GATE:    tensor = &w.ffn_1;                  break;
        case TensorType::LAYER_MLP_DOWN:    tensor = &w.ffn_2;                  break;
        case TensorType::LAYER_MLP_UP:      tensor = &w.ffn_3;                  break;
        case TensorType::LAYER_POST_NORM:   tensor = &w.ffn_norm;               break;
        default:
            tf_log_error("Unsupported tensor type:%d", int(block.get_tensor_type()));
            return false;
    }

    int layers  = 0;
    int rows    = 0;
    int columns = 0;

    auto shape  = block.get_tensor_shape();
    if (shape.size() >= 3) {
        layers  = int(shape[0]);
        rows    = int(shape[1]);
        columns = int(shape[2]);
    } else if (shape.size() >= 2) {
        rows    = int(shape[0]);
        columns = int(shape[1]);
    } else {
        columns = int(shape[0]);
    }

    int layer_id = block.get_layer_id();
    if (layer_id <= 0) {
        if (block.is_layer_tensor()) {
            layers = c.n_layers;
            if (rows <= 0) {
                rows = 1;
            }
        }
        QuantType qtype = QuantType::NONE;
        if (auto dtype = block.get_data_type(); dtype == DataType::INT16) {
            qtype = QuantType::INT16;
        } else if (dtype == DataType::INT8) {
            qtype = QuantType::INT8;
        }
        tensor->reset(columns, rows, layers, qtype, c.quant_group_size);
        if (!tensor->reserve_memory()) {
            tf_log_error("Out of memory for tensor:%s with size:%lu", block.get_name(), tensor->memory_size());
            return false;
        }
    }
    if (block.is_layer_tensor()) {
        if (!(*tensor)[layer_id].read_data(file)) {
            return false;
        }
    } else {
        if (!tensor->read_data(file)) {
            return false;
        }
    }
    return true;
}

bool TransformerModel::load_flm(std::string_view model_path) noexcept {
    std::ifstream file(model_path.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        tf_log_error("Failed to open gguf file:%s", model_path.data());
        return false;
    }
    size_t file_size = size_t(file.tellg());
    file.seekg(0, std::ios::beg);

    FileHeader header;
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        tf_log_error("Failed to read file header");
        return false;
    }
    if (header.file_tag != FLM_FILE_TAG) {
        tf_log_error("Not a valid model file. Invalid file tag:%08X", header.file_tag);
        return false;
    }
    if (is_debug) {
        tf_log_debug("FLM version:%d.%d.%d", int(header.version.v1), int(header.version.v2), int(header.version.v3));
    }

    Block block;
    for (size_t file_pos = sizeof(header); file_pos < file_size; file_pos += block.get_block_size()) {
        file.seekg(file_pos, std::ios::beg);
        if (!block.read_header(file)) {
            tf_log_error("Reading block header error at file_pos:%lu", size_t(file.tellg()));
            return false;
        }

        std::string_view name = block.get_name();
        char desc[256] = {};
        if (name == "model_config") {
            if (is_debug) {
                tf_log_debug("Loading model config ...");
            }
            if (!load_config(file, file_pos + block.header_size, block.lb.data_size, conf)) {
                tf_log_error("Loading config error at file_pos:%lu", file_pos);
                return false;
            }
        } else if (name == "tokenizer") {
            if (is_debug) {
                tf_log_debug("Loading tokenizer ...");
            }
            if (!load_tokenizer(file, tokenizer)) {
                tf_log_error("Loading tokenizer error at file_pos:%lu", file_pos);
                return false;
            }
        } else if (block.is_tensor()) {
            if (is_debug) {
                block.get_tensor_info(desc, true);
                tf_log_debug("Loading %s", desc);
            }
            if (!load_tensor(file, block, conf, weights)) {
                tf_log_error("Loading tensor:%s error at file_pos:%lu", block.get_name(), file_pos);
                return false;
            }
        } else {
            if (is_debug) {
                block.get_block_info(desc);
                tf_log_debug("Loading %s", desc);
            }
        }
    }

    if (is_debug) {
        print_summary();
    }

    return true;
}

} // namespace cpuft

