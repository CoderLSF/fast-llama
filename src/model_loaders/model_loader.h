#pragma once

#include "tensor.h"
#include "tokenizer.h"
#include "log.h"

namespace cpuft {

enum class ModelArchitecture {
    NONE  = 0,
    LLAMA = 1,
};

enum class WeightType {
    FP32    = 0,
    FP16    = 1,
    Q16_0   = 2,
    Q8_0    = 3,
};

enum class ActivationType {
    NONE    = 0,
    SILU    = 1,
    SWIGLU  = 2,
};

enum class ModelFileType {
    UNKNOWN = 0,
    FLM,
    GGUF,
    LLAMA2C,

    _MAX_,
};

struct TransformerConfig {
    std::string         name;
    ModelArchitecture   arch;
    WeightType          wtype = WeightType::FP32;

    ActivationType      act_type = ActivationType::SWIGLU;

    int         n_layers = 0;
    int         dim = 0;
    int         kv_dim = 0;
    int         hidden_dim = 0;
    int         n_heads = 0;
    int         n_kv_heads = 0;
    int         head_size = 0;
    int         vocab_size = 0;
    int         max_seq_len = 0;

    int         rope_dimension_count    = 0;
    float       rope_freq_base          = 10000.;
    float       layer_norm_rms_epsilon  = 1e-5;

    QuantType   quant_type              = QuantType::NONE;
    int         quant_group_size        = 64;
};

struct TransformerWeights {
    Tensor      token_embedding_table;

    Tensor      attn_q;
    Tensor      attn_k;
    Tensor      attn_v;
    Tensor      attn_o;
    Tensor      attn_norm;

    Tensor      ffn_1;
    Tensor      ffn_2;
    Tensor      ffn_3;
    Tensor      ffn_norm;;

    Tensor      out_norm;
    Tensor      classifier;

    Tensor      rope_freq_cis;

    bool build_rope_freq_cis();
};

class TransformerModel {
public:
    bool                is_debug = false;
    TransformerConfig   conf;
    TransformerWeights  weights;
    Tokenizer           tokenizer;

public:
    TransformerModel() {}
    explicit TransformerModel(bool is_debug_) : is_debug(is_debug_) {}

    bool load(std::string_view checkpoint_path,
              std::string_view tokenizer_path="",
              ModelFileType mft=ModelFileType::UNKNOWN);

    void print_summary() const noexcept;

protected:
    static ModelFileType detect_file_type(std::string_view model_path) noexcept;
    static bool is_valid_flm_header(std::span<const char> file_header) noexcept;
    static bool is_valid_gguf_header(std::span<const char> file_header) noexcept;
    static bool is_valid_llama2c_header(std::span<const char> file_header) noexcept;

    bool load_flm    (std::string_view model_path) noexcept;
    bool load_gguf   (std::string_view model_path) noexcept;
    bool load_llama2c(std::string_view checkpoint_path, std::string_view tokenizer_path) noexcept;
};

} // namespace cpuft
