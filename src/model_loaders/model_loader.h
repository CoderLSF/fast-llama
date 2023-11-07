#pragma once

#include "tensor.h"
#include "tokenizer.h"
#include "log.h"

namespace cpuft {

enum class ModelArchitecture {
    NONE = 0,
    LLAMA = 1,
};

enum class WeightType {
    FP32    = 0,
    FP16    = 1,
    Q16_0   = 2,
    Q8_0    = 3,
};

enum class ModelFileType {
    NONE = 0,
    GGUF,
    LLAMA2C,

    _MAX_,
};

struct TransformerConfig {
    std::string         name;
    ModelArchitecture   arch;
    WeightType          wtype;

    int     n_layers;
    int     dim;
    int     kv_dim;
    int     hidden_dim;
    int     n_heads;
    int     n_kv_heads;
    int     head_size;
    int     vocab_size;
    int     max_seq_len;

    int     rope_dimension_count    = 0;
    float   rope_freq_base          = 10000.;
    float   layer_norm_rms_epsilon  = 1e-5;
    int     quant_group_size        = 64;
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
    TransformerConfig   conf;
    TransformerWeights  weights;
    Tokenizer           tokenizer;

public:
    bool load(std::string_view checkpoint_path, std::string_view tokenizer_path, ModelFileType mft, bool print_detail=false) {
        switch (mft) {
        case ModelFileType::GGUF:
            return load_gguf(checkpoint_path, print_detail);
        case ModelFileType::LLAMA2C:
            return load_llama2c(checkpoint_path, tokenizer_path, print_detail);
        default:
            tf_log_error("Unsupported model file type:%d", int(mft));
            return false;
        }
        return false;
    }

    void print_summary() const noexcept;

protected:
    bool load_gguf   (std::string_view checkpoint_path, bool print_detail) noexcept;
    bool load_llama2c(std::string_view checkpoint_path, std::string_view tokenizer_path, bool print_detail) noexcept;
};

} // namespace cpuft
