/* Inference for Llama-2 Transformer model in pure C */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <fstream>
#include <algorithm>

#include "log.h"
#include "tf_operators.h"
#include "model_loader.h"

// ----------------------------------------------------------------------------
// Transformer model

namespace cpuft {

struct Llama2cConfig {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int max_seq_len; // max sequence length
};

bool TransformerModel::is_valid_llama2c_header(std::span<const char> file_header_data) noexcept {
    if (file_header_data.size() < sizeof(Llama2cConfig)) {
        return false;
    }
    auto& conf = *reinterpret_cast<const Llama2cConfig*>(file_header_data.data());
    return conf.dim >= 512 && conf.dim <= 64000 && conf.hidden_dim >= 512 && conf.hidden_dim <= 64000
            && conf.n_layers > 0 && conf.n_layers < 512 && conf.n_heads >= 4 && conf.n_heads <= 1024
            && conf.n_kv_heads >= 1 && conf.n_kv_heads <= conf.n_heads
            && conf.vocab_size >= 1000 && conf.vocab_size < (256 << 10);
}

bool TransformerModel::load_llama2c(std::string_view ckpt_path, std::string_view tokenizer_path) noexcept {
    std::ifstream file(ckpt_path.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        tf_log_error("Failed to open model file:%s", ckpt_path.data());
        return false;
    }
    file.tellg();
    file.seekg(0, std::ios::beg);

    Llama2cConfig raw_conf;
    if (is_debug) tf_log_debug("Loading model configurations ...");
    if (!file.read(reinterpret_cast<char*>(&raw_conf), sizeof(raw_conf))) {
        tf_log_error("Reading model file error:%s", ckpt_path.data());
        return false;
    }
    conf.dim         = raw_conf.dim;
    conf.hidden_dim  = raw_conf.hidden_dim;
    conf.n_layers    = raw_conf.n_layers;
    conf.n_heads     = raw_conf.n_heads;
    conf.n_kv_heads  = raw_conf.n_kv_heads;
    conf.vocab_size  = raw_conf.vocab_size;
    conf.max_seq_len = raw_conf.max_seq_len;
    conf.vocab_size  = abs(raw_conf.vocab_size);
    conf.head_size   = conf.dim / conf.n_heads;
    conf.kv_dim      = conf.head_size * conf.n_kv_heads;

    conf.arch = ModelArchitecture::LLAMA;

    bool shared_weights = raw_conf.vocab_size > 0;

    print_summary();

    if (is_debug) tf_log_debug("Loading tokenizer from:[%s] ...", tokenizer_path.data());
    if (!tokenizer.load(tokenizer_path, conf.vocab_size)) {
        tf_log_error("Failed to load tokenizer from:%s", tokenizer_path.data());
        return false;
    }

    if (is_debug) tf_log_debug("Loading model weights from:[%s] ...", ckpt_path.data());

    weights.token_embedding_table.reset(conf.dim, conf.vocab_size);

    weights.attn_norm.reset(conf.dim, conf.n_layers);
    weights.attn_q.reset(conf.dim, conf.dim,    conf.n_layers);
    weights.attn_k.reset(conf.dim, conf.kv_dim, conf.n_layers);
    weights.attn_v.reset(conf.dim, conf.kv_dim, conf.n_layers);
    weights.attn_o.reset(conf.dim, conf.dim,    conf.n_layers);
    weights.ffn_1.reset(conf.dim,        conf.hidden_dim, conf.n_layers);
    weights.ffn_2.reset(conf.hidden_dim, conf.dim,        conf.n_layers);
    weights.ffn_3.reset(conf.dim,        conf.hidden_dim, conf.n_layers);
    weights.ffn_norm.reset(conf.dim, conf.n_layers);
    weights.out_norm.reset(conf.dim);
    weights.classifier.reset(conf.dim, conf.vocab_size);
    weights.rope_freq_cis.reset(conf.head_size, conf.max_seq_len);

    if (!weights.token_embedding_table.reserve_memory()
            || !weights.attn_q.reserve_memory()
            || !weights.attn_k.reserve_memory()
            || !weights.attn_v.reserve_memory()
            || !weights.attn_o.reserve_memory()
            || !weights.attn_norm.reserve_memory()
            || !weights.ffn_1.reserve_memory()
            || !weights.ffn_2.reserve_memory()
            || !weights.ffn_3.reserve_memory()
            || !weights.ffn_norm.reserve_memory()
            || !weights.out_norm.reserve_memory()
            || !weights.rope_freq_cis.reserve_memory()
            || (!shared_weights && !weights.classifier.reserve_memory())) {
        tf_log_error("Out of memory for loading model:%s", ckpt_path.data());
        return false;
    }

    if (!file.read(weights.token_embedding_table.data(), weights.token_embedding_table.memory_size())) {
        tf_log_error("Failed to read token embeddings while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.attn_norm.data(), weights.attn_norm.memory_size())) {
        tf_log_error("Failed to read weights of attn_norm while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.attn_q.data(), weights.attn_q.memory_size())) {
        tf_log_error("Failed to read weights of Q while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.attn_k.data(), weights.attn_k.memory_size())) {
        tf_log_error("Failed to read weights of K while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.attn_v.data(), weights.attn_v.memory_size())) {
        tf_log_error("Failed to read weights of V while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.attn_o.data(), weights.attn_o.memory_size())) {
        tf_log_error("Failed to read weights of O while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.ffn_norm.data(), weights.ffn_norm.memory_size())) {
        tf_log_error("Failed to read weights of FFN RMSNorm while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.ffn_1.data(), weights.ffn_1.memory_size())) {
        tf_log_error("Failed to read weights of FFN Gate while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.ffn_2.data(), weights.ffn_2.memory_size())) {
        tf_log_error("Failed to read weights of FFN Down while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.ffn_3.data(), weights.ffn_3.memory_size())) {
        tf_log_error("Failed to read weights of FFN Up while loading:%s", ckpt_path.data());
        return false;
    }
    if (!file.read(weights.out_norm.data(), weights.out_norm.memory_size())) {
        tf_log_error("Failed to read weights of output rms norm while loading:%s", ckpt_path.data());
        return false;
    }
    {
        size_t n = conf.head_size*conf.max_seq_len/2;
        std::unique_ptr<float[]> real(new float[n]);
        std::unique_ptr<float[]> imag(new float[n]);
        if (real == nullptr || imag == nullptr) {
            tf_log_error("Out of memory for rope freq cis while loading:%s", ckpt_path.data());
            return false;
        }
        if (!file.read(reinterpret_cast<char*>(real.get()), sizeof(float)*n)
                || !file.read(reinterpret_cast<char*>(imag.get()), sizeof(float)*n)) {
            tf_log_error("Failed to read rope freq cis while loading:%s", ckpt_path.data());
            return false;
        }
        auto pdst = weights.rope_freq_cis.float_data();
        for (size_t i = 0; i < n; ++i) {
            pdst[i*2  ] = real[i];
            pdst[i*2+1] = imag[i];
        }
    }
    if (shared_weights) {
        weights.classifier = weights.token_embedding_table.slice(0, conf.vocab_size);
    } else if (!file.read(weights.classifier.data(), weights.classifier.memory_size())) {
        tf_log_error("Failed to read weights of final classifier while loading:%s", ckpt_path.data());
        return false;
    }

    if (is_debug) tf_log_debug("Model loaded");
    return true;
}

}

