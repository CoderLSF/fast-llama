#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <functional>
#include <string>
#include <string_view>
#include <filesystem>

#include "log.h"
#include "console.h"
#include "model_loader.h"

namespace cpuft {

bool TransformerWeights::build_rope_freq_cis() {
    if (!this->rope_freq_cis.reserve_memory()) {
        tf_log_error("Out of memory for rope_freq_cis");
        return false;
    }

    auto pd = this->rope_freq_cis.float_data();
    int head_size = this->rope_freq_cis.columns();
    for (int i = 0, n = this->rope_freq_cis.total_rows(); i < n; ++i, pd += head_size) {
        for (int j = 0; j < head_size; j += 2) {
            float v = i / powf(10000.0f, i / (float)head_size);
            pd[j]   = cosf(v);
            pd[j+1] = sinf(v);
        }
    }

    return true;
}

ModelFileType TransformerModel::detect_file_type(std::string_view model_path) noexcept {
    if (!std::filesystem::is_regular_file(model_path)) {
        tf_log_error("Not a regular file:%s", model_path.data());
        return ModelFileType::UNKNOWN;
    }
    std::ifstream file(model_path.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        tf_log_error("Cannot open model file:%s", model_path.data());
        return ModelFileType::UNKNOWN;
    }
    file.seekg(0, std::ios::beg);

    char  buffer[128];
    if (!file.read(buffer, sizeof(buffer))) {
        tf_log_error("Cannot read model file:%s", model_path.data());
        return ModelFileType::UNKNOWN;
    }
    if (is_valid_flm_header(buffer)) {
        return ModelFileType::FLM;
    } else if (is_valid_gguf_header(buffer)) {
        return ModelFileType::GGUF;
    } else if (is_valid_llama2c_header(buffer)) {
        return ModelFileType::LLAMA2C;
    } else {
        return ModelFileType::UNKNOWN;
    }
}

bool TransformerModel::load(std::string_view checkpoint_path, std::string_view tokenizer_path, ModelFileType mft) {
    if (mft == ModelFileType::UNKNOWN) {
        mft = detect_file_type(checkpoint_path);
    }

    switch (mft) {
    case ModelFileType::FLM:
        return load_flm(checkpoint_path);
    case ModelFileType::GGUF:
        return load_gguf(checkpoint_path);
    case ModelFileType::LLAMA2C:
        return load_llama2c(checkpoint_path, tokenizer_path);
    default:
        tf_log_error("Unsupported model file type:%d", int(mft));
        return false;
    }
    return false;
}

void TransformerModel::print_summary() const noexcept {
    Console con;
    tf_log_debug("\t                   name:%s%s%s", con.yellow(), conf.name.c_str(), con.endtag());
    tf_log_debug("\t           architecture:%s%d%s", con.yellow(), int(conf.arch), con.endtag());
    tf_log_debug("\t            weight type:%s%d%s", con.yellow(), int(conf.wtype), con.endtag());
    tf_log_debug("");
    tf_log_debug("\t            max_seq_len:%s%d%s", con.yellow(), conf.max_seq_len, con.endtag());
    tf_log_debug("\t             vocab_size:%s%d%s", con.yellow(), conf.vocab_size, con.endtag());
    tf_log_debug("\t               n_layers:%s%d%s", con.yellow(), conf.n_layers, con.endtag());
    tf_log_debug("\t                    dim:%s%d%s", con.yellow(), conf.dim, con.endtag());
    tf_log_debug("\t                 kv_dim:%s%d%s", con.yellow(), conf.kv_dim, con.endtag());
    tf_log_debug("\t                n_heads:%s%d%s", con.yellow(), conf.n_heads, con.endtag());
    tf_log_debug("\t             n_kv_heads:%s%d%s", con.yellow(), conf.n_kv_heads, con.endtag());
    tf_log_debug("\t              head_size:%s%d%s", con.yellow(), conf.head_size, con.endtag());
    tf_log_debug("\t             hidden_dim:%s%d%s", con.yellow(), conf.hidden_dim, con.endtag());
    tf_log_debug("");
    tf_log_debug("\t   rope_dimension_count:%s%d%s", con.yellow(), conf.rope_dimension_count, con.endtag());
    tf_log_debug("\t         rope_freq_base:%s%g%s", con.yellow(), conf.rope_freq_base, con.endtag());
    tf_log_debug("\t layer_norm_rms_epsilon:%s%g%s", con.yellow(), conf.layer_norm_rms_epsilon, con.endtag());
    tf_log_debug("");
    tf_log_debug("\t       quant_group_size:%s%d%s", con.yellow(), conf.quant_group_size, con.endtag());
    tf_log_debug("");
    tf_log_debug("\t           bos_token_id:%s%d%s", con.yellow(), tokenizer.bos_token_id(), con.endtag());
    tf_log_debug("\t           eos_token_id:%s%d%s", con.yellow(), tokenizer.eos_token_id(), con.endtag());
    tf_log_debug("\t           pad_token_id:%s%d%s", con.yellow(), tokenizer.pad_token_id(), con.endtag());
}

} // namespace cpuft

#ifdef DEBUG_MODEL_LOADER
int main(int argc, const char** argv) {
    std::string_view ckpt_path = "";
    std::string_view tknr_path = "";
    cpuft::ModelFileType mft = cpuft::ModelFileType::GGUF;
    for (int i = 0; i < argc;) {
        auto arg = std::string_view(argv[i++]);
        if (arg == "-c" || arg == "--checkpoint") {
            ckpt_path = argv[i++];
        } else if (arg == "-z" || arg == "--tokenizer") {
            tknr_path = argv[i++];
        } else if (arg == "-f" || arg == "--file-type") {
            auto mtfs = argv[i++];
            if (strcasecmp(mtfs, "gguf") == 0) {
                mft = cpuft::ModelFileType::GGUF;
            } else if (strcasecmp(mtfs, "llama2c") == 0) {
                mft = cpuft::ModelFileType::LLAMA2C;
            }
        }
    }
    if (ckpt_path.empty()) {
        fprintf(stderr, "Usage: %s -c|--checkpoint <checkpoint path> [-z|--tokenizer <tokenizer path>]\n", argv[0]);
        return -1;
    }

    cpuft::TransformerModel model;
    if (!model.load(ckpt_path, tknr_path, mft)) {
        fprintf(stderr, "Loading failed\n");
        return -1;
    }

    const char* text = "Hi, there! How are you going lately?";

    std::unique_ptr<int[]> tokens(new int[2048]);

    int num_tokens = model.tokenizer.encode(text, {tokens.get(), 2048});
    fprintf(stderr, "num_tokens:%d\n", num_tokens);
    if (num_tokens < 1) {
        exit(-1);
    }

    std::unique_ptr<char[]> sbuf(new char[2048]);
    auto s = model.tokenizer.decode({tokens.get(), size_t(num_tokens)}, {sbuf.get(), 2048});
    std::cout << "decoded text:\x1b[32m" << s << "\x1b[0m" << std::endl;

    return 0;
}
#endif

#ifdef TEST_MODEL_LOADER
int main(int argc, const char** argv) {
    using namespace cpuft;

    auto model_path = argv[1];

    TransformerModel tfm(true);
    bool ret = tfm.load(model_path);
    if (!ret) {
        fprintf(stderr, "Failed to load model:%s\n", model_path);
    }

    return 0;
}
#endif
