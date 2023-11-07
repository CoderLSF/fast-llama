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

#include "log.h"
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

void TransformerModel::print_summary() const noexcept {
    tf_log_debug("\t                   name:\x1b[33m%s\x1b[0m", conf.name.c_str());
    tf_log_debug("\t           architecture:\x1b[33m%d\x1b[0m", int(conf.arch));
    tf_log_debug("\t            weight type:\x1b[33m%d\x1b[0m", int(conf.wtype));
    tf_log_debug("");
    tf_log_debug("\t            max_seq_len:\x1b[33m%d\x1b[0m", conf.max_seq_len);
    tf_log_debug("\t             vocab_size:\x1b[33m%d\x1b[0m", conf.vocab_size);
    tf_log_debug("\t               n_layers:\x1b[33m%d\x1b[0m", conf.n_layers);
    tf_log_debug("\t                    dim:\x1b[33m%d\x1b[0m", conf.dim);
    tf_log_debug("\t                 kv_dim:\x1b[33m%d\x1b[0m", conf.kv_dim);
    tf_log_debug("\t                n_heads:\x1b[33m%d\x1b[0m", conf.n_heads);
    tf_log_debug("\t             n_kv_heads:\x1b[33m%d\x1b[0m", conf.n_kv_heads);
    tf_log_debug("\t              head_size:\x1b[33m%d\x1b[0m", conf.head_size);
    tf_log_debug("\t             hidden_dim:\x1b[33m%d\x1b[0m", conf.hidden_dim);
    tf_log_debug("");
    tf_log_debug("\t   rope_dimension_count:\x1b[33m%d\x1b[0m", conf.rope_dimension_count);
    tf_log_debug("\t         rope_freq_base:\x1b[33m%g\x1b[0m", conf.rope_freq_base);
    tf_log_debug("\t layer_norm_rms_epsilon:\x1b[33m%g\x1b[0m", conf.layer_norm_rms_epsilon);
    tf_log_debug("");
    tf_log_debug("\t           bos_token_id:\x1b[33m%d\x1b[0m", tokenizer.bos_token_id());
    tf_log_debug("\t           eos_token_id:\x1b[33m%d\x1b[0m", tokenizer.eos_token_id());
    tf_log_debug("\t           pad_token_id:\x1b[33m%d\x1b[0m", tokenizer.pad_token_id());
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
