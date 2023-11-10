/*************************************************************************************
   Author: Liu Shaofeng
     Date: 2023/10/16
**************************************************************************************/

#pragma once

#include <memory>
#include <span>
#include <vector>
#include <string_view>
#include <mutex>

#include "threadparallel.hpp"
#include "alignmem.hpp"
#include "tf_operators.h"
#include "sampler.h"
#include "model_loader.h"
#include "tensor.h"
#include "log.h"
#include "ftdebug.h"

namespace cpuft {

class ParallelTransformer {
    enum class TaskType;
    struct Task;
    struct GlobalWeights { // Weight tensors that are used during inferencing in main thread.
        Tensor  token_embedding_table;  // (vocab_size, dim)
        Tensor  rms_att_weight;         // (layer, dim) rmsnorm weights
        Tensor  rms_ffn_weight;         // (layer, dim)
        Tensor  rms_final_weight;       // (dim,)
        Tensor  rope_freq_cis;
    };
    struct TransformerRuntime { // Global runtime data used during inferencing in main thread.
        std::unique_ptr<int[]> tokens;
        int  num_total_tokens  = 0;
        int  num_prompt_tokens = 0;

        AlignedMemory<char, 64> buf;
    };
    struct ThreadRuntime { // Runtime data used during inferencing in each thread of the thread pool.
        Tensor k_cache;
        Tensor v_cache;

        AlignedMemory<char, 64> buf;
    };

    // Weight tensors used in threads of the thread pool.
    struct ThreadTransformerWeights {
        Tensor  qkv;
        Tensor  attn_o;
        Tensor  ffn_1;
        Tensor  ffn_2;
        Tensor  ffn_3;
        Tensor  cls;
    };
    struct ThreadWeightsConfig {
        int qkv_offset;
        int attn_o_offset;
        int ffn_13_offset;
        int ffn_2_offset;
        int cls_offset;
        int kv_heads_offset;
    };

    // Thread local data for each thread of the thread pool.
    struct ThreadData {
        ThreadWeightsConfig      c;
        ThreadTransformerWeights w;
        ThreadRuntime            r;
    };

    using TransformerThreadParallel = ThreadParallel<ParallelTransformer, TransformerModel, ThreadData>;
    friend TransformerThreadParallel;

public:
    ParallelTransformer(bool debug) : _debug(debug) {}

    bool load(std::string_view ckpt_path, std::string_view tknr_path, ModelFileType mft, QuantType qt=QuantType::INT8,
            int num_threads=-1, bool use_numa=true, int max_batch_size=64, uint64_t rand_seed=0);

    std::vector<int> encode(const char* prompt);

    bool generate(const char* prompt,
            std::function<bool(const char* text, int num_input_tokens, int num_output_tokens, bool ended)> cb,
            int max_tokens=512, float temperature=1.0f, float topp=0.9f);

    bool generate(const std::vector<int>& prompt_tokens,
            std::function<bool(std::span<const int> tokens, int index, bool ended)> cb,
            int max_tokens=512, float temperature=1.0f, float topp=0.9f);

private:
    Tensor forward(std::span<const int> tokens, int pos);

    inline void execute(TaskType tt, short pos, short layer, Tensor* t1, Tensor* t2=nullptr, Tensor *t3=nullptr) noexcept {
        _task.tt    = tt;
        _task.pos   = pos;
        _task.layer = layer;
        _task.t1    = t1;
        _task.t2    = t2;
        _task.t3    = t3;
        _tp.notify_task(tt2gid(tt));
        _tp.wait();
    }
    inline void wait() noexcept {
        _tp.wait();
    }

    bool parallel_global_init(const TransformerModel& gd, int num_threads, bool use_numa, ThreadGroupBuilder& tgb);
    bool parallel_thread_init(const std::vector<ThreadGroupInfo>& tgis, const TransformerModel& gd, ThreadData& td);

    void parallel_process_task(ThreadData& td, const ThreadGroupInfo& tgi) {
        switch (_task.tt) {
            case TaskType::QKV:    return execute_qkv   (td, _task, tgi);
            case TaskType::ATTN:   return execute_attn  (td, _task, tgi);
            case TaskType::ATTN_O: return execute_attn_o(td, _task, tgi);
            case TaskType::FFN13:  return execute_ffn13 (td, _task, tgi);
            case TaskType::FFN2:   return execute_ffn2  (td, _task, tgi);
            case TaskType::CLS:    return execute_cls   (td, _task, tgi);
            default:
                ft_log_error("Invalid task type:%d", int(_task.tt));
                throw std::runtime_error("invalid task type");
        }
    }
    void execute_qkv   (ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;
    void execute_attn  (ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;
    void execute_attn_o(ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;
    void execute_ffn13 (ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;
    void execute_ffn2  (ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;
    void execute_cls   (ThreadData& td, Task t, const ThreadGroupInfo& tgi) const noexcept;

    inline short tt2gid(TaskType tt) const noexcept {
        return _tt_gid_map[int(tt)];
    }
    inline void set_tt_gid(TaskType tt, short gid) noexcept {
        _tt_gid_map[int(tt)] = gid;
    }

private:
    enum class TaskType {
        NONE = 0,
        QKV,
        ATTN,
        ATTN_O,
        FFN13,
        SWIGLU,
        FFN2,
        CLS,

        _MAX_
    };
    struct Task {
        TaskType      tt;
        int           pos;
        int           layer;
        Tensor*       t1;
        Tensor*       t2;
        Tensor*       t3;
    };

    short   _tt_gid_map[int(TaskType::_MAX_)] = {};
    Task    _task;

    TransformerThreadParallel _tp;

    Tokenizer   _tkn;
    Sampler     _sampler;
    QuantType   _qtype = QuantType::INT8;

    TransformerConfig    _tfc;
    GlobalWeights        _tfw;
    TransformerRuntime   _tfr;

    int     _num_threads = 0;
    int     _max_batch_size = 1;
    bool    _use_numa = false;
    bool    _debug = false;
};

} // namespace cpuft
