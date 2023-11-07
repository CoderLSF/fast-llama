/*********************************************************************************
 @author: Liu Shaofeng
 @date: 2023/10/16
 *********************************************************************************/

#include <stdint.h>
#include "tensor.h"

namespace cpuft {

struct Sampler {
public:
    ~Sampler();
    void build(int vocab_size, uint64_t rng_seed);
    int sample(Tensor& logits, float temperature=1.0f, float topp=0.9f);

protected:
    struct ProbIndex {
        float prob;
        int   index;
    };

    int sample_mult(float* probabilities, float coin);
    int sample_topp(float* probabilities, float topp, float coin);

    static int compare_prob_index(const void* a, const void* b);

private:
    int         _vocab_size;
    ProbIndex*  _probindex = nullptr;
    uint64_t    _rng_state;
};

} // namespace cpuft
