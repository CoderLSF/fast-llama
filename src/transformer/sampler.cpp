
#include "sampler.h"

#include <stdlib.h>
#include <math.h>

#include "tf_operators.h"

namespace cpuft {

void Sampler::build(int vocab_size, uint64_t rng_seed) {
    _vocab_size = vocab_size;
    _rng_state  = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    _probindex = (ProbIndex*)aligned_alloc(64, vocab_size * sizeof(ProbIndex));
}

Sampler::~Sampler() {
    if (_probindex != nullptr) {
        free(_probindex);
    }
}

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int Sampler::sample_mult(float* probabilities, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < _vocab_size; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return _vocab_size - 1; // in case of rounding errors
}

int Sampler::compare_prob_index(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int Sampler::sample_topp(float* probabilities, float topp, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (_vocab_size - 1);
    for (int i = 0; i < _vocab_size; i++) {
        if (probabilities[i] >= cutoff) {
            _probindex[n0].index = i;
            _probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(_probindex, n0, sizeof(ProbIndex), compare_prob_index);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += _probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += _probindex[i].prob;
        if (r < cdf) {
            return _probindex[i].index;
        }
    }
    return _probindex[last_idx].index; // in case of rounding errors
}

int Sampler::sample(Tensor& t, float temperature, float topp) {
    auto logits = t.float_data();
    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, _vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q < _vocab_size; q++) { logits[q] /= temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, _vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&_rng_state);
        // we sample from this distribution to get the next token
        if (topp <= 0 || topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, topp, coin);
        }
    }
    return next;
}


} // namespace cpuft
