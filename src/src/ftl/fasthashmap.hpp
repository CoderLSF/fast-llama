#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <type_traits>

namespace ftl { // ftl = fast template libraries

uint64_t murmurhash64(const void* key, int len) {
    constexpr uint64_t  MURMURHASH64A_CONSTANT = 0xc6a4a7935bd1e995;
    constexpr int       MURMURHASH64A_R = 47;

    uint64_t h = (len * MURMURHASH64A_CONSTANT);
    auto data = reinterpret_cast<const uint64_t*>(key);
    if (len >= 8) {
        const uint64_t* end = data + (len / 8);
        while (data != end) {
            uint64_t k = *data++;
            k *= MURMURHASH64A_CONSTANT;
            k ^= k >> MURMURHASH64A_R;
            k *= MURMURHASH64A_CONSTANT;
            h ^= k;
            h *= MURMURHASH64A_CONSTANT;
        }
    }
    if (auto r = (uint64_t(len) & 7) * 8; r != 0) {
        h ^= (*data) & ((1ul << r) - 1);
    }
    h ^= h >> MURMURHASH64A_R;
    h *= MURMURHASH64A_CONSTANT;
    h ^= h >> MURMURHASH64A_R;
    return h;
}

template <typename T, typename = void>
struct has_data_and_size : std::false_type {};

template <typename T>
struct has_data_and_size<T, std::void_t<decltype(std::declval<T>().data()), decltype(std::declval<T>().size())>> : std::true_type {};

template<typename T>
struct is_char_pointer {
    using T_no_cv = typename std::remove_cv<T>::type;
    static constexpr bool value = std::is_same<char*, T_no_cv>::value || std::is_same<const char*, T_no_cv>::value;
};

template<typename T>
struct is_string_type {
    static constexpr bool value = is_char_pointer<T>::value || std::is_same<T, std::string>::value || std::is_same<T, std::string_view>::value;
};


template <typename T>
class Hasher {
public:
    inline uint64_t operator()(const T& v) noexcept {
        static_assert((std::is_integral<T>::value || std::is_floating_point<T>::value
                || has_data_and_size<T>::value || is_char_pointer<T>::value), "Unsupported type for Hasher");

        if constexpr (std::is_integral<T>::value) {
            return static_cast<uint64_t>(v);
        } else if constexpr (std::is_floating_point<T>::value) {
            union UnionValue {
                uint64_t    uv;
                T           fv;
            };
            return reinterpret_cast<const UnionValue*>(&v)->uv;
        } else if constexpr (is_string_type<T>::value) {
            if constexpr (is_char_pointer<T>::value) {
                return murmurhash64(v, strlen(v));
            } else {
                return murmurhash64(v.data(), v.size());
            }
        } else if constexpr (has_data_and_size<T>::value) {
            return murmurhash64(v.data(), v.size() * sizeof(v.data()[0]));
        } else {
            return 0ul;
        }
    }
};

template <typename KeyType, typename ValType, bool KEY_IS_BIG, bool VAL_IS_BIG>
struct KeyValue;

template<typename KeyType, typename ValType> 
struct KeyValue<KeyType, ValType, false, false> {
    KeyType key;
    ValType val;
    inline void set_key(const KeyType& k) {
        key = k;
    }
    inline void set_val(const ValType& v) {
        val = v;
    }
    inline const KeyType& get_key() const noexcept {
        return key;
    }
    inline const ValType& get_val() const noexcept {
        return val;
    }
};
template<typename KeyType, typename ValType> 
struct KeyValue<KeyType, ValType, false, true> {
    KeyType                     key;
    std::unique_ptr<ValType>    val;

    inline void set_key(const KeyType& k) {
        key = k;
    }
    inline void set_val(const ValType& v) {
        val.reset(new ValType(v));
    }
    inline const KeyType& get_key() const noexcept {
        return key;
    }
    inline const ValType& get_val() const noexcept {
        return *val;
    }
};
template<typename KeyType, typename ValType> 
struct KeyValue<KeyType, ValType, true, false> {
    std::unique_ptr<KeyType>    key;
    ValType                     val;

    inline void set_key(const KeyType& k) {
        key.reset(new KeyType(k));
    }
    inline void set_val(const ValType& v) {
        val = v;
    }
    inline const KeyType& get_key() const noexcept {
        return *key;
    }
    inline const ValType& get_val() const noexcept {
        return val;
    }
};
template<typename KeyType, typename ValType> 
struct KeyValue<KeyType, ValType, true, true> {
    std::unique_ptr<KeyType>    key;
    std::unique_ptr<ValType>    val;

    inline void set_key(const KeyType& k) {
        key.reset(new KeyType(k));
    }
    inline void set_val(const ValType& v) {
        val.reset(new ValType(v));
    }
    inline const KeyType& get_key() const noexcept {
        return *key;
    }
    inline const ValType& get_val() const noexcept {
        return *val;
    }
};

template <typename KeyType, typename ValType, typename Hasher=Hasher<KeyType>, bool STORE_HASH_AS_KEY=true>
class FastHashMap {
    using IndexKeyType = typename std::conditional<STORE_HASH_AS_KEY, uint64_t, KeyType>::type;
    using KeyValuePair = KeyValue<IndexKeyType, ValType, sizeof(IndexKeyType) >= 17, sizeof(ValType) >= 33>;
    static constexpr size_t NUM_RESIDENT_KV    = 48 / sizeof(KeyValuePair);
    static constexpr size_t EXPECT_LOAD_FACTOR = (NUM_RESIDENT_KV + 1) / 2;

    struct ListNode {
        KeyValuePair                kvs[2];
        std::unique_ptr<ListNode>   next;
    };

    struct Bucket {
        union {
            struct {
                uint8_t   hash_heads[7];
                uint8_t   kv_num = 0;
            };
            uint64_t hhu64;
        };

        union {
            char __kvd[48];
            struct {
                KeyValuePair                resident_kvs[NUM_RESIDENT_KV];
                std::unique_ptr<ListNode>   external_head;
            };
        };
    };
    static constexpr uint64_t PRIME = 39283918209382409ul;
public:
    FastHashMap() {}

    inline uint64_t hash2index(uint64_t hash) const noexcept {
        return ((hash < 37) ^ hash) % _hash_size;
    }
    inline uint8_t hash2head(uint64_t hash) const noexcept {
        return (hash * PRIME) & 0xFF;
    }

    bool build(const auto& data) noexcept {
        _hash_size = (data.size() + EXPECT_LOAD_FACTOR - 1) / EXPECT_LOAD_FACTOR;
        _hash_bkts.reset(new(std::nothrow) Bucket[_hash_size]);
        memset(_hash_bkts.get(), 0, _hash_size * sizeof(Bucket));
        for (auto& [k, v] : data) {
            auto hash = Hasher()(k);
            auto& bkt = _hash_bkts[hash2index(hash)];
            if (bkt.kv_num < NUM_RESIDENT_KV) {
                if constexpr (STORE_HASH_AS_KEY) {
                    bkt.resident_kvs[bkt.kv_num].set_key(hash); 
                } else {
                    bkt.resident_kvs[bkt.kv_num].set_key(k); 
                }
                bkt.resident_kvs[bkt.kv_num].set_val(v); 
            } else {
                KeyValuePair* kv;
                if (bkt.external_head == nullptr) {
                    bkt.external_head.reset(new ListNode);
                    kv = &bkt.external_head->kvs[0];
                } else {
                    auto node = bkt.external_head.get();
                    for (; node->next != nullptr; node = node->next.get());
                    if (((bkt.kv_num - NUM_RESIDENT_KV) & 1) == 0) {
                        node->next.reset(new ListNode);
                        kv = &node->next->kvs[0];
                    } else {
                        kv = &node->kvs[1];
                    }
                }
                if constexpr (STORE_HASH_AS_KEY) {
                    kv->set_key(hash);
                } else {
                    kv->set_key(k);
                }
                kv->set_val(v);
            }
            if (bkt.kv_num < 7) {
                bkt.hash_heads[bkt.kv_num] = hash2head(hash);
            }
            ++bkt.kv_num;
        }
        _data_size = data.size();
        return true;
    }
    void avg_list_size() const noexcept {
        size_t total = 0;
        for (size_t i = 0; i < _hash_size; ++i) {
            for (auto node = _hash_bkts[i].external_head.get(); node != nullptr; node = node->next.get()) {
                ++total;
            }
        }
        printf("average list length:%.2f\n", double(total) / _hash_size);
    }

    size_t size() const noexcept {
        return _data_size;
    }

    inline const ValType* find(const KeyType& k) const noexcept {
        auto hash = Hasher()(k);
        const auto& bkt = _hash_bkts[hash%_hash_size];

        static auto cmp = [&hash, &k](const KeyValuePair& kv) -> bool {
            if constexpr (STORE_HASH_AS_KEY) {
                return kv.get_key() == hash;
            } else if constexpr (is_char_pointer<KeyType>::value) {
                return strcmp(kv.get_key(), k) == 0;
            } else {
                return kv.get_key() == k;
            }
        };

        uint8_t  ih = 0;
        for (auto hh = hash2head(hash), nh = bkt.kv_num < 7 ? bkt.kv_num : 7; ih < nh; ++ih) {
            if (bkt.hash_heads[ih] == hh) {
                if (ih >= NUM_RESIDENT_KV) {
                    break;
                } else if (cmp(bkt.resident_kvs[ih])) {
                    return &bkt.resident_kvs[ih].get_val();
                }
            }
        }
        if (ih >= bkt.kv_num) {
            return nullptr;
        }

        auto node = bkt.external_head.get();
        for (uint8_t rn = (ih - NUM_RESIDENT_KV) / 2, i = 0; i < rn; ++i) {
            node = node->next.get();
        }
        
        uint8_t rn = ih - NUM_RESIDENT_KV;
        if (cmp(node->kvs[rn & 1])) {
            return &node->kvs[rn & 1].get_val();
        }
        if ((rn & 1) == 0 && cmp(node->kvs[1])) {
            return &node->kvs[1].get_val();
        }

        for (node = node->next.get(); node != nullptr; node = node->next.get()) {
            if (cmp(node->kvs[0])) {
                return &node->kvs[0].get_val();
            } else if (cmp(node->kvs[1])) {
                return &node->kvs[1].get_val();
            }
        }
        return nullptr;
    }

private:
    size_t                      _hash_size = 0;
    std::unique_ptr<Bucket[]>   _hash_bkts;
    size_t                      _data_size = 0;
};

} // namespace ftl

class Timer {
public:
    Timer() : _time_us(get_timestamp_us()) {}

    void reset() noexcept {
        _time_us = get_timestamp_us();
    }

    int64_t elapsed_us() const noexcept {
        return get_timestamp_us() - _time_us;
    }

    static int64_t get_timestamp_us() noexcept {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return (int64_t)tv.tv_sec * 1000000 + (int64_t)tv.tv_usec;
    }

private:
    int64_t _time_us;
};

#ifdef TEST_FAST_HASHMAP
int main() {
    using namespace ftl;

    uint64_t seed = 1;
    auto rand = [&seed]() -> int {
        constexpr uint64_t PRIME = 39283918209382409ul;
        seed = (seed * PRIME) ^ (seed >> 32);
        return static_cast<int>(seed & ((1ul << 31) - 1));
    };

    constexpr int NUM_VALS = 10000000;
    char sbuf[64];

    FastHashMap<std::string_view, int> fhm;
    std::unordered_map<std::string_view, int> um;
    
    constexpr int MAX_RV = 20000000;

    std::vector<std::string> svec;
    for (int i = 0; i < NUM_VALS; ++i) {
        auto rv = rand() % MAX_RV;
        svec.emplace_back(std::to_string(rv));
    }
    for (size_t i = 0; i < svec.size(); ++i) {
        std::string_view sv = {svec[i].c_str(), svec[i].size()};
        um[sv] = std::stoi(sv.data());
    }

    fhm.build(um);
    fhm.avg_list_size();
    printf("uom.size:%lu\tfhm.size:%lu\n", um.size(), fhm.size());

    constexpr size_t rounds = 8000000;
    int64_t hit = 0;
    int64_t res = 0;

    Timer tmr;
    int64_t cost_us;

    hit = 0;
    tmr.reset();
    for (size_t i = 0; i < rounds; ++i) {
        int sn = sprintf(sbuf, "%d", (rand() % MAX_RV));
        std::string_view sv{sbuf, size_t(sn)};
        auto r = fhm.find(sv);
        if (r != nullptr) {
            ++hit;
            res += *r;
        }
    }
    cost_us = tmr.elapsed_us();
    printf("[fhm] latancy:\x1b[32m%3ld\x1b[0mns\thit_rate:%5.2f%%\n", cost_us * 1000 / rounds, hit * 100. / rounds);

    hit = 0;
    tmr.reset();
    for (size_t i = 0; i < rounds; ++i) {
        int sn = sprintf(sbuf, "%d", rand() % MAX_RV);
        std::string_view sv{sbuf, size_t(sn)};
        auto r = um.find(sv);
        if (r != um.end()) {
            ++hit;
            res += r->second;
        }
    }
    cost_us = tmr.elapsed_us();
    printf("[uom] latancy:\x1b[32m%3ld\x1b[0mns\thit_rate:%5.2f%%\n", cost_us * 1000 / rounds, hit * 100. / rounds);
    if (res == 0) {
        printf("\n");
    }

    return 0;
}
#endif
