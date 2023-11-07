#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <numa.h>
#include <sched.h>

#include <immintrin.h>
#include <iostream>
#include <functional>

#include "tensor.h"
#include "utility.h"

#include "tf_operators.h"
#include "quant_operators.h"

using namespace cpuft;

void printer(const char* title, const auto* x, size_t n) {
    std::cout << title << "[";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << x[i] << ", ";
    }
    std::cout << "..., " << x[n-1] << "]" << std::endl;
};

void set_cpu_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void* numa_alloc(size_t size) {
    return numa_alloc_onnode(size, numa_node_of_cpu(sched_getcpu()));
}

template <typename T>
inline void matmul1(const T* mat1, const T* mat2, T* out, int m, int n, int w) {
    const T* x1 = mat1;
    for (int i = 0; i < m; ++i, x1 += n) {
        const T* x2 = mat2;
        for (int j = 0; j < w; ++j, x2 += n) {
            out[m*j+i] = cpuft::dot_product(x1, x2, n);
        }
    }
}

void test3();
void test4();
void test5();
void test6();
void test7();
void test8();

int main() {
    set_cpu_affinity(2);
    test8();
    return 0;
}

void test8() {
    int m = 4;
    int n = 8;
    Tensor x0(n*2, m*2);
    Tensor x1(n*2, m*2);
    Tensor x2(n*1, m);

    x0.random(-10, 10);
    x2.random(-10, 10);

    x0[0].print("x0 => ", false, false, x0.columns());
    x2[0].print("x2 => ", false, false, x2.columns());

    x1.copy_from(x0);
    x1.add(x2);
    x1[0].print("x1 => ", false, false, x1.columns());

    x1.copy_from(x0);
    x1.add(x2, n);
    x1[0].print("x1 => ", false, false, x1.columns());
}

void _test7(int m, int n, int layers, int k);
void test7() {
    _test7(1024, 1024, 64, 2);
}

void _test7(int m, int n, int layers, int k) {
    constexpr QuantType qt = QuantType::INT16;
    Tensor t1(n, 2);
    Tensor q1(n, 2, qt);
    Tensor d1(n, 2);
    Tensor w1(n, 1);
    Tensor o1(n, 2);

    t1.random(-60, 80);
    q1.quantize(t1);
    d1.dequantize(q1);
    w1.random(-0.5, 1);

    //t1[0].print("(0) [t1 row:0] ", false, false, 32);
    //q1[0].print("(0) [q1 row:0] ", false, false, 32);
    //d1[0].print("(0) [d1 row:0] ", false, false, 32);
    //t1[1].print("(0) [t1 row:1] ", false, false, 32);
    //q1[1].print("(0) [q1 row:1] ", false, false, 32);
    //d1[1].print("(0) [d1 row:1] ", false, false, 32);

    auto px = new float[n*k];
    Tensor tx(n, 2);
    tx.manage(px);
    tx.random(-0.5, 1);
    tx.print("tx => ");

    Tensor t2(t1.shape());
    t2.copy_from(t1);

    int smn = n;
    cpuft::softmax_sisd(t1.float_data(), t1.columns());
    t1.print("softmax_sisd => ", false, false, 30);
    cpuft::softmax(t2.float_data(), t2.columns());
    t2.print("softmax_simd => ", false, false, 30);
    cpuft::softmax_sisd(t1.float_data()+n, smn);
    //t1.softmax(smn);
    q1.softmax(smn);
    d1.softmax(smn);

    std::cout << std::endl;
    t1[0].print("(1) [t1 row:0] ", false, false, 32);
    q1[0].print("(1) [q1 row:0] ", false, false, 32);
    d1[0].print("(1) [d1 row:0] ", false, false, 32);
    std::cout << std::endl;
    t1[1].print("(1) [t1 row:1] ", false, false, 32);
    q1[1].print("(1) [q1 row:1] ", false, false, 32);
    d1[1].print("(1) [d1 row:1] ", false, false, 32);

    return;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    o1.zero();
    cpuft::rmsnorm(o1.float_data(), t1.float_data(), w1.float_data(), n);
    cpuft::rmsnorm(o1.float_data()+n, t1.float_data()+n, w1.float_data(), n);
    o1[0].print("(2) [o1 row:0] ", false, false, 32);
    o1[1].print("(2) [o1 row:1] ", false, false, 32);

    o1.zero();
    o1.rmsnorm(t1, w1);
    o1[0].print("(2) [o1 row:0] ", false, false, 32);
    o1[1].print("(2) [o1 row:1] ", false, false, 32);
}


void _test6(int m, int n, int layers, int k);
void test6() {
    _test6(1024, 1024, 64, 2);
}

void _test6(int m, int n, int layers, int k) {
    constexpr QuantType qt = QuantType::INT8;
    Tensor v(n, m, layers);
    Tensor q(n, m, layers, qt);
    Tensor w(m, k, 1);
    Tensor o1(n, k, 1);
    Tensor o2(n, k, 1);

    v.random(-120, 120);
    printf("v.sum:%8g\t", v[0].sum());
    v[0].print("[v row:0] ");

    w.random(0, 0.1);
    q.quantize(v);

    printf("q.sum:%8g\t", q[0].sum());
    q[0].print("[q row:0] ");

    Tensor f(m, k);
    f.random(-20, 50);
    for (int i = 0; i < f.rows(); ++i) {
        f[i].softmax(f.columns() - f.rows() + i + 1);
    }
    f[ 0].print("f row: 0 => ");
    f[-1].print("f row:-1 => ");

    o1.zero();
    o2.zero();
    v.weighted_sum(f, o1);
    auto pv = v.float_data();
    for (int i = 0; i < v.rows(); ++i, pv += v.columns()) {
        auto pf = f.float_data();
        auto po = o2.float_data();
        for (int k = 0; k < o2.rows(); ++k, po += o2.columns(), pf += f.columns()) {
            auto f = pf[i];
            if (f > 1e-16) {
                for (int j = 0; j < v.columns(); ++j) {
                    po[j] += pv[j] * f;
                }
            }
        }
    }
    o1[ 0].print("o1 row: 0 => ");
    o2[ 0].print("o2 row: 0 => ");
    o1[-1].print("o1 row:-1 => ");
    o2[-1].print("o2 row:-1 => ");

    o2.zero();
    q.weighted_sum(f, o2);
    o2[ 0].print("o2 row: 0 => ");
    o2[-1].print("o2 row:-1 => ");

}

void _test5(size_t m, size_t n, size_t w, int l);
void test5() {
    size_t n = 768;
    _test5(2048, n,  1, 128);
    _test5(2048, n,  4, 128);
    _test5(2048, n,  8, 128);
    return;
    _test5(2048, n, 16, 128);
    _test5(2048, n, 32, 128);
    _test5(2048, n, 64, 128);
}

int dot_product_i8(const int8_t* x1, const int8_t* x2, size_t n) noexcept {
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += int(x1[i]) * int(x2[i]);
    return sum;
}

void _test5(size_t m, size_t n, size_t w, int l) {
    w = 2;
    constexpr QuantType qt = QuantType::INT16;
    std::cout << "===============================================================================" << std::endl;
    Tensor a1(n, m, l);
    Tensor q1(n, m, l, qt);
    a1.random(-120, 120);
    q1.quantize(a1);
    a1.print("(0) a1 => ");
    q1.print("(0) q1 => ");
    std::cout << std::endl;

    Tensor a2(n, w, 1);
    Tensor q2(n, w, 1, qt);
    a2.random(-120, 120);
    q2.quantize(a2);
    //a2.print("(0) a2 => ");
    //q2.print("(0) q2 => ");
    std::cout << std::endl;
    Tensor a3(m, w, 1);

    /*a1.matmul(a2, a3);
    a3.print("(1)");
    q1.matmul(q2, a3);
    a3.print("(2)");
    return;*/

    constexpr int TN = 8;

    auto tester = [](const char* title, const Tensor& t1, const Tensor& t2, Tensor& t3) -> bool {
        float sum = 0.;
        for (int i = 0; i < t1.layers(); ++i) {
            t1.matmul(t2, t3, i);
            sum += t3.sum();
        }
        Timer tmr;
        tmr.reset();
        for (int t = 0; t < TN; ++t) {
            for (int i = 0; i < t1.layers(); ++i) {
                t1.matmul(t2, t3, i);
                sum += t3.sum();
            }
        }
        auto cost = tmr.elapsed_us() / (TN*1000.);
        printf("%s cost_ms:%6.2fms\n", title, cost);
        return sum > 1.0;
    };

    bool ret = false;
    ret |= tester("(1) ", a1, a2, a3);
    ret |= tester("(2) ", q1, q2, a3);
    std::cout << (ret ? "\n" : "\n\n") << std::endl;
}
void _test5_0(size_t m, size_t n, size_t w, int l) {
    using VT = int;
    auto fill_random = [](VT* x, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            x[i] = VT(rand() % 1000);
        }
    };

    auto x1_size = size_t(m) * n * l;
    auto x2_size = size_t(w) * n;
    auto x3_size = size_t(w) * m;
    auto x1 = (VT*)numa_alloc(sizeof(VT) * x1_size);
    auto x2 = (VT*)numa_alloc(sizeof(VT) * x2_size);
    auto x3 = (VT*)numa_alloc(sizeof(VT) * x3_size);
    fill_random(x1, x1_size);
    fill_random(x2, x2_size);

    auto tester = [&](std::function<void(const VT*, const VT*, VT*, int, int, int)> f, int testing_times=8) {
        Timer tmr;
        for (int i = 0; i < testing_times; ++i) {
            auto mat = x1;
            for (int j = 0; j < l; ++j) {
                f(mat, x2, x3, m, n, w);
                mat += m * n;
            }
        }
        return tmr.elapsed_us() / testing_times;
    };

    printf("\n");
    printf("m:%4lu\t\tn:%4lu\t\tw:%4lu\t\tl:%3d\n", m, n, w, l);
    tester(matmul1<VT>, 1);
    printf("\tmatmul1:\x1b[32m%6.1f\x1b[0mms\n", tester(matmul1<VT>)/1000.);
    printer("matmul1:", x3, size_t(m)*w);
    numa_free(x1, sizeof(VT) * x1_size);
    numa_free(x2, sizeof(VT) * x2_size);
    numa_free(x3, sizeof(VT) * x3_size);
}

void test4() {
    Tensor a1(768, 256, 1);
    Tensor q1(768, 256, 1, QuantType::INT4);
    Tensor a2(768,  32, 1);
    Tensor q2(768,  32, 1, QuantType::INT4);

    a1.random(-120, 120);
    a2.random(-120, 120);
    q1.quantize(a1);
    q2.quantize(a2);

    a1.print("a1 => ", false);
    q1.print("q1 => ", false);
    a2.print("a2 => ", false);
    q2.print("q2 => ", false);
}

void test3() {
    Tensor a1(768, 256, 1);
    Tensor q1(768, 256, 1, QuantType::INT8);
    Tensor a2(768,  32, 1);
    Tensor q2(768,  32, 1, QuantType::INT8);

    a1.random(-10, 10);
    a2.random(-10, 10);
    q1.quantize(a1);
    q2.quantize(a2);

    a1.print("a1 => ", false, false);
    q1.print("q1 => ", false, false);
    a2.print("a2 => ", false, false);
    q2.print("q2 => ", false, false);
}

