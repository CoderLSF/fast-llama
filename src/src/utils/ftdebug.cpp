
#include "ftdebug.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

namespace cpuft {

void print_vector_feat(const char* tip, const float* x, int n) {
    typedef struct {
        int   i;
        float v;
    } TokenProb;

    TokenProb max2[2];
    TokenProb min2[2];
    memset(max2, 0, sizeof(max2));
    memset(min2, 0, sizeof(max2));
    for (int i = 0; i < 2 && i < n; ++i) {
        max2[i].i = i;
        max2[i].v = x[i];
        min2[i].i = i;
        min2[i].v = x[i];
    }
    for (int i = 2; i < n; ++i) {
        int j = max2[0].v < max2[1].v ? 0 : 1;
        if (x[i] > max2[j].v) {
            max2[j].i = i;
            max2[j].v = x[i];
        }
        j = min2[0].v > min2[1].v ? 0 : 1;
        if (x[i] < min2[j].v) {
            min2[j].i = i;
            min2[j].v = x[i];
        }
    }
    for (int p = 0; p < 2; ++p) {
        TokenProb* arr = p == 0 ? max2 : min2;
        if ((arr[0].v < arr[1].v) || (arr[0].v == arr[1].v && arr[0].i < arr[1].i)) {
            TokenProb v = arr[0];
            arr[0] = arr[1];
            arr[1] = v;
        }
    }
    double feat = 0.;
    for (int i = 0; i < n; ++i) {
        int sign = x[i] > 0 ? 1 : -1;
        feat += sign * sqrt(fabs(x[i]) * sqrt(i+1)/n);
    }
    printf("%s:feat:[\x1b[32m%9.4f\x1b[0m], len:%5d, data:[%9.5f, %9.5f, %9.5f, ..., %9.5f, %9.5f]\n",
            tip, feat, n, x[0], x[1], x[2], x[n-2], x[n-1]);
}

void print_vector_topk(const char* tip, const float* x, int n) {
    const int K = 4;
    typedef struct {
        int   i;
        float v;
    } TokenProb;

    TokenProb topk[K];
    topk[0].i = 0;
    topk[0].v = x[0];
    for (int i = 1; i < n; ++i) {
        if (i < K) {
            topk[i].i = i;
            topk[i].v = x[i];
            for (int j = i; j > 0; ) {
                int p = (j - 1) / 2;
                if (topk[j].v >= topk[p].v) {
                    break;
                }
                TokenProb v = topk[j];
                topk[j] = topk[p];
                topk[p] = v;
                j = p;
            }
        } else if (x[i] > topk[0].v) {
            topk[0].i = i;
            topk[0].v = x[i];
            for (int j = 0; ;) {
                int c = j * 2 + 1;
                if (c >= K) {
                    break;
                }
                if (c + 1 < K && topk[c].v > topk[c+1].v) {
                    ++c;
                }
                if (topk[c].v >= topk[j].v) {
                    break;
                }
                TokenProb v = topk[j];
                topk[j] = topk[c];
                topk[c] = v;
                j = c;
            }
        }
    }
    for (int k = K-1; k > 0; --k) {
        TokenProb v = topk[0];
        topk[0] = topk[k];
        topk[k] = v;
        for (int i = 0; ;) {
            int c = i * 2 + 1;
            if (c >= k) {
                break;
            }
            if (c + 1 < k && topk[c].v > topk[c+1].v) {
                ++c;
            }
            if (topk[c].v >= topk[i].v) {
                break;
            }
            TokenProb v = topk[i];
            topk[i] = topk[c];
            topk[c] = v;
            i = c;
        }
    }
    printf("%stopk:", tip);
    for (int i = 0; i < K; ++i) {
        printf("(%5d, %5d, %g),\t", i, topk[i].i, topk[i].v);
    }
    printf("\n");
}

} // namespace cpuft
