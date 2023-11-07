/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 @Author: Liu Shaofeng
 @Date: 2023/10/16
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <map>
#include <memory>
#include <span>
#include <vector>

namespace cpuft {

void softmax(float* x, int size) noexcept;
void softmax_sisd(float* x, int size) noexcept;

void rmsnorm(float* o, const float* x, const float* weight, size_t size) noexcept;
void rmsnorm_sisd(float* o, const float* x, const float* weight, const int size) noexcept;

void swiglu(float* xo, const float* xi, size_t d) noexcept;

void add(float* x1, const float* x2, size_t n) noexcept;
void add(float* x, float v, size_t n) noexcept;
// 将向量各元素乘以系数r后累加到向量o同位置元素上
void add(float* x1, const float* x2, float r, size_t n) noexcept;

void multiply(float* x, float v, size_t n) noexcept;

/**
 * @brief 根据输入矩阵和权重向量，计算加权求和结果，并将结果存储在输出向量中。
 * @param output_vector 输出结果向量，长度为 n
 * @param matrix 输入矩阵，大小为 rows x columns
 * @param weights 权重向量，长度为 rows，元素总和为 1
 * @param rows 矩阵的行数
 * @param columns 矩阵的列数
 */
void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, float min_w=0) noexcept;
void weighted_sum(float* out, const float* matrix, const float* weights, int m, int n, int bs, float min_w=0) noexcept;

void fill_random(float* x, size_t n, float min_value, float max_value) noexcept;

/**
 * @brief 计算两个向量的点积
 *
 * @param x1 指向第一个向量的指针
 * @param x2 指向第二个向量的指针
 * @param n 向量的长度
 * @return float 计算得到的点积值
 */
float dot_product(const float*  x1, const float*  x2, size_t n) noexcept;
int   dot_product(const   int*  x1, const   int*  x2, size_t n) noexcept;
int   dot_product(const short*  x1, const short*  x2, size_t n) noexcept;
int   dot_product(const int8_t* x1, const int8_t* x2, size_t n) noexcept;
float dot_product_sisd(const float*  x1, const float*  x2, size_t n) noexcept;
int   dot_product_sisd(const   int*  x1, const   int*  x2, size_t n) noexcept;

/**
* @brief 计算向量平方和
* @param x 指向向量的指针
* @param n 向量的长度
* @return float 计算得到的平方和值
*/
float square_sum(const float*  x, size_t n) noexcept;
int   square_sum(const short*  x, size_t n) noexcept;
int   square_sum(const int8_t* x, size_t n) noexcept;

void  matmul(const float* mat1, const float* mat2, float* out, int m, int n, int k) noexcept;
void  matmul(const   int* mat1, const   int* mat2,   int* out, int m, int n, int k) noexcept;

void fill_random(float* x, int n);

float  array_sum(const float*  x, size_t n) noexcept;
int    array_sum(const   int*  x, size_t n) noexcept;
int    array_sum(const short*  x, size_t n) noexcept;
int    array_sum(const int8_t* x, size_t n) noexcept;
float  array_max(const float*  x, size_t n) noexcept;
float  array_min(const float*  x, size_t n) noexcept;
int    array_max(const int*  x, size_t n) noexcept;
int    array_min(const int*  x, size_t n) noexcept;
short  array_max(const short*  x, size_t n) noexcept;
short  array_min(const short*  x, size_t n) noexcept;
int8_t array_max(const int8_t* x, size_t n) noexcept;
int8_t array_min(const int8_t* x, size_t n) noexcept;
float  array_max_abs(const float*  arr, size_t n) noexcept;
short  array_max_abs(const short*  arr, size_t n) noexcept;
int8_t array_max_abs(const int8_t* arr, size_t n) noexcept;

struct MinMax {
    float min;
    float max;
};
MinMax array_min_max(const float* x, size_t n) noexcept;

void rope(float* x, int n, int pos, int head_size, float freq_base=10000.) noexcept;
void rope(float* x, int n, const float* freq_cis, int head_size) noexcept;
void rope(float* q, float* k, int pos, int head_size, int dim, int kv_dim, float freq_base=10000.);

void rope_v2(float * o, const float* x, int head_size, int n_orig_ctx, int pos, int ith, int nth);

} // namespace cpuft
