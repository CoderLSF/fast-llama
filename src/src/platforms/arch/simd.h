// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Shaofeng Liu (liushaofeng01@baidu.com)
//
// 

#pragma once

#include <cstdint>
#include <cstddef>

namespace cpuft::simd {

size_t get_simd_size() noexcept;
size_t set_simd_size(size_t simd_size) noexcept;

float dot_product(const float*  x1, const float*  x2, size_t n) noexcept;
int   dot_product(const short*  x1, const short*  x2, size_t n) noexcept;
int   dot_product(const int8_t* x1, const int8_t* x2, size_t n) noexcept;

float square_sum(const float*  x, size_t n) noexcept;
int   square_sum(const short*  x, size_t n) noexcept;
int   square_sum(const int8_t* x, size_t n) noexcept;

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

void multiply(float* x, float v, size_t n) noexcept;

void add(float* x,  float v, size_t n) noexcept;
void add(float* x1, const float* x2, size_t n) noexcept;
void add(float* x1, const float* x2, float r, size_t n) noexcept;

void rmsnorm(float* o, const float* x, const float* weight, size_t size) noexcept;
void rmsnorm(float* x, const float* weight, size_t size) noexcept;
void swiglu(float* xo, const float* xr, size_t n) noexcept;
void swiglu(float* x, size_t n) noexcept;
void sigmoid(float* x, size_t n) noexcept;

} // namespace cpuft::simd
