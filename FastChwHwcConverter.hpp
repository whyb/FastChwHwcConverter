/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2024] [張小凡](https://github.com/whyb)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Check if the compiler supports C++17
#if __cplusplus >= 201703L
// If C++17 is supported, use std::clamp from the standard library
#define STD_CLAMP(value, low, high) (std::clamp)(value, low, high)
#else
// If C++17 is not supported but C++11 is, implement std::clamp using std::min and std::max
#define STD_CLAMP(value, low, high) ((std::max)(low, (std::min)(value, high)))
#endif

#if defined(__has_include)
#  if __has_include(<execution>)
#    include <execution>
#    include <numeric> // for std::iota
#    define HAS_STD_EXECUTION 1
#  else
#    define HAS_STD_EXECUTION 0
#  endif
#else
#  define HAS_STD_EXECUTION 0
#endif

namespace whyb {
    class cpu {
    private:
        cpu() {
            static bool init0([]() {
                return true;
                }());
        }
    public:
        ~cpu() = default;
        cpu(const cpu&) = delete;
        cpu& operator=(const cpu&) = delete;
        cpu(cpu&&) = delete;
        cpu& operator=(cpu&&) = delete;

    public:
        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        * @param clamp Whether to clamp the data values
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        * @param normalized_mean_stds Whether to use mean and standard deviation for normalization
        * @param mean Array of mean values for normalization
        * @param stds Array of standard deviation values for normalization
        */
        template <typename Stype, typename Dtype>
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const bool clamp = false, const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const bool normalized_mean_stds = false,
            const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
            const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }) {
            std::function<Dtype(const Stype&, const size_t&)> cvt_fun;
            if (clamp) {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>((src_val - mean[c]) / stds[c], min_v, max_v)); };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val, min_v, max_v)); };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>((src_val * alpha - mean[c]) / stds[c], min_v, max_v)); };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val * alpha, min_v, max_v)); };
                    }
                }
            }
            else {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>((src_val - mean[c]) / stds[c]); };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val); };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& c) {return static_cast<Dtype>((src_val * alpha - mean[c]) / stds[c]); };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val * alpha); };
                    }
                }
            }

#ifdef _OPENMP
            const size_t hw_stride = w * h;
            const size_t num_threads = omp_get_max_threads();
            const size_t chunk_size = hw_stride / num_threads;
#pragma omp parallel
            {
                const size_t thread_id = omp_get_thread_num();
                const size_t start_idx = thread_id * chunk_size;
                const size_t end_idx = (thread_id == num_threads - 1) ? hw_stride : (start_idx + chunk_size);
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[stride_index] = cvt_fun(src[index++], c1);
                    }
                }
            }
#else
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[stride_index] = cvt_fun(src[index++], c1);
                }
            }
#endif
        }


        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        * @param clamp Whether to clamp the data values
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        */
        template <typename Stype, typename Dtype>
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const bool clamp = false, const Dtype min_v = 0, const Dtype max_v = 255) {
            std::function<Dtype(const Stype&, const size_t&)> cvt_fun;
            if (clamp) {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val * alpha, min_v, max_v)); };
                }
                else {
                    cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(std_clamp<Dtype>(src_val, min_v, max_v)); };
                }
            }
            else {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val); };
                }
                else {
                    cvt_fun = [&alpha](const Stype& src_val, const size_t& c) {return static_cast<Dtype>(src_val * alpha); };
                }
            }

#ifdef _OPENMP
            const size_t hw_stride = w * h;
            const size_t num_threads = omp_get_max_threads();
            const size_t chunk_size = hw_stride / num_threads;
#pragma omp parallel
            {
                const size_t thread_id = omp_get_thread_num();
                const size_t start_idx = thread_id * chunk_size;
                const size_t end_idx = (thread_id == num_threads - 1) ? hw_stride : (start_idx + chunk_size);
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[index++] = cvt_fun(src[stride_index], c1);
                    }
                }
            }
#else
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[index++] = cvt_fun(src[stride_index], c1);
                }
            }
#endif
        }

    private:
        template <typename T>
        inline static T std_clamp(const T& value, const T& low, const T& high) {
            return STD_CLAMP(value, low, high);
        }

        /**
        * @brief Determines if two numbers are approximately equal
        *
        * @tparam Type Type of the numbers
        * @param a First number
        * @param b Second number
        * @return true if the numbers are approximately equal
        * @return false if the numbers are not approximately equal
        */
        template <typename Type>
        inline static bool is_number_equal(const Type& a, const Type& b) {
            static Type epsilon = std::numeric_limits<Type>::epsilon();
            return std::abs(a - b) < epsilon;
        }

#if HAS_STD_EXECUTION
        template <typename Stype, typename Dtype>
        static void hwc2chw_execution_impl(
            const size_t h, const size_t w, const size_t c,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const bool clamp = false, const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const bool normalized_mean_stds = false,
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f },
            const std::array<float, 3> stds = { 0.229f, 0.224f, 0.225f }) {

            std::function<Dtype(const Stype&, const size_t&)> cvt_fun;
            if (clamp) {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& ch) {
                            return static_cast<Dtype>(std_clamp<Dtype>((src_val - mean[ch]) / stds[ch], min_v, max_v));
                        };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& /*ch*/) {
                            return static_cast<Dtype>(std_clamp<Dtype>(src_val, min_v, max_v));
                        };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &min_v, &max_v, &mean, &stds](const Stype& src_val, const size_t& ch) {
                            return static_cast<Dtype>(std_clamp<Dtype>((src_val * alpha - mean[ch]) / stds[ch], min_v, max_v));
                        };
                    }
                    else {
                        cvt_fun = [&alpha, &min_v, &max_v](const Stype& src_val, const size_t& /*ch*/) {
                            return static_cast<Dtype>(std_clamp<Dtype>(src_val * alpha, min_v, max_v));
                        };
                    }
                }
            }
            else {
                if (is_number_equal<Dtype>(alpha, 1)) {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& ch) {
                            return static_cast<Dtype>((src_val - mean[ch]) / stds[ch]);
                        };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& /*ch*/) {
                            return static_cast<Dtype>(src_val);
                        };
                    }
                }
                else {
                    if (normalized_mean_stds) {
                        cvt_fun = [&alpha, &mean, &stds](const Stype& src_val, const size_t& ch) {
                            return static_cast<Dtype>((src_val * alpha - mean[ch]) / stds[ch]);
                        };
                    }
                    else {
                        cvt_fun = [&alpha](const Stype& src_val, const size_t& /*ch*/) {
                            return static_cast<Dtype>(src_val * alpha);
                        };
                    }
                }
            }

            // ch = i % c, s = i / c, dst_index = s + ch * (w * h)
            const size_t hw_stride = w * h;
            const size_t total_elements = hw_stride * c;

            std::vector<size_t> indices(total_elements);
            std::iota(indices.begin(), indices.end(), 0);
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [=, &cvt_fun](size_t idx) {
                    size_t ch = idx % c;
                    size_t s = idx / c;
                    size_t dst_index = s + ch * hw_stride;
                    dst[dst_index] = cvt_fun(src[idx], ch);
                }
            );
        }
#endif
    };

}
