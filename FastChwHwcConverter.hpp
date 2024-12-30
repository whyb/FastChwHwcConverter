#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace whyb {

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
inline bool is_number_equal(Type& a, Type& b) {
    static Type epsilon = std::numeric_limits<Type>::epsilon();
    return std::abs(a - b) < epsilon;
}


/**
 * @brief Converts image data from HWC format to CHW format
 *
 * @tparam Stype Source data type
 * @tparam Dtype Destination data type
 * @param ch Number of channels
 * @param w Image width
 * @param h Image height
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
inline void hwc2chw(
    const size_t ch, const size_t w, const size_t h,
    const Stype* src, Dtype* dst,
    const Dtype alpha = 1, const bool clamp = false,
    const Dtype min_v = 0.0, const Dtype max_v = 1.0,
    const bool normalized_mean_stds = false,
    const std::array<float, 3> mean = { 0.485, 0.456, 0.406 },
    const std::array<float, 3> stds = { 0.229, 0.224, 0.225 }) {
    std::function<Dtype(Stype)> cvt_fun;
    if(clamp) {
        if(is_number_equal<Dtype>(alpha, 1)) {
            if(normalized_mean_stds) {
                cvt_fun = [&alpha,&min_v,&max_v,&mean,&stds](Stype& src_val, size_t& c){return static_cast<Dtype>(std::clamp<Dtype>((src_val - mean[c]) / stds[c], min_v, max_v));};
            } else {
                cvt_fun = [&alpha,&min_v,&max_v](Stype& src_val){return static_cast<Dtype>(std::clamp<Dtype>(src_val, min_v, max_v));};
            }
        } else {
            if(normalized_mean_stds) {
                cvt_fun = [&alpha,&min_v,&max_v,&mean,&stds](Stype& src_val, size_t& c){return static_cast<Dtype>(std::clamp<Dtype>((src_val * alpha - mean[c]) / stds[c], min_v, max_v));};
            } else {
                cvt_fun = [&alpha,&min_v,&max_v](Stype& src_val){return static_cast<Dtype>(std::clamp<Dtype>(src_val * alpha, min_v, max_v));};
            }
        }
    } else {
        if(is_number_equal<Dtype>(alpha, 1)) {
            if(normalized_mean_stds) {
                cvt_fun = [&alpha,&mean,&stds](Stype& src_val, size_t& c){return static_cast<Dtype>((src_val - mean[c]) / stds[c]);};
            } else {
                cvt_fun = [&alpha](Stype& src_val){return static_cast<Dtype>(src_val);};
            }
        } else {
            if(normalized_mean_stds) {
                cvt_fun = [&alpha,&mean,&stds](Stype& src_val, size_t& c){return static_cast<Dtype>((src_val * alpha - mean[c]) / stds[c]);};
            } else {
                cvt_fun = [&alpha](Stype& src_val){return static_cast<Dtype>(src_val * alpha);};
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
        size_t index = start_idx * ch;
        for (size_t s = start_idx; s < end_idx; ++s) {
            size_t stride_index = s;
            for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
                dst[stride_index] = cvt_fun(src[index++]);
            }
        }
    }
#else
    size_t index = 0UL;
    const size_t hw_stride = w * h;
    for (size_t s = 0UL; s < hw_stride; ++s) {
        size_t stride_index = s;
        for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
            dst[stride_index] = cvt_fun(src[index++]);
        }
    }
#endif
}


/**
 * @brief Converts image data from CHW format to HWC format
 *
 * @tparam Stype Source data type
 * @tparam Dtype Destination data type
 * @param ch Number of channels
 * @param w Image width
 * @param h Image height
 * @param src Pointer to the source data in CHW format
 * @param dst Pointer to the destination data in HWC format
 * @param alpha Scaling factor
 * @param clamp Whether to clamp the data values
 * @param min_v Minimum value for clamping
 * @param max_v Maximum value for clamping
 */
template <typename Stype, typename Dtype>
inline void chw2hwc(
    const size_t ch, const size_t w, const size_t h, 
    const Stype* src, Dtype* dst, 
    const double alpha = 1, const bool clamp = false,
    const Dtype min_v = 0, const Dtype max_v = 255) {
    std::function<Dtype(Stype)> cvt_fun;
    if(clamp) {
        if(is_number_equal<Dtype>(alpha, 1)) {
            cvt_fun = [&alpha,&min_v,&max_v](Stype& src_val){return static_cast<Dtype>(std::clamp<Dtype>(src_val * alpha, min_v, max_v));};
        } else {
            cvt_fun = [&alpha,&min_v,&max_v](Stype& src_val){return static_cast<Dtype>(std::clamp<Dtype>(src_val, min_v, max_v));};
        }
    } else {
        if(is_number_equal<Dtype>(alpha, 1)) {
            cvt_fun = [&alpha](Stype& src_val){return static_cast<Dtype>(src_val);};
        } else {
            cvt_fun = [&alpha](Stype& src_val){return static_cast<Dtype>(src_val * alpha);};
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
        size_t index = start_idx * ch;
        for (size_t s = start_idx; s < end_idx; ++s) {
            size_t stride_index = s;
            for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
                dst[index++] = cvt_fun(src[stride_index]);
            }
        }
    }
#else
    size_t index = 0UL;
    const size_t hw_stride = w * h;
    for (size_t s = 0UL; s < hw_stride; ++s) {
        size_t stride_index = s;
        for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
            dst[index++] = cvt_fun(src[stride_index]);
        }
    }
#endif
}

}
