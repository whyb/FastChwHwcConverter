/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2024-2026] [張小凡](https://github.com/whyb)
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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
#include <limits>
#include <type_traits>
#include <vector>
#include <thread>

#ifdef USE_OPENMP
#include <omp.h>
#endif


#ifndef __APPLE__
#ifdef USE_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif
#endif // __APPLE__

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

    private:
        // A small process-lifetime pool.  Reusing workers keeps the default
        // C++ backend from paying thread-creation costs on every call.
        class thread_pool {
        public:
            static thread_pool& instance() {
                static thread_pool pool;
                return pool;
            }

            size_t worker_count() const {
                return workers_.size();
            }

            // Splits [0, element_count) into balanced ranges and waits until
            // every submitted range has completed.  The range callback is
            // called instead of a per-pixel callback, keeping the hot loop
            // inlineable.
            template <typename Func>
            void run_parallel(size_t element_count, Func&& func) {
                if (element_count == 0UL) {
                    return;
                }

                const size_t task_count = (std::min)(element_count, workers_.size());
                {
                    std::unique_lock<std::mutex> lock(task_mutex_);
                    if (stop_) {
                        throw std::runtime_error("The CPU thread pool has been stopped.");
                    }

                    pending_tasks_ += task_count;
                    for (size_t task_id = 0; task_id < task_count; ++task_id) {
                        const size_t start_idx = element_count * task_id / task_count;
                        const size_t end_idx = element_count * (task_id + 1) / task_count;
                        if (start_idx < end_idx) {
                            tasks_.emplace([func, start_idx, end_idx]() {
                                func(start_idx, end_idx);
                            });
                        } else {
                            --pending_tasks_;
                        }
                    }
                }
                task_cv_.notify_all();

                std::unique_lock<std::mutex> lock(task_mutex_);
                task_done_cv_.wait(lock, [this]() {
                    return pending_tasks_ == 0UL;
                });
            }

        private:
            thread_pool() {
                const unsigned int hardware_threads = std::thread::hardware_concurrency();
                const size_t worker_count = (std::max)(size_t(1), static_cast<size_t>(hardware_threads));
                workers_.reserve(worker_count);

                for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                    workers_.emplace_back([this]() {
                        worker_loop();
                    });
                }
            }

            ~thread_pool() {
                {
                    std::lock_guard<std::mutex> lock(task_mutex_);
                    stop_ = true;
                }
                task_cv_.notify_all();

                for (auto& worker : workers_) {
                    worker.join();
                }
            }

            thread_pool(const thread_pool&) = delete;
            thread_pool& operator=(const thread_pool&) = delete;
            thread_pool(thread_pool&&) = delete;
            thread_pool& operator=(thread_pool&&) = delete;

            void worker_loop() {
                while (true) {
                    std::unique_lock<std::mutex> lock(task_mutex_);
                    task_cv_.wait(lock, [this]() {
                        return stop_ || !tasks_.empty();
                    });

                    if (tasks_.empty()) {
                        return;
                    }

                    std::function<void()> task = std::move(tasks_.front());
                    tasks_.pop();
                    lock.unlock();

                    task();

                    lock.lock();
                    --pending_tasks_;
                    task_done_cv_.notify_all();
                }
            }

            std::vector<std::thread> workers_;
            std::queue<std::function<void()>> tasks_;
            std::mutex task_mutex_;
            std::condition_variable task_cv_;
            std::condition_variable task_done_cv_;
            size_t pending_tasks_ = 0UL;
            bool stop_ = false;
        };
    public:
        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @tparam NeedNormalizedMeanStds input mean and stds are requires participation in calculations
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        * @param mean Array of mean values for normalization
        * @param stds Array of standard deviation values for normalization
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false,
                  bool NeedNormalizedMeanStds = false>
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0.0, const Dtype max_v = 1.0,
            const std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f },
            const std::array<float, 3> stds = { 0.229f, 0.224f, 0.225f }) {
            // A direct lambda lets the compiler inline the selected conversion
            // path and removes the per-pixel type-erased call overhead.
            const auto convert_pixel = [&](const Stype& src_val, const size_t channel) -> Dtype {
                if constexpr (NeedClamp) {
                    if constexpr (HasAlpha) {
                        if constexpr (NeedNormalizedMeanStds) {
                            return clamp_cast<Dtype>((src_val * alpha - mean[channel]) / stds[channel], min_v, max_v);
                        }
                        else {
                            return clamp_cast<Dtype>(src_val * alpha, min_v, max_v);
                        }
                    }
                    else {
                        if constexpr (NeedNormalizedMeanStds) {
                            return clamp_cast<Dtype>((src_val - mean[channel]) / stds[channel], min_v, max_v);
                        }
                        else {
                            return clamp_cast<Dtype>(src_val, min_v, max_v);
                        }
                    }
                }
                else {
                    if constexpr (HasAlpha) {
                        if constexpr (NeedNormalizedMeanStds) {
                            return static_cast<Dtype>((src_val * alpha - mean[channel]) / stds[channel]);
                        }
                        else {
                            return static_cast<Dtype>(src_val * alpha);
                        }
                    }
                    else {
                        if constexpr (NeedNormalizedMeanStds) {
                            return static_cast<Dtype>((src_val - mean[channel]) / stds[channel]);
                        }
                        else {
                            return static_cast<Dtype>(src_val);
                        }
                    }
                }
            };

#ifdef SINGLE_THREAD
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[stride_index] = convert_pixel(src[index++], c1);
                }
            }
#else
#ifdef USE_OPENMP
            const size_t hw_stride = w * h;
            if (hw_stride < get_parallel_threshold() || get_num_threads() == 1UL) {
                run_serial_range(hw_stride, c, src, dst, convert_pixel);
            }
            else {
#pragma omp parallel
                {
                    const int thread_count = omp_get_num_threads();
                    const int thread_id = omp_get_thread_num();
                    const size_t start_idx = hw_stride * static_cast<size_t>(thread_id) / static_cast<size_t>(thread_count);
                    const size_t end_idx = hw_stride * static_cast<size_t>(thread_id + 1) / static_cast<size_t>(thread_count);
                    size_t index = start_idx * c;
                    for (size_t s = start_idx; s < end_idx; ++s) {
                        size_t stride_index = s;
                        for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                            dst[stride_index] = convert_pixel(src[index++], c1);
                        }
                    }
                }
            }
#elif defined(USE_TBB)
            const size_t hw_stride = h * w;
            if (hw_stride < get_parallel_threshold() || get_num_threads() == 1UL) {
                run_serial_range(hw_stride, c, src, dst, convert_pixel);
            }
            else {
                const size_t grain_size = (std::max)(size_t(1), hw_stride / get_num_threads());
                tbb::parallel_for(tbb::blocked_range<size_t>(0, hw_stride, grain_size),
                    [&](const tbb::blocked_range<size_t>& range) {
                    size_t index = range.begin() * c;
                    for (size_t s = range.begin(); s < range.end(); ++s) {
                        size_t stride_index = s;
                        for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                            dst[stride_index] = convert_pixel(src[index++], c1);
                        }
                    }
                });
            }
#else
            const size_t hw_stride = h * w;
            parallel_chunks(hw_stride, [&](const size_t start_idx, const size_t end_idx) {
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[stride_index] = convert_pixel(src[index++], c1);
                    }
                }
            });
#endif
#endif
        }


        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @tparam Stype Source data type
        * @tparam Dtype Destination data type
        * @tparam HasAlpha input alpha is requires participation in calculations
        * @tparam NeedClamp input min_v and max_v are requires participation in calculations
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        * @param min_v Minimum value for clamping
        * @param max_v Maximum value for clamping
        */
        template <typename Stype, typename Dtype,
                  bool HasAlpha = false,
                  bool NeedClamp = false>
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const Stype* src, Dtype* dst,
            const Dtype alpha = 1,
            const Dtype min_v = 0, const Dtype max_v = 255) {
            // Keeping the conversion lambda local preserves exact API behavior
            // while giving the optimizer a direct, inlineable call target.
            const auto convert_pixel = [&](const Stype& src_val, const size_t channel) -> Dtype {
                if constexpr (NeedClamp) {
                    if constexpr (HasAlpha) {
                        return clamp_cast<Dtype>(src_val * alpha, min_v, max_v);
                    }
                    else {
                        return clamp_cast<Dtype>(src_val, min_v, max_v);
                    }
                }
                else {
                    if constexpr (HasAlpha) {
                        return static_cast<Dtype>(src_val * alpha);
                    }
                    else {
                        return static_cast<Dtype>(src_val);
                    }
                }
            };

#ifdef SINGLE_THREAD
            size_t index = 0UL;
            const size_t hw_stride = w * h;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[index++] = convert_pixel(src[stride_index], c1);
                }
            }
#else
#ifdef USE_OPENMP
            const size_t hw_stride = w * h;
            if (hw_stride < get_parallel_threshold() || get_num_threads() == 1UL) {
                run_serial_range_chw2hwc(hw_stride, c, src, dst, convert_pixel);
            }
            else {
#pragma omp parallel
                {
                    const int thread_count = omp_get_num_threads();
                    const int thread_id = omp_get_thread_num();
                    const size_t start_idx = hw_stride * static_cast<size_t>(thread_id) / static_cast<size_t>(thread_count);
                    const size_t end_idx = hw_stride * static_cast<size_t>(thread_id + 1) / static_cast<size_t>(thread_count);
                    size_t index = start_idx * c;
                    for (size_t s = start_idx; s < end_idx; ++s) {
                        size_t stride_index = s;
                        for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                            dst[index++] = convert_pixel(src[stride_index], c1);
                        }
                    }
                }
            }
#elif defined(USE_TBB)
            const size_t hw_stride = h * w;
            if (hw_stride < get_parallel_threshold() || get_num_threads() == 1UL) {
                run_serial_range_chw2hwc(hw_stride, c, src, dst, convert_pixel);
            }
            else {
                const size_t grain_size = (std::max)(size_t(1), hw_stride / get_num_threads());
                tbb::parallel_for(tbb::blocked_range<size_t>(0, hw_stride, grain_size),
                    [&](const tbb::blocked_range<size_t>& range) {
                    size_t index = range.begin() * c;
                    for (size_t s = range.begin(); s < range.end(); ++s) {
                        size_t stride_index = s;
                        for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                            dst[index++] = convert_pixel(src[stride_index], c1);
                        }
                    }
                });
            }
#else
            const size_t hw_stride = h * w;
            parallel_chunks(hw_stride, [&](const size_t start_idx, const size_t end_idx) {
                size_t index = start_idx * c;
                for (size_t s = start_idx; s < end_idx; ++s) {
                    size_t stride_index = s;
                    for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                        dst[index++] = convert_pixel(src[stride_index], c1);
                    }
                }
            });
#endif
#endif
        }

    private:
        // Images below this pixel count stay single-threaded; worker
        // synchronization would otherwise dominate the conversion time.
        static constexpr size_t get_parallel_threshold() {
            return 32768UL;
        }

        // Dispatches balanced pixel ranges to the persistent pool.  Every
        // pixel is covered exactly once, including remainder rows.
        template <typename Func>
        static void parallel_chunks(const size_t element_count, Func&& func) {
            if (element_count == 0UL) {
                return;
            }

            if (element_count < get_parallel_threshold()) {
                func(0UL, element_count);
                return;
            }

            auto& pool = thread_pool::instance();
            if (pool.worker_count() == 1UL) {
                func(0UL, element_count);
            }
            else {
                pool.run_parallel(element_count, std::forward<Func>(func));
            }
        }

        template <typename Stype, typename Dtype, typename Convert>
        static void run_serial_range(
            const size_t hw_stride, const size_t c,
            const Stype* src, Dtype* dst, Convert& convert_pixel) {
            size_t index = 0UL;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[stride_index] = convert_pixel(src[index++], c1);
                }
            }
        }

        template <typename Stype, typename Dtype, typename Convert>
        static void run_serial_range_chw2hwc(
            const size_t hw_stride, const size_t c,
            const Stype* src, Dtype* dst, Convert& convert_pixel) {
            size_t index = 0UL;
            for (size_t s = 0UL; s < hw_stride; ++s) {
                size_t stride_index = s;
                for (size_t c1 = 0UL; c1 < c; ++c1, stride_index += hw_stride) {
                    dst[index++] = convert_pixel(src[stride_index], c1);
                }
            }
        }

        /**
        * @brief Clamps a value to [low, high] and then narrows it to Dtype
        *
        * Clamping happens in the common type of the value and the destination,
        * so narrowing conversions (e.g. float -> uint8_t) never run on
        * out-of-range values, which would be undefined behavior.
        */
        template <typename Dtype, typename VType>
        inline static Dtype clamp_cast(const VType& value, const Dtype& low, const Dtype& high) {
            using CommonType = typename std::common_type<VType, Dtype>::type;
            return static_cast<Dtype>((std::clamp)(static_cast<CommonType>(value),
                                                   static_cast<CommonType>(low),
                                                   static_cast<CommonType>(high)));
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

        inline static size_t get_num_threads() {
#ifdef SINGLE_THREAD
            static size_t num_threads = 1;
#else
#ifdef USE_OPENMP
            static size_t num_threads = omp_get_max_threads();
#elif defined(USE_TBB)
            static size_t num_threads = tbb::this_task_arena::max_concurrency();
#else
            static size_t num_threads = std::thread::hardware_concurrency();
#endif
#endif
            return num_threads;
        }
    };

}
