/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2026] [張小凡](https://github.com/whyb)
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

#include "DynamicLibraryManager.hpp"
#include "FastChwHwcConverter.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace whyb {

// ---------------------------------------------------------------------------
// glslang C interface ABI types (glslang_c_interface.h counterpart).
// These declarations must stay byte-for-byte compatible with the glslang
// shared library that is loaded at runtime. The glslang library is loaded
// dynamically (like NVRTC/HIPRTC in the CUDA/ROCm backends), so no glslang
// headers or link-time dependency are required.
// ---------------------------------------------------------------------------

// glslang_shader_types.h counterparts
typedef enum {
    GLSLANG_STAGE_VERTEX = 0,
    GLSLANG_STAGE_TESSCONTROL = 1,
    GLSLANG_STAGE_TESSEVALUATION = 2,
    GLSLANG_STAGE_GEOMETRY = 3,
    GLSLANG_STAGE_FRAGMENT = 4,
    GLSLANG_STAGE_COMPUTE = 5
} glslang_stage_t;

typedef enum {
    GLSLANG_SOURCE_NONE = 0,
    GLSLANG_SOURCE_GLSL = 1,
    GLSLANG_SOURCE_HLSL = 2
} glslang_source_t;

typedef enum {
    GLSLANG_CLIENT_NONE = 0,
    GLSLANG_CLIENT_VULKAN = 1,
    GLSLANG_CLIENT_OPENGL = 2
} glslang_client_t;

typedef enum {
    GLSLANG_TARGET_NONE = 0,
    GLSLANG_TARGET_SPV = 1
} glslang_target_language_t;

typedef enum {
    GLSLANG_TARGET_VULKAN_1_0 = (1 << 22),
    GLSLANG_TARGET_VULKAN_1_1 = (1 << 22) | (1 << 12),
    GLSLANG_TARGET_VULKAN_1_2 = (1 << 22) | (2 << 12),
    GLSLANG_TARGET_VULKAN_1_3 = (1 << 22) | (3 << 12)
} glslang_target_client_version_t;

typedef enum {
    GLSLANG_TARGET_SPV_1_0 = (1 << 16),
    GLSLANG_TARGET_SPV_1_1 = (1 << 16) | (1 << 8),
    GLSLANG_TARGET_SPV_1_2 = (1 << 16) | (2 << 8),
    GLSLANG_TARGET_SPV_1_3 = (1 << 16) | (3 << 8),
    GLSLANG_TARGET_SPV_1_4 = (1 << 16) | (4 << 8),
    GLSLANG_TARGET_SPV_1_5 = (1 << 16) | (5 << 8)
} glslang_target_language_version_t;

typedef enum {
    GLSLANG_MSG_DEFAULT_BIT = 0,
    GLSLANG_MSG_RELAXED_ERRORS_BIT = (1 << 0),
    GLSLANG_MSG_SUPPRESS_WARNINGS_BIT = (1 << 1),
    GLSLANG_MSG_AST_BIT = (1 << 2),
    GLSLANG_MSG_SPV_RULES_BIT = (1 << 3),
    GLSLANG_MSG_VULKAN_RULES_BIT = (1 << 4),
    GLSLANG_MSG_ONLY_PREPROCESSOR_BIT = (1 << 5),
    GLSLANG_MSG_READ_HLSL_BIT = (1 << 6),
    GLSLANG_MSG_CASCADING_ERRORS_BIT = (1 << 7),
    GLSLANG_MSG_KEEP_UNCALLED_BIT = (1 << 8),
    GLSLANG_MSG_HLSL_OFFSETS_BIT = (1 << 9),
    GLSLANG_MSG_DEBUG_INFO_BIT = (1 << 10),
    GLSLANG_MSG_HLSL_ENABLE_16BIT_TYPES_BIT = (1 << 11),
    GLSLANG_MSG_HLSL_LEGALIZATION_BIT = (1 << 12),
    GLSLANG_MSG_HLSL_DX9_COMPATIBLE_BIT = (1 << 13),
    GLSLANG_MSG_BUILTIN_SYMBOL_TABLE_BIT = (1 << 14),
    GLSLANG_MSG_ENHANCED = (1 << 15)
} glslang_messages_t;

typedef enum {
    GLSLANG_NO_PROFILE = (1 << 0),
    GLSLANG_CORE_PROFILE = (1 << 1),
    GLSLANG_COMPATIBILITY_PROFILE = (1 << 2),
    GLSLANG_ES_PROFILE = (1 << 3)
} glslang_profile_t;

typedef struct glslang_shader_s glslang_shader_t;
typedef struct glslang_program_s glslang_program_t;

typedef struct glsl_include_result_s {
    const char* header_name;
    const char* header_data;
    size_t header_length;
} glsl_include_result_t;

typedef glsl_include_result_t* (*glsl_include_local_func)(void* ctx, const char* header_name,
                                                          const char* includer_name, size_t include_depth);
typedef glsl_include_result_t* (*glsl_include_system_func)(void* ctx, const char* header_name,
                                                           const char* includer_name, size_t include_depth);
typedef int (*glsl_free_include_result_func)(void* ctx, glsl_include_result_t* result);

typedef struct glsl_include_callbacks_s {
    glsl_include_system_func include_system;
    glsl_include_local_func include_local;
    glsl_free_include_result_func free_include_result;
} glsl_include_callbacks_t;

typedef struct glslang_limits_s {
    bool non_inductive_for_loops;
    bool while_loops;
    bool do_while_loops;
    bool general_uniform_indexing;
    bool general_attribute_matrix_vector_indexing;
    bool general_varying_indexing;
    bool general_sampler_indexing;
    bool general_variable_indexing;
    bool general_constant_matrix_vector_indexing;
} glslang_limits_t;

// TBuiltInResource counterpart. Every field must stay in this exact order.
typedef struct glslang_resource_s {
    int max_lights;
    int max_clip_planes;
    int max_texture_units;
    int max_texture_coords;
    int max_vertex_attribs;
    int max_vertex_uniform_components;
    int max_varying_floats;
    int max_vertex_texture_image_units;
    int max_combined_texture_image_units;
    int max_texture_image_units;
    int max_fragment_uniform_components;
    int max_draw_buffers;
    int max_vertex_uniform_vectors;
    int max_varying_vectors;
    int max_fragment_uniform_vectors;
    int max_vertex_output_vectors;
    int max_fragment_input_vectors;
    int min_program_texel_offset;
    int max_program_texel_offset;
    int max_clip_distances;
    int max_compute_work_group_count_x;
    int max_compute_work_group_count_y;
    int max_compute_work_group_count_z;
    int max_compute_work_group_size_x;
    int max_compute_work_group_size_y;
    int max_compute_work_group_size_z;
    int max_compute_uniform_components;
    int max_compute_texture_image_units;
    int max_compute_image_uniforms;
    int max_compute_atomic_counters;
    int max_compute_atomic_counter_buffers;
    int max_varying_components;
    int max_vertex_output_components;
    int max_geometry_input_components;
    int max_geometry_output_components;
    int max_fragment_input_components;
    int max_image_units;
    int max_combined_image_units_and_fragment_outputs;
    int max_combined_shader_output_resources;
    int max_image_samples;
    int max_vertex_image_uniforms;
    int max_tess_control_image_uniforms;
    int max_tess_evaluation_image_uniforms;
    int max_geometry_image_uniforms;
    int max_fragment_image_uniforms;
    int max_combined_image_uniforms;
    int max_geometry_texture_image_units;
    int max_geometry_output_vertices;
    int max_geometry_total_output_components;
    int max_geometry_uniform_components;
    int max_geometry_varying_components;
    int max_tess_control_input_components;
    int max_tess_control_output_components;
    int max_tess_control_texture_image_units;
    int max_tess_control_uniform_components;
    int max_tess_control_total_output_components;
    int max_tess_evaluation_input_components;
    int max_tess_evaluation_output_components;
    int max_tess_evaluation_texture_image_units;
    int max_tess_evaluation_uniform_components;
    int max_tess_patch_components;
    int max_patch_vertices;
    int max_tess_gen_level;
    int max_viewports;
    int max_vertex_atomic_counters;
    int max_tess_control_atomic_counters;
    int max_tess_evaluation_atomic_counters;
    int max_geometry_atomic_counters;
    int max_fragment_atomic_counters;
    int max_combined_atomic_counters;
    int max_atomic_counter_bindings;
    int max_vertex_atomic_counter_buffers;
    int max_tess_control_atomic_counter_buffers;
    int max_tess_evaluation_atomic_counter_buffers;
    int max_geometry_atomic_counter_buffers;
    int max_fragment_atomic_counter_buffers;
    int max_combined_atomic_counter_buffers;
    int max_atomic_counter_buffer_size;
    int max_transform_feedback_buffers;
    int max_transform_feedback_interleaved_components;
    int max_cull_distances;
    int max_combined_clip_and_cull_distances;
    int max_samples;
    int max_mesh_output_vertices_nv;
    int max_mesh_output_primitives_nv;
    int max_mesh_work_group_size_x_nv;
    int max_mesh_work_group_size_y_nv;
    int max_mesh_work_group_size_z_nv;
    int max_task_work_group_size_x_nv;
    int max_task_work_group_size_y_nv;
    int max_task_work_group_size_z_nv;
    int max_mesh_view_count_nv;
    int max_mesh_output_vertices_ext;
    int max_mesh_output_primitives_ext;
    int max_mesh_work_group_size_x_ext;
    int max_mesh_work_group_size_y_ext;
    int max_mesh_work_group_size_z_ext;
    int max_task_work_group_size_x_ext;
    int max_task_work_group_size_y_ext;
    int max_task_work_group_size_z_ext;
    int max_mesh_view_count_ext;
    int max_dual_source_draw_buffers_ext;
    glslang_limits_t limits;
} glslang_resource_t;

typedef struct glslang_input_s {
    glslang_source_t language;
    glslang_stage_t stage;
    glslang_client_t client;
    glslang_target_client_version_t client_version;
    glslang_target_language_t target_language;
    glslang_target_language_version_t target_language_version;
    const char* code;
    int default_version;
    glslang_profile_t default_profile;
    int force_default_version_and_profile;
    int forward_compatible;
    glslang_messages_t messages;
    const glslang_resource_t* resource;
    glsl_include_callbacks_t callbacks;
    void* callbacks_ctx;
} glslang_input_t;

// ---------------------------------------------------------------------------
// Minimal Vulkan ABI types (vulkan_core.h counterparts).
// Only the subset used by this backend is declared; the numeric values are
// frozen Vulkan ABI constants so they stay compatible with any driver.
// ---------------------------------------------------------------------------

// Vulkan handles are 64-bit values on every platform (pointers on 64-bit,
// uint64_t on 32-bit), so uint64_t matches the driver ABI everywhere.
typedef uint64_t VkInstance;
typedef uint64_t VkPhysicalDevice;
typedef uint64_t VkDevice;
typedef uint64_t VkQueue;
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkCommandPool;
typedef uint64_t VkCommandBuffer;
typedef uint64_t VkDescriptorSetLayout;
typedef uint64_t VkDescriptorPool;
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkPipelineLayout;
typedef uint64_t VkPipeline;
typedef uint64_t VkPipelineCache;
typedef uint64_t VkShaderModule;
typedef uint64_t VkSemaphore;
typedef uint64_t VkFence;
typedef uint64_t VkDeviceSize;

#ifndef VK_NULL_HANDLE
#define VK_NULL_HANDLE 0ULL

#endif
#ifndef VK_WHOLE_SIZE
#define VK_WHOLE_SIZE (~0ULL)

#endif
#ifndef VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256U

#endif
#ifndef VK_UUID_SIZE
#define VK_UUID_SIZE 16U

#endif
#ifndef VK_MAX_MEMORY_TYPES
#define VK_MAX_MEMORY_TYPES 32U

#endif
#ifndef VK_MAX_MEMORY_HEAPS
#define VK_MAX_MEMORY_HEAPS 16U

#endif

#ifndef VK_MAKE_VERSION
#define VK_MAKE_VERSION(major, minor, patch) ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))
#endif
#ifndef VK_API_VERSION_1_0
#define VK_API_VERSION_1_0 VK_MAKE_VERSION(1, 0, 0)

#endif

typedef int32_t VkResult;
#ifndef VK_SUCCESS
#define VK_SUCCESS 0

#endif

typedef enum {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46
} VkStructureType;

typedef enum {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4
} VkPhysicalDeviceType;

typedef enum { VK_SHARING_MODE_EXCLUSIVE = 0 } VkSharingMode;
typedef enum { VK_PIPELINE_BIND_POINT_COMPUTE = 1 } VkPipelineBindPoint;
typedef enum { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7 } VkDescriptorType;
typedef enum { VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0 } VkCommandBufferLevel;

// Bitmask enums (values are Vulkan ABI constants)
typedef uint32_t VkFlags;
typedef enum {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
} VkBufferUsageFlagBits;
typedef enum {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
} VkMemoryPropertyFlagBits;
typedef enum {
    VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001
} VkMemoryHeapFlagBits;
typedef enum {
    VK_QUEUE_COMPUTE_BIT = 0x00000002,
    VK_QUEUE_TRANSFER_BIT = 0x00000004
} VkQueueFlagBits;
typedef enum {
    VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
} VkShaderStageFlagBits;
typedef enum {
    VK_ACCESS_SHADER_READ_BIT = 0x00000020,
    VK_ACCESS_SHADER_WRITE_BIT = 0x00000040,
    VK_ACCESS_TRANSFER_READ_BIT = 0x00000800,
    VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000,
    VK_ACCESS_HOST_READ_BIT = 0x00002000,
    VK_ACCESS_HOST_WRITE_BIT = 0x00004000,
    VK_ACCESS_MEMORY_READ_BIT = 0x00008000,
    VK_ACCESS_MEMORY_WRITE_BIT = 0x00010000
} VkAccessFlagBits;
typedef enum {
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800,
    VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000,
    VK_PIPELINE_STAGE_HOST_BIT = 0x00004000,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000
} VkPipelineStageFlagBits;
typedef enum {
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
} VkCommandBufferUsageFlagBits;
typedef enum {
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT = 0x00000001
} VkCommandPoolCreateFlagBits;

typedef struct VkApplicationInfo {
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkMemoryType {
    VkFlags propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryProperties;

// Raw view of VkPhysicalDeviceProperties. Only the leading fields are used;
// the trailing bytes give room for the large VkPhysicalDeviceLimits and
// VkPhysicalDeviceSparseProperties that vkGetPhysicalDeviceProperties writes.
typedef struct VkPhysicalDevicePropertiesRaw {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    int32_t deviceType;
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t pipelineCacheUUID[VK_UUID_SIZE];
    uint8_t reserved[4096];
} VkPhysicalDevicePropertiesRaw;

typedef struct VkExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
} VkExtent3D;

typedef struct VkQueueFamilyProperties {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkShaderStageFlagBits stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkPushConstantRange {
    VkShaderStageFlagBits stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkShaderStageFlagBits stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkDescriptorPoolSize {
    VkDescriptorType type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkBufferCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy;

typedef struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    const VkPipelineStageFlagBits* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore* pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
} VkMemoryBarrier;

// ---------------------------------------------------------------------------
// glslang C API function types
// ---------------------------------------------------------------------------
typedef int (*glslang_initialize_process_t)(void);
typedef void (*glslang_finalize_process_t)(void);
typedef glslang_shader_t* (*glslang_shader_create_t)(const glslang_input_t*);
typedef void (*glslang_shader_delete_t)(glslang_shader_t*);
typedef int (*glslang_shader_preprocess_t)(glslang_shader_t*, const glslang_input_t*);
typedef int (*glslang_shader_parse_t)(glslang_shader_t*, const glslang_input_t*);
typedef const char* (*glslang_shader_get_info_log_t)(glslang_shader_t*);
typedef glslang_program_t* (*glslang_program_create_t)(void);
typedef void (*glslang_program_delete_t)(glslang_program_t*);
typedef void (*glslang_program_add_shader_t)(glslang_program_t*, glslang_shader_t*);
typedef int (*glslang_program_link_t)(glslang_program_t*, int);
typedef const char* (*glslang_program_get_info_log_t)(glslang_program_t*);
typedef void (*glslang_program_SPIRV_generate_t)(glslang_program_t*, glslang_stage_t);
typedef size_t (*glslang_program_SPIRV_get_size_t)(glslang_program_t*);
typedef void (*glslang_program_SPIRV_get_t)(glslang_program_t*, unsigned int*);

// Vulkan API function types
typedef VkResult (*vkCreateInstance_t)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef void (*vkDestroyInstance_t)(VkInstance, const void*);
typedef VkResult (*vkEnumeratePhysicalDevices_t)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*vkGetPhysicalDeviceProperties_t)(VkPhysicalDevice, VkPhysicalDevicePropertiesRaw*);
typedef void (*vkGetPhysicalDeviceMemoryProperties_t)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef void (*vkGetPhysicalDeviceQueueFamilyProperties_t)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef VkResult (*vkCreateDevice_t)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
typedef void (*vkDestroyDevice_t)(VkDevice, const void*);
typedef void (*vkGetDeviceQueue_t)(VkDevice, uint32_t, uint32_t, VkQueue*);
typedef VkResult (*vkDeviceWaitIdle_t)(VkDevice);
typedef VkResult (*vkQueueWaitIdle_t)(VkQueue);
typedef VkResult (*vkQueueSubmit_t)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
typedef VkResult (*vkCreateShaderModule_t)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
typedef void (*vkDestroyShaderModule_t)(VkDevice, VkShaderModule, const void*);
typedef VkResult (*vkCreateDescriptorSetLayout_t)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
typedef void (*vkDestroyDescriptorSetLayout_t)(VkDevice, VkDescriptorSetLayout, const void*);
typedef VkResult (*vkCreateDescriptorPool_t)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
typedef void (*vkDestroyDescriptorPool_t)(VkDevice, VkDescriptorPool, const void*);
typedef VkResult (*vkAllocateDescriptorSets_t)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
typedef void (*vkUpdateDescriptorSets_t)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
typedef VkResult (*vkCreatePipelineLayout_t)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
typedef void (*vkDestroyPipelineLayout_t)(VkDevice, VkPipelineLayout, const void*);
typedef VkResult (*vkCreateComputePipelines_t)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
typedef void (*vkDestroyPipeline_t)(VkDevice, VkPipeline, const void*);
typedef VkResult (*vkCreateCommandPool_t)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
typedef void (*vkDestroyCommandPool_t)(VkDevice, VkCommandPool, const void*);
typedef VkResult (*vkAllocateCommandBuffers_t)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
typedef void (*vkFreeCommandBuffers_t)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
typedef VkResult (*vkBeginCommandBuffer_t)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
typedef VkResult (*vkEndCommandBuffer_t)(VkCommandBuffer);
typedef void (*vkCmdBindPipeline_t)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
typedef void (*vkCmdBindDescriptorSets_t)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
typedef void (*vkCmdPushConstants_t)(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlagBits, uint32_t, uint32_t, const void*);
typedef void (*vkCmdDispatch_t)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef void (*vkCmdPipelineBarrier_t)(VkCommandBuffer, VkPipelineStageFlagBits, VkPipelineStageFlagBits, uint32_t, uint32_t, const VkMemoryBarrier*, uint32_t, const void*, uint32_t, const void*);
typedef void (*vkCmdCopyBuffer_t)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
typedef VkResult (*vkCreateBuffer_t)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void (*vkDestroyBuffer_t)(VkDevice, VkBuffer, const void*);
typedef void (*vkGetBufferMemoryRequirements_t)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*vkAllocateMemory_t)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef void (*vkFreeMemory_t)(VkDevice, VkDeviceMemory, const void*);
typedef VkResult (*vkBindBufferMemory_t)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*vkMapMemory_t)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, uint32_t, void**);
typedef void (*vkUnmapMemory_t)(VkDevice, VkDeviceMemory);

// glslang C API function pointers. Inline variables so every translation unit
// that includes this header observes the same initialized set.
inline glslang_initialize_process_t glslang_initialize_process = nullptr;
inline glslang_finalize_process_t glslang_finalize_process = nullptr;
inline glslang_shader_create_t glslang_shader_create = nullptr;
inline glslang_shader_delete_t glslang_shader_delete = nullptr;
inline glslang_shader_preprocess_t glslang_shader_preprocess = nullptr;
inline glslang_shader_parse_t glslang_shader_parse = nullptr;
inline glslang_shader_get_info_log_t glslang_shader_get_info_log = nullptr;
inline glslang_program_create_t glslang_program_create = nullptr;
inline glslang_program_delete_t glslang_program_delete = nullptr;
inline glslang_program_add_shader_t glslang_program_add_shader = nullptr;
inline glslang_program_link_t glslang_program_link = nullptr;
inline glslang_program_get_info_log_t glslang_program_get_info_log = nullptr;
inline glslang_program_SPIRV_generate_t glslang_program_SPIRV_generate = nullptr;
inline glslang_program_SPIRV_get_size_t glslang_program_SPIRV_get_size = nullptr;
inline glslang_program_SPIRV_get_t glslang_program_SPIRV_get = nullptr;

// Vulkan API function pointers.
inline vkCreateInstance_t vkCreateInstance = nullptr;
inline vkDestroyInstance_t vkDestroyInstance = nullptr;
inline vkEnumeratePhysicalDevices_t vkEnumeratePhysicalDevices = nullptr;
inline vkGetPhysicalDeviceProperties_t vkGetPhysicalDeviceProperties = nullptr;
inline vkGetPhysicalDeviceMemoryProperties_t vkGetPhysicalDeviceMemoryProperties = nullptr;
inline vkGetPhysicalDeviceQueueFamilyProperties_t vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
inline vkCreateDevice_t vkCreateDevice = nullptr;
inline vkDestroyDevice_t vkDestroyDevice = nullptr;
inline vkGetDeviceQueue_t vkGetDeviceQueue = nullptr;
inline vkDeviceWaitIdle_t vkDeviceWaitIdle = nullptr;
inline vkQueueWaitIdle_t vkQueueWaitIdle = nullptr;
inline vkQueueSubmit_t vkQueueSubmit = nullptr;
inline vkCreateShaderModule_t vkCreateShaderModule = nullptr;
inline vkDestroyShaderModule_t vkDestroyShaderModule = nullptr;
inline vkCreateDescriptorSetLayout_t vkCreateDescriptorSetLayout = nullptr;
inline vkDestroyDescriptorSetLayout_t vkDestroyDescriptorSetLayout = nullptr;
inline vkCreateDescriptorPool_t vkCreateDescriptorPool = nullptr;
inline vkDestroyDescriptorPool_t vkDestroyDescriptorPool = nullptr;
inline vkAllocateDescriptorSets_t vkAllocateDescriptorSets = nullptr;
inline vkUpdateDescriptorSets_t vkUpdateDescriptorSets = nullptr;
inline vkCreatePipelineLayout_t vkCreatePipelineLayout = nullptr;
inline vkDestroyPipelineLayout_t vkDestroyPipelineLayout = nullptr;
inline vkCreateComputePipelines_t vkCreateComputePipelines = nullptr;
inline vkDestroyPipeline_t vkDestroyPipeline = nullptr;
inline vkCreateCommandPool_t vkCreateCommandPool = nullptr;
inline vkDestroyCommandPool_t vkDestroyCommandPool = nullptr;
inline vkAllocateCommandBuffers_t vkAllocateCommandBuffers = nullptr;
inline vkFreeCommandBuffers_t vkFreeCommandBuffers = nullptr;
inline vkBeginCommandBuffer_t vkBeginCommandBuffer = nullptr;
inline vkEndCommandBuffer_t vkEndCommandBuffer = nullptr;
inline vkCmdBindPipeline_t vkCmdBindPipeline = nullptr;
inline vkCmdBindDescriptorSets_t vkCmdBindDescriptorSets = nullptr;
inline vkCmdPushConstants_t vkCmdPushConstants = nullptr;
inline vkCmdDispatch_t vkCmdDispatch = nullptr;
inline vkCmdPipelineBarrier_t vkCmdPipelineBarrier = nullptr;
inline vkCmdCopyBuffer_t vkCmdCopyBuffer = nullptr;
inline vkCreateBuffer_t vkCreateBuffer = nullptr;
inline vkDestroyBuffer_t vkDestroyBuffer = nullptr;
inline vkGetBufferMemoryRequirements_t vkGetBufferMemoryRequirements = nullptr;
inline vkAllocateMemory_t vkAllocateMemory = nullptr;
inline vkFreeMemory_t vkFreeMemory = nullptr;
inline vkBindBufferMemory_t vkBindBufferMemory = nullptr;
inline vkMapMemory_t vkMapMemory = nullptr;
inline vkUnmapMemory_t vkUnmapMemory = nullptr;

// ---------------------------------------------------------------------------
// GLSL compute shader sources (compiled to SPIR-V at runtime with glslang).
//
// hwc2chw: HWC (uint8) -> CHW (float). The byte source is read as uint32
// words and the requested byte is extracted with bit operations, so no
// 8-bit integer storage feature is required.
//
// chw2hwc: CHW (float) -> HWC (uint8). Every output uint32 word is owned by
// exactly one thread, which packs its four bytes itself. This keeps the byte
// writes race-free without 8-bit storage support.
// ---------------------------------------------------------------------------
static const char* vulkanHwc2ChwSource = R"(#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer SrcBuffer { uint src[]; };
layout(std430, binding = 1) writeonly buffer DstBuffer { float dst[]; };

layout(push_constant) uniform PushConstants
{
    uint h;
    uint w;
    uint c;
    float alpha;
} pc;

void main()
{
    const uint pixel = gl_GlobalInvocationID.x;
    const uint pixel_count = pc.h * pc.w;
    if (pixel >= pixel_count)
    {
        return;
    }

    const uint src_base = pixel * pc.c;
    const uint dst_base = pixel;
    for (uint ch = 0u; ch < pc.c; ++ch)
    {
        const uint byte_index = src_base + ch;
        const uint word = src[byte_index >> 2u];
        const uint byte_value = (word >> ((byte_index & 3u) << 3u)) & 0xFFu;
        dst[dst_base + ch * pixel_count] = float(byte_value) * pc.alpha;
    }
}
)";

static const char* vulkanChw2HwcSource = R"(#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer SrcBuffer { float src[]; };
layout(std430, binding = 1) writeonly buffer DstBuffer { uint dst[]; };

layout(push_constant) uniform PushConstants
{
    uint c;
    uint h;
    uint w;
    float alpha;
} pc;

void main()
{
    const uint total_bytes = pc.h * pc.w * pc.c;
    const uint total_words = (total_bytes + 3u) >> 2u;
    const uint word_index = gl_GlobalInvocationID.x;
    if (word_index >= total_words)
    {
        return;
    }

    const uint byte_start = word_index << 2u;
    const uint byte_end = min(byte_start + 4u, total_bytes);
    uint packed = 0u;
    for (uint b = byte_start; b < byte_end; ++b)
    {
        const uint pixel = b / pc.c;
        const uint ch = b - pixel * pc.c;
        float value = src[pixel + ch * pc.h * pc.w] * pc.alpha;
        value = clamp(value, 0.0, 255.0);
        packed |= uint(value) << ((b & 3u) << 3u);
    }
    dst[word_index] = packed;
}
)";

enum struct InitVulkanStatusEnum : int {
        Ready = 0,
        Inited = 1,
        Failed = 2,
    };

    class vulkan {
    private:
        vulkan() {
            static bool init0([]() {
                return initAll();
                }());
        }

    public:
        ~vulkan() = default;
        vulkan(const vulkan&) = delete;
        vulkan& operator=(const vulkan&) = delete;
        vulkan(vulkan&&) = delete;
        vulkan& operator=(vulkan&&) = delete;

    public:
        static bool init() { return initAll(); }
        static bool release() { return releaseAll(); }

        // Query initialization state and the last backend error without
        // requiring library-side terminal diagnostics.
        static InitVulkanStatusEnum status() { return initVulkanStatus.load(std::memory_order_acquire); }
        static std::string lastError()
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            return lastVulkanErrorStr;
        }


        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Pointer to the source data in HWC format
        * @param dst Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        */
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const uint8_t* src, float* dst,
            const float alpha = 1.f / 255.f) {
            vulkan();
            if (h == 0 || w == 0 || c == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                // use cpu
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }
            // use vulkan
            const size_t pixel_size = h * w * c;
            const size_t src_bytes = pixel_size; // uint8_t
            const size_t src_size = (src_bytes + 3u) & ~size_t(3u);
            const size_t dst_bytes = pixel_size * sizeof(float); // multiple of 4

            VkBuffer dev_src = 0, dev_dst = 0, host_src = 0, host_dst = 0;
            VkDeviceMemory dev_src_mem = 0, dev_dst_mem = 0, host_src_mem = 0, host_dst_mem = 0;

            bool ok = createBuffer(src_size,
                (VkFlags)(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev_src, dev_src_mem);
            ok = ok && createBuffer(dst_bytes,
                (VkFlags)(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev_dst, dev_dst_mem);
            ok = ok && createBuffer(src_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                host_src, host_src_mem);
            ok = ok && createBuffer(dst_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                host_dst, host_dst_mem);
            if (!ok) {
                destroyBuffer(dev_src, dev_src_mem);
                destroyBuffer(dev_dst, dev_dst_mem);
                destroyBuffer(host_src, host_src_mem);
                destroyBuffer(host_dst, host_dst_mem);
                recordCallFailure("Vulkan hwc2chw buffer allocation failed");
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
                return;
            }

            auto cleanup_buffers = [&]() {
                destroyBuffer(dev_src, dev_src_mem);
                destroyBuffer(dev_dst, dev_dst_mem);
                destroyBuffer(host_src, host_src_mem);
                destroyBuffer(host_dst, host_dst_mem);
            };
            auto fallback_to_cpu = [&]() {
                cleanup_buffers();
                recordCallFailure("Vulkan hwc2chw GPU execution failed");
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
            };

            if (!writeHostBuffer(host_src_mem, src_bytes, src)) {
                fallback_to_cpu();
                return;
            }

            VkCommandBuffer cmd = 0;
            if (!beginCommandBuffer(cmd)) {
                fallback_to_cpu();
                return;
            }

            // copy host staging buffer to device-local buffer
            VkBufferCopy copy_region = {};
            copy_region.srcOffset = 0;
            copy_region.dstOffset = 0;
            copy_region.size = src_bytes;
            vkCmdCopyBuffer(cmd, host_src, dev_src, 1, &copy_region);

            // barrier: transfer write -> shader read
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            updateDescriptorSet(hwc2chwDescSet, dev_src, src_size, dev_dst, dst_bytes);

            PushConstantsHwc2Chw push_constants = {};
            push_constants.h = (uint32_t)h;
            push_constants.w = (uint32_t)w;
            push_constants.c = (uint32_t)c;
            push_constants.alpha = alpha;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hwc2chwPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                0, 1, &hwc2chwDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, (uint32_t)sizeof(PushConstantsHwc2Chw), &push_constants);
            const uint32_t group_count = (uint32_t)((h * w + 255u) / 256u);
            vkCmdDispatch(cmd, group_count, 1, 1);

            // barrier: shader write -> transfer read
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // copy device-local buffer back to host staging buffer
            copy_region.size = dst_bytes;
            vkCmdCopyBuffer(cmd, dev_dst, host_dst, 1, &copy_region);

            if (!submitCommandBuffer(cmd)) {
                fallback_to_cpu();
                return;
            }

            const bool read_ok = readHostBuffer(host_dst_mem, dst_bytes, dst);
            cleanup_buffers();
            if (!read_ok) {
                recordCallFailure("Vulkan hwc2chw result readback failed");
                cpu::hwc2chw<uint8_t, float, true>(h, w, c, src, dst, alpha);
            }
            clearCallFailures();
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Pointer to the source data in CHW format
        * @param dst Pointer to the destination data in HWC format
        * @param alpha Scaling factor
        */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const float* src, uint8_t* dst,
            const uint8_t alpha = 255.0f) {
            vulkan();
            if (c == 0 || h == 0 || w == 0 || src == nullptr || dst == nullptr) {
                return;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                // use cpu (clamp before narrowing, matching the shader behavior)
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha, 0, 255);
                return;
            }
            // use vulkan
            const size_t pixel_size = h * w * c;
            const size_t src_bytes = pixel_size * sizeof(float); // multiple of 4
            const size_t dst_bytes = pixel_size; // uint8_t
            const size_t dst_size = (dst_bytes + 3u) & ~size_t(3u);

            VkBuffer dev_src = 0, dev_dst = 0, host_src = 0, host_dst = 0;
            VkDeviceMemory dev_src_mem = 0, dev_dst_mem = 0, host_src_mem = 0, host_dst_mem = 0;

            bool ok = createBuffer(src_bytes,
                (VkFlags)(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev_src, dev_src_mem);
            ok = ok && createBuffer(dst_size,
                (VkFlags)(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev_dst, dev_dst_mem);
            ok = ok && createBuffer(src_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                host_src, host_src_mem);
            ok = ok && createBuffer(dst_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                host_dst, host_dst_mem);
            if (!ok) {
                destroyBuffer(dev_src, dev_src_mem);
                destroyBuffer(dev_dst, dev_dst_mem);
                destroyBuffer(host_src, host_src_mem);
                destroyBuffer(host_dst, host_dst_mem);
                recordCallFailure("Vulkan chw2hwc buffer allocation failed");
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha, 0, 255);
                return;
            }

            auto cleanup_buffers = [&]() {
                destroyBuffer(dev_src, dev_src_mem);
                destroyBuffer(dev_dst, dev_dst_mem);
                destroyBuffer(host_src, host_src_mem);
                destroyBuffer(host_dst, host_dst_mem);
            };
            auto fallback_to_cpu = [&]() {
                cleanup_buffers();
                recordCallFailure("Vulkan chw2hwc GPU execution failed");
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha, 0, 255);
            };

            if (!writeHostBuffer(host_src_mem, src_bytes, src)) {
                fallback_to_cpu();
                return;
            }

            VkCommandBuffer cmd = 0;
            if (!beginCommandBuffer(cmd)) {
                fallback_to_cpu();
                return;
            }

            // copy host staging buffer to device-local buffer
            VkBufferCopy copy_region = {};
            copy_region.srcOffset = 0;
            copy_region.dstOffset = 0;
            copy_region.size = src_bytes;
            vkCmdCopyBuffer(cmd, host_src, dev_src, 1, &copy_region);

            // barrier: transfer write -> shader read
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            updateDescriptorSet(chw2hwcDescSet, dev_src, src_bytes, dev_dst, dst_size);

            PushConstantsChw2Hwc push_constants = {};
            push_constants.c = (uint32_t)c;
            push_constants.h = (uint32_t)h;
            push_constants.w = (uint32_t)w;
            push_constants.alpha = (float)alpha;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, chw2hwcPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                0, 1, &chw2hwcDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, (uint32_t)sizeof(PushConstantsChw2Hwc), &push_constants);
            const uint32_t group_count = (uint32_t)((dst_size + 255u) / 256u);
            vkCmdDispatch(cmd, group_count, 1, 1);

            // barrier: shader write -> transfer read
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // copy device-local buffer back to host staging buffer
            copy_region.size = dst_bytes;
            vkCmdCopyBuffer(cmd, dev_dst, host_dst, 1, &copy_region);

            if (!submitCommandBuffer(cmd)) {
                fallback_to_cpu();
                return;
            }

            const bool read_ok = readHostBuffer(host_dst_mem, dst_bytes, dst);
            cleanup_buffers();
            if (!read_ok) {
                recordCallFailure("Vulkan chw2hwc result readback failed");
                cpu::chw2hwc<float, uint8_t, true, true>(c, h, w, src, dst, alpha, 0, 255);
            }
            clearCallFailures();
        }

        /**
        * @brief Converts image data from HWC format to CHW format
        *
        * @param h Height of image
        * @param w Width of image
        * @param c Number of channels
        * @param src Vulkan Buffer (uint8_t) Pointer to the source data in HWC format.
        *        Should be padded to a multiple of 4 bytes (createDeviceBuffer does this automatically).
        * @param dst Vulkan Buffer (float) Pointer to the destination data in CHW format
        * @param alpha Scaling factor
        */
        static void hwc2chw(
            const size_t h, const size_t w, const size_t c,
            const VkBuffer src, const VkBuffer dst,
            const float alpha = 1.f / 255.f) {
            vulkan();
            if (h == 0 || w == 0 || c == 0 || src == 0 || dst == 0) {
                return;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                setLastError("Vulkan device-memory conversion called before successful initialization.");
                return;
            }
            VkCommandBuffer cmd = 0;
            if (!beginCommandBuffer(cmd)) {
                recordCallFailure("Vulkan device-memory command buffer setup failed");
                return;
            }

            // barrier: external write -> shader read
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            updateDescriptorSet(hwc2chwDescSet, src, (VkDeviceSize)VK_WHOLE_SIZE, dst, (VkDeviceSize)VK_WHOLE_SIZE);

            PushConstantsHwc2Chw push_constants = {};
            push_constants.h = (uint32_t)h;
            push_constants.w = (uint32_t)w;
            push_constants.c = (uint32_t)c;
            push_constants.alpha = alpha;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hwc2chwPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                0, 1, &hwc2chwDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, (uint32_t)sizeof(PushConstantsHwc2Chw), &push_constants);
            const uint32_t group_count = (uint32_t)((h * w + 255u) / 256u);
            vkCmdDispatch(cmd, group_count, 1, 1);

            // barrier: shader write -> external read
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            if (!submitCommandBuffer(cmd)) {
                recordCallFailure("Vulkan device-memory conversion submission failed");
                return;
            }
            clearCallFailures();
        }

        /**
        * @brief Converts image data from CHW format to HWC format
        *
        * @param c Number of channels
        * @param h Height of image
        * @param w Width of image
        * @param src Vulkan Buffer (float) Pointer to the source data in CHW format
        * @param dst Vulkan Buffer (uint8_t) Pointer to the destination data in HWC format.
        *        Should be padded to a multiple of 4 bytes (createDeviceBuffer does this automatically).
        * @param alpha Scaling factor
        */
        static void chw2hwc(
            const size_t c, const size_t h, const size_t w,
            const VkBuffer src, const VkBuffer dst,
            const uint8_t alpha = 255.0f) {
            vulkan();
            if (c == 0 || h == 0 || w == 0 || src == 0 || dst == 0) {
                return;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                setLastError("Vulkan device-memory conversion called before successful initialization.");
                return;
            }
            VkCommandBuffer cmd = 0;
            if (!beginCommandBuffer(cmd)) {
                recordCallFailure("Vulkan device-memory command buffer setup failed");
                return;
            }

            // barrier: external write -> shader read
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            updateDescriptorSet(chw2hwcDescSet, src, (VkDeviceSize)VK_WHOLE_SIZE, dst, (VkDeviceSize)VK_WHOLE_SIZE);

            PushConstantsChw2Hwc push_constants = {};
            push_constants.c = (uint32_t)c;
            push_constants.h = (uint32_t)h;
            push_constants.w = (uint32_t)w;
            push_constants.alpha = (float)alpha;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, chw2hwcPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                0, 1, &chw2hwcDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, (uint32_t)sizeof(PushConstantsChw2Hwc), &push_constants);
            const size_t total_bytes = h * w * c;
            const size_t dst_size = (total_bytes + 3u) & ~size_t(3u);
            const uint32_t group_count = (uint32_t)((dst_size + 255u) / 256u);
            vkCmdDispatch(cmd, group_count, 1, 1);

            // barrier: shader write -> external read
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            if (!submitCommandBuffer(cmd)) {
                recordCallFailure("Vulkan device-memory conversion submission failed");
                return;
            }
            clearCallFailures();
        }

        /**
        * @brief Creates a device-local Vulkan storage buffer (helper for device-memory usage)
        *
        * The actual allocation is padded up to a multiple of 4 bytes so the
        * buffer can safely hold either float data or packed uint8 data.
        *
        * @param bytes Size of the buffer in bytes
        * @param buffer Output VkBuffer handle
        * @param memory Output VkDeviceMemory handle
        * @return true on success, false otherwise
        */
        static bool createDeviceBuffer(const size_t bytes, VkBuffer* buffer, VkDeviceMemory* memory) {
            vulkan();
            if (bytes == 0 || buffer == nullptr || memory == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                return false;
            }
            // Round the allocation up to 4 bytes: the chw2hwc shader packs
            // four output bytes into every uint32 word, so a uint8 buffer
            // whose size is not a multiple of 4 must still provide padding.
            const size_t padded_bytes = (bytes + 3u) & ~size_t(3u);
            return createBuffer(padded_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, *buffer, *memory);
        }

        /**
        * @brief Destroys a Vulkan buffer created by createDeviceBuffer
        *
        * @param buffer VkBuffer handle
        * @param memory VkDeviceMemory handle
        */
        static void destroyDeviceBuffer(const VkBuffer buffer, const VkDeviceMemory memory) {
            vulkan();
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                return;
            }
            if (buffer != 0) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
            if (memory != 0) {
                vkFreeMemory(device, memory, nullptr);
            }
        }
        /**
        * @brief Uploads host data into an existing device buffer
        *
        * @param buffer Vulkan device buffer handle (created by createDeviceBuffer)
        * @param memory VkDeviceMemory handle of the device buffer
        * @param bytes Number of bytes to upload (may be smaller than the allocation)
        * @param src Pointer to the host source data
        * @return true on success, false otherwise
        */
        static bool uploadToDeviceBuffer(const VkBuffer buffer, const VkDeviceMemory memory,
                                         const size_t bytes, const void* src) {
            vulkan();
            if (buffer == 0 || memory == 0 || bytes == 0 || src == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                return false;
            }
            const size_t padded_bytes = (bytes + 3u) & ~size_t(3u);

            VkBuffer staging = 0;
            VkDeviceMemory staging_memory = 0;
            if (!createBuffer(padded_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                staging, staging_memory)) {
                return false;
            }

            bool ok = writeHostBuffer(staging_memory, bytes, src);
            if (ok) {
                VkCommandBuffer cmd = 0;
                ok = beginCommandBuffer(cmd);
                if (ok) {
                    // copy host staging buffer to the device-local buffer
                    VkBufferCopy copy_region = {};
                    copy_region.srcOffset = 0;
                    copy_region.dstOffset = 0;
                    copy_region.size = (VkDeviceSize)padded_bytes;
                    vkCmdCopyBuffer(cmd, staging, buffer, 1, &copy_region);

                    // barrier: transfer write -> shader read (consumers)
                    VkMemoryBarrier barrier = {};
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

                    ok = submitCommandBuffer(cmd);
                }
            }
            destroyBuffer(staging, staging_memory);
            if (ok) {
                clearCallFailures();
            } else {
                recordCallFailure("Vulkan device-buffer upload failed");
            }
            return ok;
        }

        /**
        * @brief Downloads data from an existing device buffer to host memory
        *
        * @param buffer Vulkan device buffer handle (created by createDeviceBuffer)
        * @param memory VkDeviceMemory handle of the device buffer
        * @param bytes Number of bytes to download (may be smaller than the allocation)
        * @param dst Pointer to the host destination data
        * @return true on success, false otherwise
        */
        static bool downloadFromDeviceBuffer(const VkBuffer buffer, const VkDeviceMemory memory,
                                             const size_t bytes, void* dst) {
            vulkan();
            if (buffer == 0 || memory == 0 || bytes == 0 || dst == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(VulkanMutex);
            if (initVulkanStatus.load(std::memory_order_acquire) != InitVulkanStatusEnum::Inited) {
                return false;
            }
            const size_t padded_bytes = (bytes + 3u) & ~size_t(3u);

            VkBuffer staging = 0;
            VkDeviceMemory staging_memory = 0;
            if (!createBuffer(padded_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                (VkFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                staging, staging_memory)) {
                return false;
            }

            VkCommandBuffer cmd = 0;
            bool ok = beginCommandBuffer(cmd);
            if (ok) {
                // barrier: any producer -> transfer read
                VkMemoryBarrier barrier = {};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

                // copy device-local buffer back to the host staging buffer
                VkBufferCopy copy_region = {};
                copy_region.srcOffset = 0;
                copy_region.dstOffset = 0;
                copy_region.size = (VkDeviceSize)padded_bytes;
                vkCmdCopyBuffer(cmd, buffer, staging, 1, &copy_region);

                ok = submitCommandBuffer(cmd);
            }
            if (ok) {
                ok = readHostBuffer(staging_memory, bytes, dst);
            }
            destroyBuffer(staging, staging_memory);
            if (ok) {
                clearCallFailures();
            } else {
                recordCallFailure("Vulkan device-buffer download failed");
            }
            return ok;
        }

    private:
        struct PushConstantsHwc2Chw {
            uint32_t h;
            uint32_t w;
            uint32_t c;
            float alpha;
        };
        struct PushConstantsChw2Hwc {
            uint32_t c;
            uint32_t h;
            uint32_t w;
            float alpha;
        };

        static void setLastError(const std::string& message)
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            lastVulkanErrorStr = message;
        }

        // Runtime failures use a small backoff threshold so repeated calls do
        // not keep paying submission/setup overhead after the device is broken.
        static void recordCallFailure(const std::string& message)
        {
            const auto failures = ++consecutiveVulkanFailures;
            const std::string fullMessage = message + " (consecutive Vulkan failures: " +
                std::to_string(failures) + ").";
            setLastError(fullMessage);

            if (failures >= maxConsecutiveVulkanFailures)
            {
                initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                setLastError(fullMessage + " Vulkan backend disabled until release()/init().");
            }
        }

        static void clearCallFailures()
        {
            consecutiveVulkanFailures.store(0, std::memory_order_release);
        }

        static bool isFileExists(const std::string& path) {
#ifdef _WIN32
            DWORD fileAttr = GetFileAttributesA(path.c_str());
            return (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY));
#else
            return (access(path.c_str(), F_OK) == 0);
#endif
        }

        static std::string getEnvironmentVariable(const char* name) {
#ifdef _WIN32
            char* buffer = nullptr;
            size_t buffer_size = 0;
            if (_dupenv_s(&buffer, &buffer_size, name) == 0 && buffer != nullptr) {
                std::string result(buffer);
                free(buffer);
                return result;
            }
            return "";
#else
            const char* value = std::getenv(name);
            return value != nullptr ? std::string(value) : "";
#endif
        }

        static std::string findGlslangModuleName() {
#ifdef _WIN32
            char currentDir[MAX_PATH] = { 0 };
            if (GetModuleFileNameA(nullptr, currentDir, MAX_PATH) == 0) {
                setLastError("Failed to get current directory on Windows.");
                return "";
            }
            std::string executablePath(currentDir);
            auto lastSlash = executablePath.find_last_of("\\/");
            if (lastSlash != std::string::npos) {
                executablePath = executablePath.substr(0, lastSlash);
            }
            std::vector<std::string> candidates;
            // next to the executable (release bundles)
            candidates.push_back(executablePath + "\\glslang.dll");
            // Vulkan SDK installation (runtime environment)
            const std::string sdk = getEnvironmentVariable("VULKAN_SDK");
            if (!sdk.empty()) {
                candidates.push_back(sdk + "\\Bin\\glslang.dll");
                candidates.push_back(sdk + "\\Lib\\glslang.dll");
            }
            for (const auto& name : candidates) {
                if (isFileExists(name)) {
                    return name;
                }
            }
            // let the system loader resolve it (application directory, System32)
            return "glslang.dll";
#elif defined(__APPLE__)
            const std::string executablePath = getExecutableDirectory();
            std::vector<std::string> candidates;
            // next to the executable (release bundles)
            candidates.push_back(executablePath + "/libglslang.dylib");
            candidates.push_back(executablePath + "/glslang.dylib");
            // Vulkan SDK installation (runtime environment). VULKAN_SDK usually
            // points to the per-OS arch root (<sdk>/macOS); also accept the SDK
            // root that contains the macOS/ folder.
            const std::string sdk = getEnvironmentVariable("VULKAN_SDK");
            if (!sdk.empty()) {
                candidates.push_back(sdk + "/lib/libglslang.dylib");
                candidates.push_back(sdk + "/macOS/lib/libglslang.dylib");
            }
            for (const auto& name : candidates) {
                if (isFileExists(name)) {
                    return name;
                }
            }
            // let dyld resolve it (DYLD_LIBRARY_PATH / system locations)
            return "libglslang.dylib";
#else
            char currentDir[PATH_MAX] = { 0 };
            if (readlink("/proc/self/exe", currentDir, PATH_MAX) == -1) {
                setLastError("Failed to get current directory on Linux.");
                return "";
            }
            std::string executablePath(currentDir);
            auto lastSlash = executablePath.find_last_of("/");
            if (lastSlash != std::string::npos) {
                executablePath = executablePath.substr(0, lastSlash);
            }
            std::vector<std::string> candidates;
            candidates.push_back(executablePath + "/libglslang.so");
            const std::string sdk = getEnvironmentVariable("VULKAN_SDK");
            if (!sdk.empty()) {
                candidates.push_back(sdk + "/lib/libglslang.so");
            }
            for (const auto& name : candidates) {
                if (isFileExists(name)) {
                    return name;
                }
            }
            return "libglslang.so";
#endif
        }

#if defined(__APPLE__)
        // Directory that contains the running executable. Used on macOS to look
        // for the bundled glslang and MoltenVK dylibs placed next to the binary.
        static std::string getExecutableDirectory() {
            uint32_t path_size = 0;
            if (_NSGetExecutablePath(nullptr, &path_size) != 0 || path_size == 0) {
                setLastError("Failed to get the executable path on macOS.");
                return "";
            }
            std::string executable_path(path_size, '\0');
            if (_NSGetExecutablePath(&executable_path[0], &path_size) != 0) {
                setLastError("Failed to get the executable path on macOS.");
                return "";
            }
            executable_path.resize(std::strlen(executable_path.c_str()));
            auto lastSlash = executable_path.find_last_of('/');
            if (lastSlash == std::string::npos) {
                return "";
            }
            return executable_path.substr(0, lastSlash);
        }
#endif

        static std::string getVulkanModuleName() {
#ifdef _WIN32
            return "vulkan-1.dll";
#elif defined(__APPLE__)
            // macOS has no system Vulkan loader/ICD by default, and the Khronos
            // loader hides MoltenVK devices unless the portability enumeration
            // extension is enabled at instance creation. Loading libMoltenVK.dylib
            // directly is simpler: MoltenVK is a complete Vulkan implementation
            // over Metal and exports the vk* entry points itself, so no loader or
            // ICD configuration is required.
            const std::string executablePath = getExecutableDirectory();
            std::vector<std::string> candidates;
            // next to the executable (release bundles)
            candidates.push_back(executablePath + "/libMoltenVK.dylib");
            // Vulkan SDK installation (runtime environment)
            const std::string sdk = getEnvironmentVariable("VULKAN_SDK");
            if (!sdk.empty()) {
                candidates.push_back(sdk + "/lib/libMoltenVK.dylib");
                candidates.push_back(sdk + "/macOS/lib/libMoltenVK.dylib");
            }
            for (const auto& name : candidates) {
                if (isFileExists(name)) {
                    return name;
                }
            }
            // let dyld resolve it (DYLD_LIBRARY_PATH / system locations)
            return "libMoltenVK.dylib";
#else
            return "libvulkan.so.1";
#endif
        }

        static bool loadGlslangApi(const std::string& library_name) {
            auto* dl_manager = DynamicLibraryManager::instance();
            auto glslang_lib = dl_manager->loadLibrary(library_name);
            if (!glslang_lib) {
                setLastError("Failed to load glslang library: " + library_name + ".");
                return false;
            }
            glslang_initialize_process = (glslang_initialize_process_t)(dl_manager->getFunction(library_name, "glslang_initialize_process"));
            glslang_finalize_process = (glslang_finalize_process_t)(dl_manager->getFunction(library_name, "glslang_finalize_process"));
            glslang_shader_create = (glslang_shader_create_t)(dl_manager->getFunction(library_name, "glslang_shader_create"));
            glslang_shader_delete = (glslang_shader_delete_t)(dl_manager->getFunction(library_name, "glslang_shader_delete"));
            glslang_shader_preprocess = (glslang_shader_preprocess_t)(dl_manager->getFunction(library_name, "glslang_shader_preprocess"));
            glslang_shader_parse = (glslang_shader_parse_t)(dl_manager->getFunction(library_name, "glslang_shader_parse"));
            glslang_shader_get_info_log = (glslang_shader_get_info_log_t)(dl_manager->getFunction(library_name, "glslang_shader_get_info_log"));
            glslang_program_create = (glslang_program_create_t)(dl_manager->getFunction(library_name, "glslang_program_create"));
            glslang_program_delete = (glslang_program_delete_t)(dl_manager->getFunction(library_name, "glslang_program_delete"));
            glslang_program_add_shader = (glslang_program_add_shader_t)(dl_manager->getFunction(library_name, "glslang_program_add_shader"));
            glslang_program_link = (glslang_program_link_t)(dl_manager->getFunction(library_name, "glslang_program_link"));
            glslang_program_get_info_log = (glslang_program_get_info_log_t)(dl_manager->getFunction(library_name, "glslang_program_get_info_log"));
            glslang_program_SPIRV_generate = (glslang_program_SPIRV_generate_t)(dl_manager->getFunction(library_name, "glslang_program_SPIRV_generate"));
            glslang_program_SPIRV_get_size = (glslang_program_SPIRV_get_size_t)(dl_manager->getFunction(library_name, "glslang_program_SPIRV_get_size"));
            glslang_program_SPIRV_get = (glslang_program_SPIRV_get_t)(dl_manager->getFunction(library_name, "glslang_program_SPIRV_get"));

            if (!glslang_initialize_process || !glslang_finalize_process ||
                !glslang_shader_create || !glslang_shader_delete || !glslang_shader_parse ||
                !glslang_shader_get_info_log || !glslang_shader_preprocess ||
                !glslang_program_create || !glslang_program_delete ||
                !glslang_program_add_shader || !glslang_program_link || !glslang_program_get_info_log ||
                !glslang_program_SPIRV_generate || !glslang_program_SPIRV_get_size || !glslang_program_SPIRV_get) {
                setLastError("Failed to load one or more glslang functions from: " + library_name + ".");
                dl_manager->unloadLibrary(library_name);
                return false;
            }
            return true;
        }

        static glslang_resource_t createDefaultResource() {
            glslang_resource_t r;
            std::memset(&r, 0, sizeof(r));
            r.max_lights = 32;
            r.max_clip_planes = 6;
            r.max_texture_units = 32;
            r.max_texture_coords = 32;
            r.max_vertex_attribs = 64;
            r.max_vertex_uniform_components = 4096;
            r.max_varying_floats = 64;
            r.max_vertex_texture_image_units = 32;
            r.max_combined_texture_image_units = 80;
            r.max_texture_image_units = 32;
            r.max_fragment_uniform_components = 4096;
            r.max_draw_buffers = 32;
            r.max_vertex_uniform_vectors = 128;
            r.max_varying_vectors = 8;
            r.max_fragment_uniform_vectors = 16;
            r.max_vertex_output_vectors = 16;
            r.max_fragment_input_vectors = 15;
            r.min_program_texel_offset = -8;
            r.max_program_texel_offset = 7;
            r.max_clip_distances = 8;
            r.max_compute_work_group_count_x = 65535;
            r.max_compute_work_group_count_y = 65535;
            r.max_compute_work_group_count_z = 65535;
            r.max_compute_work_group_size_x = 1024;
            r.max_compute_work_group_size_y = 1024;
            r.max_compute_work_group_size_z = 64;
            r.max_compute_uniform_components = 1024;
            r.max_compute_texture_image_units = 16;
            r.max_compute_image_uniforms = 8;
            r.max_compute_atomic_counters = 8;
            r.max_compute_atomic_counter_buffers = 1;
            r.max_varying_components = 64;
            r.max_vertex_output_components = 60;
            r.max_geometry_input_components = 64;
            r.max_geometry_output_components = 64;
            r.max_fragment_input_components = 128;
            r.max_image_units = 8;
            r.max_combined_image_units_and_fragment_outputs = 8;
            r.max_combined_shader_output_resources = 8;
            r.max_image_samples = 0;
            r.max_vertex_image_uniforms = 4;
            r.max_tess_control_image_uniforms = 4;
            r.max_tess_evaluation_image_uniforms = 4;
            r.max_geometry_image_uniforms = 4;
            r.max_fragment_image_uniforms = 4;
            r.max_combined_image_uniforms = 8;
            r.max_geometry_texture_image_units = 16;
            r.max_geometry_output_vertices = 256;
            r.max_geometry_total_output_components = 1024;
            r.max_geometry_uniform_components = 4096;
            r.max_geometry_varying_components = 64;
            r.max_tess_control_input_components = 64;
            r.max_tess_control_output_components = 64;
            r.max_tess_control_texture_image_units = 16;
            r.max_tess_control_uniform_components = 1024;
            r.max_tess_control_total_output_components = 128;
            r.max_tess_evaluation_input_components = 64;
            r.max_tess_evaluation_output_components = 64;
            r.max_tess_evaluation_texture_image_units = 16;
            r.max_tess_evaluation_uniform_components = 1024;
            r.max_tess_patch_components = 128;
            r.max_patch_vertices = 32;
            r.max_tess_gen_level = 64;
            r.max_viewports = 16;
            r.max_vertex_atomic_counters = 8;
            r.max_tess_control_atomic_counters = 8;
            r.max_tess_evaluation_atomic_counters = 8;
            r.max_geometry_atomic_counters = 8;
            r.max_fragment_atomic_counters = 8;
            r.max_combined_atomic_counters = 8;
            r.max_atomic_counter_bindings = 1;
            r.max_vertex_atomic_counter_buffers = 1;
            r.max_tess_control_atomic_counter_buffers = 0;
            r.max_tess_evaluation_atomic_counter_buffers = 0;
            r.max_geometry_atomic_counter_buffers = 0;
            r.max_fragment_atomic_counter_buffers = 1;
            r.max_combined_atomic_counter_buffers = 1;
            r.max_atomic_counter_buffer_size = 16384;
            r.max_transform_feedback_buffers = 4;
            r.max_transform_feedback_interleaved_components = 64;
            r.max_cull_distances = 8;
            r.max_combined_clip_and_cull_distances = 16;
            r.max_samples = 4;
            r.max_mesh_output_vertices_nv = 256;
            r.max_mesh_output_primitives_nv = 512;
            r.max_mesh_work_group_size_x_nv = 32;
            r.max_mesh_work_group_size_y_nv = 32;
            r.max_mesh_work_group_size_z_nv = 32;
            r.max_task_work_group_size_x_nv = 32;
            r.max_task_work_group_size_y_nv = 32;
            r.max_task_work_group_size_z_nv = 32;
            r.max_mesh_view_count_nv = 4;
            r.max_mesh_output_vertices_ext = 256;
            r.max_mesh_output_primitives_ext = 512;
            r.max_mesh_work_group_size_x_ext = 32;
            r.max_mesh_work_group_size_y_ext = 32;
            r.max_mesh_work_group_size_z_ext = 32;
            r.max_task_work_group_size_x_ext = 32;
            r.max_task_work_group_size_y_ext = 32;
            r.max_task_work_group_size_z_ext = 32;
            r.max_mesh_view_count_ext = 4;
            r.max_dual_source_draw_buffers_ext = 1;
            r.limits.non_inductive_for_loops = true;
            r.limits.while_loops = true;
            r.limits.do_while_loops = true;
            r.limits.general_uniform_indexing = true;
            r.limits.general_attribute_matrix_vector_indexing = true;
            r.limits.general_varying_indexing = true;
            r.limits.general_sampler_indexing = true;
            r.limits.general_variable_indexing = true;
            r.limits.general_constant_matrix_vector_indexing = true;
            return r;
        }

        static bool compileGLSLWithGlslang(const std::string& library_name, const char* glsl_source, std::vector<uint32_t>& spirv) {
            glslang_resource_t resource = createDefaultResource();
            glslang_input_t input;
            std::memset(&input, 0, sizeof(input));
            input.language = GLSLANG_SOURCE_GLSL;
            input.stage = GLSLANG_STAGE_COMPUTE;
            input.client = GLSLANG_CLIENT_VULKAN;
            input.client_version = GLSLANG_TARGET_VULKAN_1_0;
            input.target_language = GLSLANG_TARGET_SPV;
            input.target_language_version = GLSLANG_TARGET_SPV_1_0;
            input.code = glsl_source;
            input.default_version = 100;
            input.default_profile = GLSLANG_NO_PROFILE;
            input.force_default_version_and_profile = 0;
            input.forward_compatible = 0;
            input.messages = (glslang_messages_t)(GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
            input.resource = &resource;

            glslang_shader_t* shader = glslang_shader_create(&input);
            if (shader == nullptr) {
                setLastError("glslang_shader_create failed.");
                return false;
            }
            int preprocessed = glslang_shader_preprocess(shader, &input);
            if (preprocessed == 0) {
                const char* log = glslang_shader_get_info_log(shader);
                setLastError(std::string("glslang shader preprocess error: ") + (log != nullptr ? log : ""));
                glslang_shader_delete(shader);
                return false;
            }
            int parsed = glslang_shader_parse(shader, &input);
            if (parsed == 0) {
                const char* log = glslang_shader_get_info_log(shader);
                setLastError(std::string("glslang shader parse error: ") + (log != nullptr ? log : ""));
                glslang_shader_delete(shader);
                return false;
            }
            glslang_program_t* program = glslang_program_create();
            if (program == nullptr) {
                setLastError("glslang_program_create failed.");
                glslang_shader_delete(shader);
                return false;
            }
            glslang_program_add_shader(program, shader);
            int linked = glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
            if (linked == 0) {
                const char* log = glslang_program_get_info_log(program);
                setLastError(std::string("glslang program link error: ") + (log != nullptr ? log : ""));
                glslang_program_delete(program);
                glslang_shader_delete(shader);
                return false;
            }
            glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
            const size_t word_count = glslang_program_SPIRV_get_size(program);
            spirv.resize(word_count);
            if (word_count > 0) {
                glslang_program_SPIRV_get(program, spirv.data());
            }
            glslang_program_delete(program);
            glslang_shader_delete(shader);
            return true;
        }

        static uint32_t findMemoryType(const uint32_t type_bits, const uint32_t required_flags) {
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
                if ((type_bits & (1u << i)) != 0 &&
                    ((memoryProperties.memoryTypes[i].propertyFlags & required_flags) == required_flags)) {
                    return i;
                }
            }
            return ~0u;
        }

        static bool createBuffer(const size_t size, const VkFlags usage, const uint32_t memory_flags,
                                 VkBuffer& buffer, VkDeviceMemory& memory) {
            buffer = 0;
            memory = 0;
            VkBufferCreateInfo buffer_info = {};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = (VkDeviceSize)size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
                setLastError("vkCreateBuffer failed.");
                return false;
            }
            VkMemoryRequirements mem_req;
            std::memset(&mem_req, 0, sizeof(mem_req));
            vkGetBufferMemoryRequirements(device, buffer, &mem_req);
            const uint32_t memory_type_index = findMemoryType(mem_req.memoryTypeBits, memory_flags);
            if (memory_type_index == ~0u) {
                setLastError("No suitable Vulkan memory type found for buffer allocation.");
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = 0;
                return false;
            }
            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = mem_req.size;
            alloc_info.memoryTypeIndex = memory_type_index;
            if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
                setLastError("vkAllocateMemory failed.");
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = 0;
                return false;
            }
            if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
                setLastError("vkBindBufferMemory failed.");
                vkFreeMemory(device, memory, nullptr);
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = 0;
                memory = 0;
                return false;
            }
            return true;
        }

        static void destroyBuffer(VkBuffer& buffer, VkDeviceMemory& memory) {
            if (buffer != 0) {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = 0;
            }
            if (memory != 0) {
                vkFreeMemory(device, memory, nullptr);
                memory = 0;
            }
        }

        static bool writeHostBuffer(const VkDeviceMemory memory, const size_t size, const void* data) {
            void* mapped = nullptr;
            if (vkMapMemory(device, memory, 0, (VkDeviceSize)size, 0, &mapped) != VK_SUCCESS) {
                setLastError("vkMapMemory (host write) failed.");
                return false;
            }
            std::memcpy(mapped, data, size);
            vkUnmapMemory(device, memory);
            return true;
        }

        static bool readHostBuffer(const VkDeviceMemory memory, const size_t size, void* data) {
            void* mapped = nullptr;
            if (vkMapMemory(device, memory, 0, (VkDeviceSize)size, 0, &mapped) != VK_SUCCESS) {
                setLastError("vkMapMemory (host read) failed.");
                return false;
            }
            std::memcpy(data, mapped, size);
            vkUnmapMemory(device, memory);
            return true;
        }

        static bool beginCommandBuffer(VkCommandBuffer& cmd) {
            VkCommandBufferAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = commandPool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &alloc_info, &cmd) != VK_SUCCESS) {
                setLastError("vkAllocateCommandBuffers failed.");
                return false;
            }
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
                setLastError("vkBeginCommandBuffer failed.");
                vkFreeCommandBuffers(device, commandPool, 1, &cmd);
                cmd = 0;
                return false;
            }
            return true;
        }

        static bool submitCommandBuffer(VkCommandBuffer& cmd) {
            VkResult res = vkEndCommandBuffer(cmd);
            if (res != VK_SUCCESS) {
                setLastError("vkEndCommandBuffer failed with error " + std::to_string(res) + ".");
                vkFreeCommandBuffers(device, commandPool, 1, &cmd);
                cmd = 0;
                return false;
            }
            VkSubmitInfo submit_info = {};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            res = vkQueueSubmit(queue, 1, &submit_info, (VkFence)VK_NULL_HANDLE);
            if (res != VK_SUCCESS) {
                setLastError("vkQueueSubmit failed with error " + std::to_string(res) + ".");
                vkFreeCommandBuffers(device, commandPool, 1, &cmd);
                cmd = 0;
                return false;
            }
            res = vkQueueWaitIdle(queue);
            if (res != VK_SUCCESS) {
                setLastError("vkQueueWaitIdle failed with error " + std::to_string(res) + ".");
            }
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            cmd = 0;
            return res == VK_SUCCESS;
        }

        static bool updateDescriptorSet(const VkDescriptorSet desc_set, const VkBuffer src, const VkDeviceSize src_size,
                                        const VkBuffer dst, const VkDeviceSize dst_size) {
            VkDescriptorBufferInfo src_info = {};
            src_info.buffer = src;
            src_info.offset = 0;
            src_info.range = src_size;
            VkDescriptorBufferInfo dst_info = {};
            dst_info.buffer = dst;
            dst_info.offset = 0;
            dst_info.range = dst_size;
            VkWriteDescriptorSet writes[2] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = desc_set;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &src_info;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = desc_set;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &dst_info;
            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
            return true;
        }

        static bool selectPhysicalDevice(std::vector<VkPhysicalDevice>& devices, VkPhysicalDevice& selected_device) {
            uint32_t device_count = 0;
            VkResult res = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
            if (res != VK_SUCCESS || device_count == 0) {
                setLastError("vkEnumeratePhysicalDevices failed or no Vulkan device found.");
                return false;
            }
            devices.resize(device_count);
            res = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
            if (res != VK_SUCCESS) {
                setLastError("vkEnumeratePhysicalDevices failed.");
                return false;
            }

            // Prefer the discrete GPU (strongest compute) first, then the device
            // with the most device-local memory, then the newest API version.
            int best_rank = -1;
            uint64_t best_memory = 0;
            uint32_t best_api = 0;
            uint32_t best_index = 0;
            bool found = false;
            for (uint32_t i = 0; i < device_count; ++i) {
                VkPhysicalDevicePropertiesRaw props;
                std::memset(&props, 0, sizeof(props));
                vkGetPhysicalDeviceProperties(devices[i], &props);

                VkPhysicalDeviceMemoryProperties mem_props;
                std::memset(&mem_props, 0, sizeof(mem_props));
                vkGetPhysicalDeviceMemoryProperties(devices[i], &mem_props);

                uint32_t family_count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, nullptr);
                std::vector<VkQueueFamilyProperties> families(family_count);
                if (family_count > 0) {
                    vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, families.data());
                }
                bool has_compute = false;
                for (const auto& family : families) {
                    if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                        has_compute = true;
                        break;
                    }
                }
                if (!has_compute) {
                    continue;
                }

                uint64_t device_memory = 0;
                for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
                    if ((mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                        device_memory += mem_props.memoryHeaps[h].size;
                    }
                }
                int rank = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2
                         : (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 1 : 0;
                if (!found || rank > best_rank ||
                    (rank == best_rank && device_memory > best_memory) ||
                    (rank == best_rank && device_memory == best_memory && props.apiVersion > best_api)) {
                    best_rank = rank;
                    best_memory = device_memory;
                    best_api = props.apiVersion;
                    best_index = i;
                    found = true;
                }
            }
            if (!found) {
                setLastError("No Vulkan physical device with compute support found.");
                return false;
            }
            selected_device = devices[best_index];

            // cache the memory properties of the selected device
            std::memset(&memoryProperties, 0, sizeof(memoryProperties));
            vkGetPhysicalDeviceMemoryProperties(selected_device, &memoryProperties);

            VkPhysicalDevicePropertiesRaw props;
            std::memset(&props, 0, sizeof(props));
            vkGetPhysicalDeviceProperties(selected_device, &props);
            //std::cout << "Vulkan selected device: " << props.deviceName << std::endl;
            return true;
        }

        static bool createShaderModule(const std::vector<uint32_t>& spirv, VkShaderModule& module) {
            VkShaderModuleCreateInfo module_info = {};
            module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            module_info.codeSize = spirv.size() * sizeof(uint32_t);
            module_info.pCode = spirv.data();
            VkResult res = vkCreateShaderModule(device, &module_info, nullptr, &module);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateShaderModule failed with error " + std::to_string(res) + ".");
                return false;
            }
            return true;
        }

        static bool createComputePipeline(const VkShaderModule module, VkPipeline& pipeline) {
            VkPipelineShaderStageCreateInfo stage_info = {};
            stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage_info.module = module;
            stage_info.pName = "main";
            VkComputePipelineCreateInfo pipeline_info = {};
            pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.stage = stage_info;
            pipeline_info.layout = pipelineLayout;
            VkResult res = vkCreateComputePipelines(device, (VkPipelineCache)VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateComputePipelines failed with error " + std::to_string(res) + ".");
                return false;
            }
            return true;
        }

        static bool initVulkanDriverAPI() {
            const std::string vulkan_lib = getVulkanModuleName();
            vulkan_library_name = vulkan_lib;
            auto* dl_manager = DynamicLibraryManager::instance();
            auto vulkan_driver = dl_manager->loadLibrary(vulkan_lib);
            if (!vulkan_driver) {
                setLastError("Failed to load Vulkan driver library: " + vulkan_lib + ".");
                return false;
            }
            vkCreateInstance = (vkCreateInstance_t)(dl_manager->getFunction(vulkan_lib, "vkCreateInstance"));
            vkDestroyInstance = (vkDestroyInstance_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyInstance"));
            vkEnumeratePhysicalDevices = (vkEnumeratePhysicalDevices_t)(dl_manager->getFunction(vulkan_lib, "vkEnumeratePhysicalDevices"));
            vkGetPhysicalDeviceProperties = (vkGetPhysicalDeviceProperties_t)(dl_manager->getFunction(vulkan_lib, "vkGetPhysicalDeviceProperties"));
            vkGetPhysicalDeviceMemoryProperties = (vkGetPhysicalDeviceMemoryProperties_t)(dl_manager->getFunction(vulkan_lib, "vkGetPhysicalDeviceMemoryProperties"));
            vkGetPhysicalDeviceQueueFamilyProperties = (vkGetPhysicalDeviceQueueFamilyProperties_t)(dl_manager->getFunction(vulkan_lib, "vkGetPhysicalDeviceQueueFamilyProperties"));
            vkCreateDevice = (vkCreateDevice_t)(dl_manager->getFunction(vulkan_lib, "vkCreateDevice"));
            vkDestroyDevice = (vkDestroyDevice_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyDevice"));
            vkGetDeviceQueue = (vkGetDeviceQueue_t)(dl_manager->getFunction(vulkan_lib, "vkGetDeviceQueue"));
            vkDeviceWaitIdle = (vkDeviceWaitIdle_t)(dl_manager->getFunction(vulkan_lib, "vkDeviceWaitIdle"));
            vkQueueWaitIdle = (vkQueueWaitIdle_t)(dl_manager->getFunction(vulkan_lib, "vkQueueWaitIdle"));
            vkQueueSubmit = (vkQueueSubmit_t)(dl_manager->getFunction(vulkan_lib, "vkQueueSubmit"));
            vkCreateShaderModule = (vkCreateShaderModule_t)(dl_manager->getFunction(vulkan_lib, "vkCreateShaderModule"));
            vkDestroyShaderModule = (vkDestroyShaderModule_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyShaderModule"));
            vkCreateDescriptorSetLayout = (vkCreateDescriptorSetLayout_t)(dl_manager->getFunction(vulkan_lib, "vkCreateDescriptorSetLayout"));
            vkDestroyDescriptorSetLayout = (vkDestroyDescriptorSetLayout_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyDescriptorSetLayout"));
            vkCreateDescriptorPool = (vkCreateDescriptorPool_t)(dl_manager->getFunction(vulkan_lib, "vkCreateDescriptorPool"));
            vkDestroyDescriptorPool = (vkDestroyDescriptorPool_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyDescriptorPool"));
            vkAllocateDescriptorSets = (vkAllocateDescriptorSets_t)(dl_manager->getFunction(vulkan_lib, "vkAllocateDescriptorSets"));
            vkUpdateDescriptorSets = (vkUpdateDescriptorSets_t)(dl_manager->getFunction(vulkan_lib, "vkUpdateDescriptorSets"));
            vkCreatePipelineLayout = (vkCreatePipelineLayout_t)(dl_manager->getFunction(vulkan_lib, "vkCreatePipelineLayout"));
            vkDestroyPipelineLayout = (vkDestroyPipelineLayout_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyPipelineLayout"));
            vkCreateComputePipelines = (vkCreateComputePipelines_t)(dl_manager->getFunction(vulkan_lib, "vkCreateComputePipelines"));
            vkDestroyPipeline = (vkDestroyPipeline_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyPipeline"));
            vkCreateCommandPool = (vkCreateCommandPool_t)(dl_manager->getFunction(vulkan_lib, "vkCreateCommandPool"));
            vkDestroyCommandPool = (vkDestroyCommandPool_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyCommandPool"));
            vkAllocateCommandBuffers = (vkAllocateCommandBuffers_t)(dl_manager->getFunction(vulkan_lib, "vkAllocateCommandBuffers"));
            vkFreeCommandBuffers = (vkFreeCommandBuffers_t)(dl_manager->getFunction(vulkan_lib, "vkFreeCommandBuffers"));
            vkBeginCommandBuffer = (vkBeginCommandBuffer_t)(dl_manager->getFunction(vulkan_lib, "vkBeginCommandBuffer"));
            vkEndCommandBuffer = (vkEndCommandBuffer_t)(dl_manager->getFunction(vulkan_lib, "vkEndCommandBuffer"));
            vkCmdBindPipeline = (vkCmdBindPipeline_t)(dl_manager->getFunction(vulkan_lib, "vkCmdBindPipeline"));
            vkCmdBindDescriptorSets = (vkCmdBindDescriptorSets_t)(dl_manager->getFunction(vulkan_lib, "vkCmdBindDescriptorSets"));
            vkCmdPushConstants = (vkCmdPushConstants_t)(dl_manager->getFunction(vulkan_lib, "vkCmdPushConstants"));
            vkCmdDispatch = (vkCmdDispatch_t)(dl_manager->getFunction(vulkan_lib, "vkCmdDispatch"));
            vkCmdPipelineBarrier = (vkCmdPipelineBarrier_t)(dl_manager->getFunction(vulkan_lib, "vkCmdPipelineBarrier"));
            vkCmdCopyBuffer = (vkCmdCopyBuffer_t)(dl_manager->getFunction(vulkan_lib, "vkCmdCopyBuffer"));
            vkCreateBuffer = (vkCreateBuffer_t)(dl_manager->getFunction(vulkan_lib, "vkCreateBuffer"));
            vkDestroyBuffer = (vkDestroyBuffer_t)(dl_manager->getFunction(vulkan_lib, "vkDestroyBuffer"));
            vkGetBufferMemoryRequirements = (vkGetBufferMemoryRequirements_t)(dl_manager->getFunction(vulkan_lib, "vkGetBufferMemoryRequirements"));
            vkAllocateMemory = (vkAllocateMemory_t)(dl_manager->getFunction(vulkan_lib, "vkAllocateMemory"));
            vkFreeMemory = (vkFreeMemory_t)(dl_manager->getFunction(vulkan_lib, "vkFreeMemory"));
            vkBindBufferMemory = (vkBindBufferMemory_t)(dl_manager->getFunction(vulkan_lib, "vkBindBufferMemory"));
            vkMapMemory = (vkMapMemory_t)(dl_manager->getFunction(vulkan_lib, "vkMapMemory"));
            vkUnmapMemory = (vkUnmapMemory_t)(dl_manager->getFunction(vulkan_lib, "vkUnmapMemory"));

            if (!vkCreateInstance || !vkDestroyInstance || !vkEnumeratePhysicalDevices ||
                !vkGetPhysicalDeviceProperties || !vkGetPhysicalDeviceMemoryProperties ||
                !vkGetPhysicalDeviceQueueFamilyProperties || !vkCreateDevice || !vkDestroyDevice ||
                !vkGetDeviceQueue || !vkDeviceWaitIdle || !vkQueueWaitIdle || !vkQueueSubmit ||
                !vkCreateShaderModule || !vkDestroyShaderModule || !vkCreateDescriptorSetLayout ||
                !vkDestroyDescriptorSetLayout || !vkCreateDescriptorPool || !vkDestroyDescriptorPool ||
                !vkAllocateDescriptorSets || !vkUpdateDescriptorSets || !vkCreatePipelineLayout ||
                !vkDestroyPipelineLayout || !vkCreateComputePipelines || !vkDestroyPipeline ||
                !vkCreateCommandPool || !vkDestroyCommandPool || !vkAllocateCommandBuffers ||
                !vkFreeCommandBuffers || !vkBeginCommandBuffer || !vkEndCommandBuffer ||
                !vkCmdBindPipeline || !vkCmdBindDescriptorSets || !vkCmdPushConstants ||
                !vkCmdDispatch || !vkCmdPipelineBarrier || !vkCmdCopyBuffer ||
                !vkCreateBuffer || !vkDestroyBuffer || !vkGetBufferMemoryRequirements ||
                !vkAllocateMemory || !vkFreeMemory || !vkBindBufferMemory || !vkMapMemory || !vkUnmapMemory) {
                setLastError("Failed to load one or more Vulkan functions from: " + vulkan_lib + ".");
                return false;
            }
            return true;
        }

        static bool initVulkanFunctions(const std::vector<uint32_t>& spv_hwc2chw, const std::vector<uint32_t>& spv_chw2hwc) {
            VkApplicationInfo app_info = {};
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "FastChwHwcConverter";
            app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            app_info.pEngineName = "FastChwHwcConverter";
            app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            app_info.apiVersion = VK_API_VERSION_1_0;
            VkInstanceCreateInfo instance_info = {};
            instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            instance_info.pApplicationInfo = &app_info;
            VkResult res = vkCreateInstance(&instance_info, nullptr, &instance);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateInstance failed with error " + std::to_string(res) + ".");
                return false;
            }

            std::vector<VkPhysicalDevice> devices;
            if (!selectPhysicalDevice(devices, physicalDevice)) {
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            if (family_count > 0) {
                vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &family_count, families.data());
            }
            uint32_t queue_family = ~0u;
            for (uint32_t i = 0; i < family_count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    queue_family = i;
                    break;
                }
            }
            if (queue_family == ~0u) {
                setLastError("No compute-capable Vulkan queue family found.");
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            queueFamilyIndex = queue_family;

            const float queue_priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info = {};
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = queue_family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &queue_priority;
            VkDeviceCreateInfo device_info = {};
            device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue_info;
            res = vkCreateDevice(physicalDevice, &device_info, nullptr, &device);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateDevice failed with error " + std::to_string(res) + ".");
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            vkGetDeviceQueue(device, queue_family, 0, &queue);

            if (!createShaderModule(spv_hwc2chw, hwc2chwModule)) {
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            if (!createShaderModule(spv_chw2hwc, chw2hwcModule)) {
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            VkDescriptorSetLayoutBinding bindings[2] = {};
            for (uint32_t i = 0; i < 2; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo layout_info = {};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 2;
            layout_info.pBindings = bindings;
            res = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptorSetLayout);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateDescriptorSetLayout failed with error " + std::to_string(res) + ".");
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            VkPushConstantRange push_range = {};
            push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push_range.offset = 0;
            push_range.size = 16;
            VkPipelineLayoutCreateInfo pipeline_layout_info = {};
            pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptorSetLayout;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push_range;
            res = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipelineLayout);
            if (res != VK_SUCCESS) {
                setLastError("vkCreatePipelineLayout failed with error " + std::to_string(res) + ".");
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            VkDescriptorPoolSize pool_size = {};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = 4;
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = 2;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            res = vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptorPool);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateDescriptorPool failed with error " + std::to_string(res) + ".");
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = 0;
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            VkDescriptorSetLayout set_layouts[2] = { descriptorSetLayout, descriptorSetLayout };
            VkDescriptorSetAllocateInfo set_alloc_info = {};
            set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_alloc_info.descriptorPool = descriptorPool;
            set_alloc_info.descriptorSetCount = 2;
            set_alloc_info.pSetLayouts = set_layouts;
            VkDescriptorSet sets[2] = { 0, 0 };
            res = vkAllocateDescriptorSets(device, &set_alloc_info, sets);
            if (res != VK_SUCCESS) {
                setLastError("vkAllocateDescriptorSets failed with error " + std::to_string(res) + ".");
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = 0;
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = 0;
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            hwc2chwDescSet = sets[0];
            chw2hwcDescSet = sets[1];

            if (!createComputePipeline(hwc2chwModule, hwc2chwPipeline)) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = 0;
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = 0;
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            if (!createComputePipeline(chw2hwcModule, chw2hwcPipeline)) {
                vkDestroyPipeline(device, hwc2chwPipeline, nullptr);
                hwc2chwPipeline = 0;
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = 0;
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = 0;
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }

            VkCommandPoolCreateInfo cmd_pool_info = {};
            cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            cmd_pool_info.queueFamilyIndex = queue_family;
            res = vkCreateCommandPool(device, &cmd_pool_info, nullptr, &commandPool);
            if (res != VK_SUCCESS) {
                setLastError("vkCreateCommandPool failed with error " + std::to_string(res) + ".");
                vkDestroyPipeline(device, chw2hwcPipeline, nullptr);
                vkDestroyPipeline(device, hwc2chwPipeline, nullptr);
                chw2hwcPipeline = 0;
                hwc2chwPipeline = 0;
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = 0;
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = 0;
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = 0;
                vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                chw2hwcModule = 0;
                hwc2chwModule = 0;
                vkDestroyDevice(device, nullptr);
                device = 0;
                vkDestroyInstance(instance, nullptr);
                instance = 0;
                return false;
            }
            return true;
        }

        static bool initAll() {
            std::lock_guard<std::mutex> lock(VulkanMutex);
            setLastError("");
            if (initVulkanStatus.load(std::memory_order_acquire) == InitVulkanStatusEnum::Ready) {
                std::string glslang_module = findGlslangModuleName();
                if (glslang_module.empty()) {
                    setLastError("Could not find glslang library.");
                    lastVulkanErrorStr = "Could not find glslang library.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                glslang_library_name = glslang_module;
                if (!loadGlslangApi(glslang_module)) {
                    setLastError("Failed to load glslang functions.");
                    lastVulkanErrorStr = "Failed to load glslang functions.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                if (glslang_initialize_process() == 0) {
                    setLastError("glslang_initialize_process failed.");
                    lastVulkanErrorStr = "glslang_initialize_process failed.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                std::vector<uint32_t> spv_hwc2chw;
                std::vector<uint32_t> spv_chw2hwc;
                if (!compileGLSLWithGlslang(glslang_module, vulkanHwc2ChwSource, spv_hwc2chw)) {
                    setLastError("Compile Vulkan HWC->CHW shader failed.");
                    lastVulkanErrorStr = "Compile Vulkan HWC->CHW shader failed.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                if (!compileGLSLWithGlslang(glslang_module, vulkanChw2HwcSource, spv_chw2hwc)) {
                    setLastError("Compile Vulkan CHW->HWC shader failed.");
                    lastVulkanErrorStr = "Compile Vulkan CHW->HWC shader failed.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                if (!initVulkanDriverAPI()) {
                    setLastError("Failed to load Vulkan driver API functions.");
                    lastVulkanErrorStr = "Failed to load Vulkan driver API functions.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                if (!initVulkanFunctions(spv_hwc2chw, spv_chw2hwc)) {
                    setLastError("Failed to initialize Vulkan device and pipelines.");
                    lastVulkanErrorStr = "Failed to initialize Vulkan device and pipelines.";
                    initVulkanStatus.store(InitVulkanStatusEnum::Failed, std::memory_order_release);
                    return false;
                }
                initVulkanStatus.store(InitVulkanStatusEnum::Inited, std::memory_order_release);
                return true;
            }
            else if (initVulkanStatus.load(std::memory_order_acquire) == InitVulkanStatusEnum::Inited) {
                return true;
            }
            else if (initVulkanStatus.load(std::memory_order_acquire) == InitVulkanStatusEnum::Failed) {
                setLastError("Vulkan initialization failed. " + lastError());
                return false;
            }
            return true;
        }

        static bool releaseAll() {
            std::lock_guard<std::mutex> lock(VulkanMutex);
            auto* dl_manager = DynamicLibraryManager::instance();
            if (device != 0) {
                if (commandPool != 0) {
                    vkDestroyCommandPool(device, commandPool, nullptr);
                    commandPool = 0;
                }
                if (chw2hwcPipeline != 0) {
                    vkDestroyPipeline(device, chw2hwcPipeline, nullptr);
                    chw2hwcPipeline = 0;
                }
                if (hwc2chwPipeline != 0) {
                    vkDestroyPipeline(device, hwc2chwPipeline, nullptr);
                    hwc2chwPipeline = 0;
                }
                if (pipelineLayout != 0) {
                    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                    pipelineLayout = 0;
                }
                if (descriptorSetLayout != 0) {
                    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                    descriptorSetLayout = 0;
                }
                if (descriptorPool != 0) {
                    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                    descriptorPool = 0;
                }
                if (chw2hwcModule != 0) {
                    vkDestroyShaderModule(device, chw2hwcModule, nullptr);
                    chw2hwcModule = 0;
                }
                if (hwc2chwModule != 0) {
                    vkDestroyShaderModule(device, hwc2chwModule, nullptr);
                    hwc2chwModule = 0;
                }
                vkDeviceWaitIdle(device);
                vkDestroyDevice(device, nullptr);
                device = 0;
            }
            if (instance != 0) {
                vkDestroyInstance(instance, nullptr);
                instance = 0;
            }
            if (glslang_finalize_process != nullptr) {
                glslang_finalize_process();
            }
            if (!vulkan_library_name.empty()) {
                dl_manager->unloadLibrary(vulkan_library_name);
                vulkan_library_name.clear();
            }
            if (!glslang_library_name.empty()) {
                dl_manager->unloadLibrary(glslang_library_name);
                glslang_library_name.clear();
            }
            consecutiveVulkanFailures.store(0, std::memory_order_release);
            initVulkanStatus.store(InitVulkanStatusEnum::Ready, std::memory_order_release);
            return true;
        }

    private:
        inline static std::atomic<InitVulkanStatusEnum> initVulkanStatus = InitVulkanStatusEnum::Ready;
        inline static std::string lastVulkanErrorStr = "";
        inline static std::mutex VulkanMutex;
        inline static std::mutex errorMutex;
        inline static std::atomic<size_t> consecutiveVulkanFailures = 0;
        static constexpr size_t maxConsecutiveVulkanFailures = 3;

        inline static std::string vulkan_library_name = "";
        inline static std::string glslang_library_name = "";

        inline static VkInstance instance = 0;
        inline static VkPhysicalDevice physicalDevice = 0;
        inline static VkDevice device = 0;
        inline static VkQueue queue = 0;
        inline static uint32_t queueFamilyIndex = 0;
        inline static VkPhysicalDeviceMemoryProperties memoryProperties = {};

        inline static VkShaderModule hwc2chwModule = 0;
        inline static VkShaderModule chw2hwcModule = 0;
        inline static VkPipelineLayout pipelineLayout = 0;
        inline static VkDescriptorSetLayout descriptorSetLayout = 0;
        inline static VkDescriptorPool descriptorPool = 0;
        inline static VkDescriptorSet hwc2chwDescSet = 0;
        inline static VkDescriptorSet chw2hwcDescSet = 0;
        inline static VkPipeline hwc2chwPipeline = 0;
        inline static VkPipeline chw2hwcPipeline = 0;
        inline static VkCommandPool commandPool = 0;
    };

} // namespace whyb
