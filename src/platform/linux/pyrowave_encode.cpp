/**
 * @file src/platform/linux/pyrowave_encode.cpp
 * @brief PyroWave encode session implementation.
 *
 * All GPU work — DMA-BUF import, RGB→YCbCr conversion, and PyroWave encode —
 * is recorded into a SINGLE Granite command buffer per frame and submitted once.
 * Raw Vulkan API calls for DMA-BUF import and the RGB→YCbCr compute dispatch
 * are recorded via cmd->get_command_buffer() (the underlying VkCommandBuffer),
 * so Granite's resource tracking remains consistent.
 */
#include "pyrowave_encode.h"

// Granite headers must come first — they set up VK_NO_PROTOTYPES via volk
// before any Vulkan types are declared.
#include "context.hpp"
#include "device.hpp"
#include "buffer.hpp"
#include "image.hpp"

#include <array>
#include <cstring>
#include <drm_fourcc.h>

// PyroWave headers
#include "pyrowave_encoder.hpp"
#include "pyrowave_common.hpp"

// Sunshine headers
#include "graphics.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/video_colorspace.h"

using namespace std::literals;

// SPIR-V C-array generated at build time from rgb2ycbcr_planes.comp
static const std::vector<uint32_t> rgb2ycbcr_planes_spv
#include "rgb2ycbcr_planes.spv.inc"
  ;

namespace pyrowave {

// ---------------------------------------------------------------------------
// Push constants — must mirror rgb2ycbcr_planes.comp layout exactly.
// ---------------------------------------------------------------------------
struct PushConstants {
  std::array<float, 4> color_vec_y;
  std::array<float, 4> color_vec_u;
  std::array<float, 4> color_vec_v;
  std::array<float, 2> range_y;
  std::array<float, 2> range_uv;
  std::array<int32_t, 2> src_offset;
  std::array<int32_t, 2> src_size;
  std::array<int32_t, 2> dst_offset;
  std::array<int32_t, 2> dst_size;
  std::array<int32_t, 2> dst_full_size;
  std::array<int32_t, 2> cursor_pos;
  std::array<int32_t, 2> cursor_size;
  int32_t y_invert;
  int32_t chroma_420;
};

#define VK_CHECK_LOG(expr, ...) \
  do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
      BOOST_LOG(error) << "[pyrowave] " #expr " failed: " << _r; \
      return __VA_ARGS__; \
    } \
  } while (0)

// ---------------------------------------------------------------------------
// session_t::impl_t
// ---------------------------------------------------------------------------
struct session_t::impl_t {
  // Granite
  Vulkan::Context ctx;
  Vulkan::Device dev;

  // YCbCr planes (Granite-owned)
  Vulkan::ImageHandle yuv_images[3];
  PyroWave::ViewBuffers yuv_views = {};
  PyroWave::ChromaSubsampling chroma = PyroWave::ChromaSubsampling::Chroma420;
  int frame_width = 0, frame_height = 0;

  // YUV plane storage format. 8-bit SDR uses R8_UNORM; 10-bit HDR uses
  // R16_UNORM so the full dynamic range survives RGB->YCbCr. PyroWave's wavelet
  // transform operates in normalized float, so the codec itself is format-agnostic.
  VkFormat yuv_format = VK_FORMAT_R8_UNORM;
  bool is_hdr = false;

  // Bitstream buffers
  Vulkan::BufferHandle meta_dev, meta_host;
  Vulkan::BufferHandle bs_dev, bs_host;
  size_t target_bytes_per_frame = 0;

  // PyroWave encoder
  PyroWave::Encoder encoder;

  // Raw Vulkan state (on Granite's VkDevice — no separate VkQueue)
  struct RawPipeline {
    // These handles belong to Granite's VkDevice; we don't destroy the device.
    VkDevice dev = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdPropertiesKHR getMemoryFdProperties = nullptr;

    // RGB→YCbCr compute pipeline
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    // Storage views for YCbCr output (VkImageView wrappers for Granite images)
    VkImageView y_view = VK_NULL_HANDLE;
    VkImageView cb_view = VK_NULL_HANDLE;
    VkImageView cr_view = VK_NULL_HANDLE;
    bool views_created = false;
    bool target_initialized = false;
    bool descriptors_dirty = true;

    // DMA-BUF source
    struct SrcImage {
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory mem = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      int width = 0, height = 0;
    } src;
    uint64_t src_sequence = 0;

    // Cursor
    struct CursorImage {
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory mem = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      bool needs_transition = false;
    } cursor;
    unsigned long cursor_serial = 0;
  } raw;

  // Colorspace push-constant cache
  PushConstants push = {};

  // Encoded output for the most recent convert()
  std::vector<uint8_t> pending_bitstream;

  // ------------------------------------------------------------------
  bool init(int width, int height,
            const video::config_t &config,
            const video::sunshine_colorspace_t &colorspace) {
    frame_width = width;
    frame_height = height;
    chroma = (::config::video.pyrowave.chroma == 1)
               ? PyroWave::ChromaSubsampling::Chroma444
               : PyroWave::ChromaSubsampling::Chroma420;
    BOOST_LOG(info) << "[pyrowave] chroma subsampling: "
                    << (chroma == PyroWave::ChromaSubsampling::Chroma444 ? "4:4:4"sv : "4:2:0"sv);

    // 10-bit HDR uses R16_UNORM planes; SDR stays on R8_UNORM.
    is_hdr = video::colorspace_is_hdr(colorspace);
    yuv_format = (colorspace.bit_depth == 10) ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
    BOOST_LOG(info) << "[pyrowave] bit depth: " << colorspace.bit_depth
                    << "-bit ("sv << (is_hdr ? "HDR"sv : "SDR"sv) << "), plane format "sv
                    << (yuv_format == VK_FORMAT_R16_UNORM ? "R16_UNORM"sv : "R8_UNORM"sv);

    // 1. Granite context + device
    if (!Vulkan::Context::init_loader(nullptr)) {
      BOOST_LOG(error) << "[pyrowave] Vulkan loader init failed"sv;
      return false;
    }
    if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0,
                                      Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT)) {
      BOOST_LOG(error) << "[pyrowave] Granite context init failed"sv;
      return false;
    }
    dev.set_context(ctx);

    // Stash raw handles (no extra device creation — same VkDevice as Granite)
    raw.dev = dev.get_device();
    raw.phys_dev = dev.get_physical_device();
    raw.getMemoryFdProperties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
      vkGetDeviceProcAddr(raw.dev, "vkGetMemoryFdPropertiesKHR"));

    // 2. YCbCr images
    Vulkan::ImageCreateInfo img_info =
      Vulkan::ImageCreateInfo::immutable_2d_image(width, height, yuv_format);
    img_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    yuv_images[0] = dev.create_image(img_info);
    dev.set_name(*yuv_images[0], "pyrowave_Y");

    int chroma_w = (chroma == PyroWave::ChromaSubsampling::Chroma420) ? width >> 1 : width;
    int chroma_h = (chroma == PyroWave::ChromaSubsampling::Chroma420) ? height >> 1 : height;
    img_info.width = chroma_w;
    img_info.height = chroma_h;
    yuv_images[1] = dev.create_image(img_info);
    dev.set_name(*yuv_images[1], "pyrowave_Cb");
    yuv_images[2] = dev.create_image(img_info);
    dev.set_name(*yuv_images[2], "pyrowave_Cr");

    for (int i = 0; i < 3; i++)
      yuv_views.planes[i] = &yuv_images[i]->get_view();

    // 3. PyroWave encoder (must come before buffer creation — get_meta_required_size() requires init)
    if (!encoder.init(&dev, width, height, chroma)) {
      BOOST_LOG(error) << "[pyrowave] PyroWave::Encoder::init() failed"sv;
      return false;
    }

    // Signal colorspace to the decoder via the bitstream sequence header.
    {
      PyroWave::Encoder::ColorMetadata meta = {};
      bool bt2020 = colorspace.colorspace == video::colorspace_e::bt2020 ||
                    colorspace.colorspace == video::colorspace_e::bt2020sdr;
      meta.color_primaries   = bt2020 ? PyroWave::COLOR_PRIMARIES_BT2020 : PyroWave::COLOR_PRIMARIES_BT709;
      meta.ycbcr_transform   = bt2020 ? PyroWave::YCBCR_TRANSFORM_BT2020 : PyroWave::YCBCR_TRANSFORM_BT709;
      // colorspace_e::bt2020 is the PQ HDR variant; bt2020sdr keeps the SDR transfer.
      meta.transfer_function = (colorspace.colorspace == video::colorspace_e::bt2020)
                                 ? PyroWave::TRANSFER_FUNCTION_PQ
                                 : PyroWave::TRANSFER_FUNCTION_BT709;
      meta.ycbcr_range       = colorspace.full_range ? PyroWave::YCBCR_RANGE_FULL : PyroWave::YCBCR_RANGE_LIMITED;
      meta.chroma_siting     = PyroWave::CHROMA_SITING_CENTER;
      encoder.set_color_metadata(meta);
    }

    // 4. Bitstream buffers (sized using encoder metadata from init above)
    target_bytes_per_frame = static_cast<size_t>(
      static_cast<uint64_t>(config.bitrate) * 1000 / 8 / config.framerate + 4096);

    Vulkan::BufferCreateInfo buf_info = {};
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    buf_info.size = encoder.get_meta_required_size();
    buf_info.domain = Vulkan::BufferDomain::Device;
    meta_dev = dev.create_buffer(buf_info);
    buf_info.domain = Vulkan::BufferDomain::CachedHost;
    meta_host = dev.create_buffer(buf_info);

    buf_info.size = target_bytes_per_frame + 2 * encoder.get_meta_required_size();
    buf_info.domain = Vulkan::BufferDomain::Device;
    bs_dev = dev.create_buffer(buf_info);
    buf_info.domain = Vulkan::BufferDomain::CachedHost;
    bs_host = dev.create_buffer(buf_info);

    // 5. Raw Vulkan compute pipeline for RGB→YCbCr
    if (!create_compute_pipeline())
      return false;

    // 6. Colorspace push constants
    apply_colorspace(colorspace);
    return true;
  }

  // ------------------------------------------------------------------
  void apply_colorspace(const video::sunshine_colorspace_t &colorspace) {
    auto *colors = video::color_vectors_from_colorspace(colorspace, true);
    if (!colors) return;
    memcpy(push.color_vec_y.data(), colors->color_vec_y, sizeof(push.color_vec_y));
    memcpy(push.color_vec_u.data(), colors->color_vec_u, sizeof(push.color_vec_u));
    memcpy(push.color_vec_v.data(), colors->color_vec_v, sizeof(push.color_vec_v));
    memcpy(push.range_y.data(),  colors->range_y,  sizeof(push.range_y));
    memcpy(push.range_uv.data(), colors->range_uv, sizeof(push.range_uv));
    push.chroma_420 = (chroma == PyroWave::ChromaSubsampling::Chroma420) ? 1 : 0;

    BOOST_LOG(debug) << "[pyrowave] colorspace=" << (int) colorspace.colorspace
                     << " full_range=" << colorspace.full_range
                     << " bit_depth=" << colorspace.bit_depth;
  }

  // ------------------------------------------------------------------
  // Entry point: import DMA-BUF, RGB→YCbCr, PyroWave encode — all in one
  // Granite command buffer.
  // ------------------------------------------------------------------
  int convert_frame(platf::img_t &img) {
    auto &descriptor = static_cast<egl::img_descriptor_t &>(img);

    auto cmd = dev.request_command_buffer();
    VkCommandBuffer raw_cmd = cmd->get_command_buffer();

    if (descriptor.sequence == 0) {
      // Dummy frame: clear YCbCr planes to 0 (black).
      clear_yuv_planes(raw_cmd);
    } else {
      // Import DMA-BUF when it changes.
      if (descriptor.sequence != raw.src_sequence) {
        if (!import_dmabuf(descriptor.sd))
          return -1;
        raw.src_sequence = descriptor.sequence;
        BOOST_LOG(verbose) << "[pyrowave] imported DMA-BUF seq="sv << descriptor.sequence
                           << " "sv << raw.src.width << "x"sv << raw.src.height;
      }

      // Lazy-create storage views for YCbCr outputs.
      if (!raw.views_created) {
        if (!create_yuv_storage_views())
          return -1;
        raw.views_created = true;
        raw.descriptors_dirty = true;
      }

      // Update cursor texture if needed.
      if (!descriptor.buffer.empty() && descriptor.serial != raw.cursor_serial) {
        raw.cursor_serial = descriptor.serial;
        if (!create_cursor_image(descriptor.src_w, descriptor.src_h,
                                 descriptor.buffer.data()))
          return -1;
        raw.descriptors_dirty = true;
      }

      if (raw.descriptors_dirty) {
        update_descriptors();
        raw.descriptors_dirty = false;
      }

      // Record RGB→YCbCr dispatch.
      record_rgb2ycbcr(raw_cmd, descriptor);
    }

    // Transition YCbCr planes to SHADER_READ_ONLY for PyroWave.
    transition_yuv_for_encode(raw_cmd);

    // PyroWave encode (recorded on the same Granite command buffer).
    PyroWave::Encoder::BitstreamBuffers buffers = {};
    buffers.meta.buffer      = meta_dev.get();
    buffers.meta.size        = meta_dev->get_create_info().size;
    buffers.bitstream.buffer = bs_dev.get();
    buffers.bitstream.size   = bs_dev->get_create_info().size;
    buffers.target_size      = target_bytes_per_frame;

    encoder.encode(*cmd, yuv_views, buffers);

    // Copy bitstream to host-visible buffers.
    cmd->copy_buffer(*bs_host, *bs_dev);
    cmd->copy_buffer(*meta_host, *meta_dev);
    cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

    // Single submit + wait.
    Vulkan::Fence fence;
    dev.submit(cmd, &fence);
    dev.next_frame_context();
    fence->wait();

    // Packetize bitstream.
    const void *mapped_meta = dev.map_host_buffer(*meta_host, Vulkan::MEMORY_ACCESS_READ_BIT);
    const void *mapped_bs   = dev.map_host_buffer(*bs_host,   Vulkan::MEMORY_ACCESS_READ_BIT);

    // Packet boundary: max 8 KiB per network packet (matches sandbox usage).
    constexpr size_t packet_boundary = 8 * 1024;
    size_t num_packets = encoder.compute_num_packets(mapped_meta, packet_boundary);
    std::vector<PyroWave::Encoder::Packet> packets(num_packets);

    std::vector<uint8_t> reordered(bs_host->get_create_info().size);
    size_t out_packets = encoder.packetize(
      packets.data(), packet_boundary,
      reordered.data(), reordered.size(),
      mapped_meta, mapped_bs);

    if (out_packets != num_packets) {
      BOOST_LOG(error) << "[pyrowave] packetize: expected " << num_packets
                       << " packets, got " << out_packets;
      return -1;
    }

    // Concatenate all reordered packets into pending_bitstream.
    size_t total = 0;
    for (auto &p : packets) total += p.size;
    pending_bitstream.resize(total);
    size_t offset = 0;
    for (auto &p : packets) {
      memcpy(pending_bitstream.data() + offset, reordered.data() + p.offset, p.size);
      offset += p.size;
    }
    return 0;
  }

  // ------------------------------------------------------------------
  // Clear all three YCbCr images to zero (used for dummy frames).
  // ------------------------------------------------------------------
  void clear_yuv_planes(VkCommandBuffer raw_cmd) {
    std::array<VkImageMemoryBarrier, 3> barriers = {};
    for (int i = 0; i < 3; i++) {
      auto &b = barriers[i];
      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.srcAccessMask = 0;
      b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      b.image = yuv_images[i]->get_image();
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    vkCmdPipelineBarrier(raw_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 3, barriers.data());

    VkClearColorValue clear = {};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (int i = 0; i < 3; i++) {
      vkCmdClearColorImage(raw_cmd, yuv_images[i]->get_image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    }
    raw.target_initialized = true;
  }

  // ------------------------------------------------------------------
  // Transition YCbCr planes to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  // ready for PyroWave's sampler reads.
  // ------------------------------------------------------------------
  void transition_yuv_for_encode(VkCommandBuffer raw_cmd) {
    std::array<VkImageMemoryBarrier, 3> barriers = {};
    for (int i = 0; i < 3; i++) {
      auto &b = barriers[i];
      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b.oldLayout = raw.target_initialized ? VK_IMAGE_LAYOUT_GENERAL
                                           : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      b.srcAccessMask = raw.target_initialized ? VK_ACCESS_SHADER_WRITE_BIT
                                               : VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      b.image = yuv_images[i]->get_image();
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    VkPipelineStageFlags src_stage = raw.target_initialized
                                       ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                       : VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(raw_cmd, src_stage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 3, barriers.data());
  }

  // ------------------------------------------------------------------
  // Record RGB→YCbCr compute dispatch on the raw VkCommandBuffer.
  // ------------------------------------------------------------------
  void record_rgb2ycbcr(VkCommandBuffer raw_cmd,
                        const egl::img_descriptor_t &descriptor) {
    // Transition source DMA-BUF image into SHADER_READ_ONLY.
    VkImageMemoryBarrier src_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    src_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    src_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    src_barrier.srcAccessMask = 0;
    src_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    src_barrier.image = raw.src.image;
    src_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(raw_cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &src_barrier);

    // Transition cursor if just uploaded.
    if (raw.cursor.needs_transition) {
      VkImageMemoryBarrier cursor_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      cursor_barrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
      cursor_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      cursor_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      cursor_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      cursor_barrier.image = raw.cursor.image;
      cursor_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      cursor_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      cursor_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      vkCmdPipelineBarrier(raw_cmd,
                           VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &cursor_barrier);
      raw.cursor.needs_transition = false;
    }

    // Transition YCbCr output images to GENERAL for storage writes.
    std::array<VkImageMemoryBarrier, 3> dst_barriers = {};
    for (int i = 0; i < 3; i++) {
      auto &b = dst_barriers[i];
      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b.oldLayout = raw.target_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
      b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      b.srcAccessMask = raw.target_initialized ? VK_ACCESS_SHADER_READ_BIT : 0;
      b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      b.image = yuv_images[i]->get_image();
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    vkCmdPipelineBarrier(raw_cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 3, dst_barriers.data());

    // Fill per-frame push constants.
    push.src_offset = {0, 0};
    push.src_size   = {raw.src.width, raw.src.height};
    push.dst_offset = {0, 0};
    push.dst_size   = {frame_width, frame_height};
    push.dst_full_size = {frame_width, frame_height};
    push.y_invert   = descriptor.y_invert ? 1 : 0;
    if (!descriptor.buffer.empty()) {
      float sx = static_cast<float>(frame_width)  / raw.src.width;
      float sy = static_cast<float>(frame_height) / raw.src.height;
      push.cursor_pos  = {static_cast<int32_t>(descriptor.x  * sx),
                          static_cast<int32_t>(descriptor.y  * sy)};
      push.cursor_size = {static_cast<int32_t>(descriptor.src_w * sx),
                          static_cast<int32_t>(descriptor.src_h * sy)};
    } else {
      push.cursor_size = {0, 0};
    }

    vkCmdBindPipeline(raw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, raw.pipeline);
    vkCmdBindDescriptorSets(raw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            raw.pipeline_layout, 0, 1, &raw.desc_set, 0, nullptr);
    vkCmdPushConstants(raw_cmd, raw.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushConstants), &push);

    uint32_t gx = (frame_width  + 15) / 16;
    uint32_t gy = (frame_height + 15) / 16;
    vkCmdDispatch(raw_cmd, gx, gy, 1);

    raw.target_initialized = true;
  }

  // ------------------------------------------------------------------
  // Map a DRM fourcc to the Vulkan format whose component layout matches the
  // buffer in memory, so that sampling .rgb in the shader yields logical R,G,B.
  static VkFormat vk_format_from_fourcc(uint32_t fourcc) {
    switch (fourcc) {
      case DRM_FORMAT_XRGB8888:
      case DRM_FORMAT_ARGB8888:
        return VK_FORMAT_B8G8R8A8_UNORM;
      case DRM_FORMAT_XBGR8888:
      case DRM_FORMAT_ABGR8888:
        return VK_FORMAT_R8G8B8A8_UNORM;
      case DRM_FORMAT_XRGB2101010:
      case DRM_FORMAT_ARGB2101010:
        return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
      case DRM_FORMAT_XBGR2101010:
      case DRM_FORMAT_ABGR2101010:
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
      default:
        return VK_FORMAT_UNDEFINED;
    }
  }

  bool import_dmabuf(const egl::surface_descriptor_t &sd) {
    destroy_src_image();

    VkFormat src_format = vk_format_from_fourcc(sd.fourcc);
    {
      uint32_t f = sd.fourcc;
      char cc[5] = { char(f & 0xff), char((f >> 8) & 0xff),
                     char((f >> 16) & 0xff), char((f >> 24) & 0xff), 0 };
      BOOST_LOG(debug) << "[pyrowave] DMA-BUF fourcc='" << cc << "' (0x"
                       << std::hex << f << std::dec << ") modifier=0x"
                       << std::hex << sd.modifier << std::dec
                       << " -> VkFormat " << (int) src_format;
    }
    if (src_format == VK_FORMAT_UNDEFINED) {
      BOOST_LOG(error) << "[pyrowave] Unsupported DMA-BUF fourcc 0x"
                       << std::hex << sd.fourcc << std::dec;
      return false;
    }

    int fd = dup(sd.fds[0]);
    if (fd < 0) return false;

    VkMemoryFdPropertiesKHR fd_props = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (raw.getMemoryFdProperties) {
      raw.getMemoryFdProperties(raw.dev,
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                fd, &fd_props);
    }

    VkExternalMemoryImageCreateInfo ext_ci = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    std::array<VkSubresourceLayout, 4> drm_layouts = {};
    VkImageDrmFormatModifierExplicitCreateInfoEXT drm_ci = {
      VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT
    };

    VkImageTiling tiling;
    if (sd.modifier != DRM_FORMAT_MOD_INVALID) {
      int planes = 0;
      for (int i = 0; i < 4 && sd.fds[i] >= 0; i++) planes++;
      for (int i = 0; i < planes; i++) {
        drm_layouts[i].offset   = sd.offsets[i];
        drm_layouts[i].rowPitch = sd.pitches[i];
      }
      drm_ci.drmFormatModifier          = sd.modifier;
      drm_ci.drmFormatModifierPlaneCount = planes;
      drm_ci.pPlaneLayouts               = drm_layouts.data();
      ext_ci.pNext = &drm_ci;
      tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    } else {
      tiling = VK_IMAGE_TILING_LINEAR;
    }

    VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    img_ci.pNext       = &ext_ci;
    img_ci.imageType   = VK_IMAGE_TYPE_2D;
    img_ci.format      = src_format;
    img_ci.extent      = {(uint32_t) sd.width, (uint32_t) sd.height, 1};
    img_ci.mipLevels   = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling      = tiling;
    img_ci.usage       = VK_IMAGE_USAGE_SAMPLED_BIT;
    img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_LOG(vkCreateImage(raw.dev, &img_ci, nullptr, &raw.src.image), false);

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(raw.dev, raw.src.image, &mem_req);

    VkImportMemoryFdInfoKHR import_fd = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import_fd.fd = fd;

    VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.pNext = &import_fd;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = find_memory_type(
      fd_props.memoryTypeBits ? fd_props.memoryTypeBits : mem_req.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK_LOG(vkAllocateMemory(raw.dev, &alloc_info, nullptr, &raw.src.mem), false);
    vkBindImageMemory(raw.dev, raw.src.image, raw.src.mem, 0);

    VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_ci.image = raw.src.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format   = src_format;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK_LOG(vkCreateImageView(raw.dev, &view_ci, nullptr, &raw.src.view), false);

    raw.src.width  = sd.width;
    raw.src.height = sd.height;
    raw.descriptors_dirty = true;
    return true;
  }

  bool create_yuv_storage_views() {
    VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format   = yuv_format;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    view_ci.image = yuv_images[0]->get_image();
    VK_CHECK_LOG(vkCreateImageView(raw.dev, &view_ci, nullptr, &raw.y_view), false);
    view_ci.image = yuv_images[1]->get_image();
    VK_CHECK_LOG(vkCreateImageView(raw.dev, &view_ci, nullptr, &raw.cb_view), false);
    view_ci.image = yuv_images[2]->get_image();
    VK_CHECK_LOG(vkCreateImageView(raw.dev, &view_ci, nullptr, &raw.cr_view), false);
    return true;
  }

  bool create_cursor_image(int w, int h, const uint8_t *pixels) {
    destroy_cursor_image();

    VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    img_ci.imageType   = VK_IMAGE_TYPE_2D;
    img_ci.format      = VK_FORMAT_B8G8R8A8_UNORM;
    img_ci.extent      = {(uint32_t) w, (uint32_t) h, 1};
    img_ci.mipLevels   = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples     = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling      = VK_IMAGE_TILING_LINEAR;
    img_ci.usage       = VK_IMAGE_USAGE_SAMPLED_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    VK_CHECK_LOG(vkCreateImage(raw.dev, &img_ci, nullptr, &raw.cursor.image), false);

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(raw.dev, raw.cursor.image, &mem_req);
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize  = mem_req.size;
    alloc.memoryTypeIndex = find_memory_type(
      mem_req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_LOG(vkAllocateMemory(raw.dev, &alloc, nullptr, &raw.cursor.mem), false);
    VK_CHECK_LOG(vkBindImageMemory(raw.dev, raw.cursor.image, raw.cursor.mem, 0), false);

    if (pixels) {
      void *mapped;
      VK_CHECK_LOG(vkMapMemory(raw.dev, raw.cursor.mem, 0, VK_WHOLE_SIZE, 0, &mapped), false);
      VkImageSubresource subres = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
      VkSubresourceLayout layout;
      vkGetImageSubresourceLayout(raw.dev, raw.cursor.image, &subres, &layout);
      for (int y = 0; y < h; y++) {
        memcpy(static_cast<uint8_t *>(mapped) + layout.offset + y * layout.rowPitch,
               pixels + y * w * 4, w * 4);
      }
      vkUnmapMemory(raw.dev, raw.cursor.mem);
    }

    VkImageViewCreateInfo view_ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_ci.image    = raw.cursor.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format   = VK_FORMAT_B8G8R8A8_UNORM;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK_LOG(vkCreateImageView(raw.dev, &view_ci, nullptr, &raw.cursor.view), false);
    raw.cursor.needs_transition = true;
    return true;
  }

  void destroy_cursor_image() {
    if (raw.cursor.view)  { vkDestroyImageView(raw.dev, raw.cursor.view, nullptr); raw.cursor.view = VK_NULL_HANDLE; }
    if (raw.cursor.image) { vkDestroyImage(raw.dev, raw.cursor.image, nullptr);    raw.cursor.image = VK_NULL_HANDLE; }
    if (raw.cursor.mem)   { vkFreeMemory(raw.dev, raw.cursor.mem, nullptr);        raw.cursor.mem = VK_NULL_HANDLE; }
  }

  void destroy_src_image() {
    if (raw.src.view)  { vkDestroyImageView(raw.dev, raw.src.view, nullptr); raw.src.view = VK_NULL_HANDLE; }
    if (raw.src.image) { vkDestroyImage(raw.dev, raw.src.image, nullptr);    raw.src.image = VK_NULL_HANDLE; }
    if (raw.src.mem)   { vkFreeMemory(raw.dev, raw.src.mem, nullptr);        raw.src.mem = VK_NULL_HANDLE; }
  }

  void update_descriptors() {
    VkDescriptorImageInfo src_info    = {raw.sampler, raw.src.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo y_info      = {VK_NULL_HANDLE, raw.y_view,   VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cb_info     = {VK_NULL_HANDLE, raw.cb_view,  VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cr_info     = {VK_NULL_HANDLE, raw.cr_view,  VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cursor_info = {raw.sampler, raw.cursor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    std::array<VkWriteDescriptorSet, 5> writes = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raw.desc_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src_info,    nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raw.desc_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         &y_info,      nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raw.desc_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         &cb_info,     nullptr, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raw.desc_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         &cr_info,     nullptr, nullptr};
    writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raw.desc_set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cursor_info, nullptr, nullptr};
    vkUpdateDescriptorSets(raw.dev, writes.size(), writes.data(), 0, nullptr);
  }

  bool create_compute_pipeline() {
    VkShaderModuleCreateInfo shader_ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_ci.codeSize = rgb2ycbcr_planes_spv.size() * sizeof(uint32_t);
    shader_ci.pCode    = rgb2ycbcr_planes_spv.data();
    VK_CHECK_LOG(vkCreateShaderModule(raw.dev, &shader_ci, nullptr, &raw.shader_module), false);

    // Bindings: 0=src, 1=Y, 2=Cb, 3=Cr, 4=cursor
    std::array<VkDescriptorSetLayoutBinding, 5> bindings = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo ds_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ds_ci.bindingCount = bindings.size();
    ds_ci.pBindings    = bindings.data();
    VK_CHECK_LOG(vkCreateDescriptorSetLayout(raw.dev, &ds_ci, nullptr, &raw.ds_layout), false);

    VkPushConstantRange pc_range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pl_ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &raw.ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc_range;
    VK_CHECK_LOG(vkCreatePipelineLayout(raw.dev, &pl_ci, nullptr, &raw.pipeline_layout), false);

    VkComputePipelineCreateInfo comp_ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    comp_ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    comp_ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    comp_ci.stage.module = raw.shader_module;
    comp_ci.stage.pName  = "main";
    comp_ci.layout       = raw.pipeline_layout;
    VK_CHECK_LOG(vkCreateComputePipelines(raw.dev, VK_NULL_HANDLE, 1, &comp_ci, nullptr, &raw.pipeline), false);

    std::array<VkDescriptorPoolSize, 2> pool_sizes = {{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
    }};
    VkDescriptorPoolCreateInfo pool_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = pool_sizes.size();
    pool_ci.pPoolSizes    = pool_sizes.data();
    VK_CHECK_LOG(vkCreateDescriptorPool(raw.dev, &pool_ci, nullptr, &raw.desc_pool), false);

    VkDescriptorSetAllocateInfo ds_alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds_alloc.descriptorPool     = raw.desc_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts        = &raw.ds_layout;
    VK_CHECK_LOG(vkAllocateDescriptorSets(raw.dev, &ds_alloc, &raw.desc_set), false);

    VkSamplerCreateInfo sampler_ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_ci.magFilter    = VK_FILTER_LINEAR;
    sampler_ci.minFilter    = VK_FILTER_LINEAR;
    sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK_LOG(vkCreateSampler(raw.dev, &sampler_ci, nullptr, &raw.sampler), false);

    // Dummy 1×1 cursor so the descriptor set is always valid.
    return create_cursor_image(1, 1, nullptr);
  }

  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(raw.phys_dev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      if ((type_bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & props) == props)
        return i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      if (type_bits & (1u << i)) return i;
    }
    return 0;
  }

  void cleanup() {
    if (!raw.dev) return;
    vkDeviceWaitIdle(raw.dev);
    destroy_src_image();
    destroy_cursor_image();
    if (raw.y_view)          vkDestroyImageView(raw.dev, raw.y_view,          nullptr);
    if (raw.cb_view)         vkDestroyImageView(raw.dev, raw.cb_view,         nullptr);
    if (raw.cr_view)         vkDestroyImageView(raw.dev, raw.cr_view,         nullptr);
    if (raw.sampler)         vkDestroySampler(raw.dev, raw.sampler,           nullptr);
    if (raw.desc_pool)       vkDestroyDescriptorPool(raw.dev, raw.desc_pool,  nullptr);
    if (raw.pipeline)        vkDestroyPipeline(raw.dev, raw.pipeline,         nullptr);
    if (raw.pipeline_layout) vkDestroyPipelineLayout(raw.dev, raw.pipeline_layout, nullptr);
    if (raw.ds_layout)       vkDestroyDescriptorSetLayout(raw.dev, raw.ds_layout, nullptr);
    if (raw.shader_module)   vkDestroyShaderModule(raw.dev, raw.shader_module, nullptr);
    raw.dev = VK_NULL_HANDLE;
  }
};

// ---------------------------------------------------------------------------
// session_t
// ---------------------------------------------------------------------------
session_t::session_t(): impl(std::make_unique<impl_t>()) {}
session_t::~session_t() { if (impl) impl->cleanup(); }

bool session_t::init(int w, int h,
                     const video::config_t &cfg,
                     const video::sunshine_colorspace_t &cs) {
  return impl->init(w, h, cfg, cs);
}

int session_t::convert(platf::img_t &img) {
  return impl->convert_frame(img);
}

void session_t::request_idr_frame()                          {}
void session_t::request_normal_frame()                       {}
void session_t::invalidate_ref_frames(int64_t, int64_t)      {}

std::vector<uint8_t> session_t::take_bitstream() {
  return std::move(impl->pending_bitstream);
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
std::unique_ptr<platf::pyrowave_encode_device_t> make_encode_device() {
  return std::make_unique<platf::pyrowave_encode_device_t>();
}

std::unique_ptr<session_t> make_session(int width, int height,
                                        const video::config_t &config,
                                        const video::sunshine_colorspace_t &colorspace) {
  auto s = std::make_unique<session_t>();
  if (!s->init(width, height, config, colorspace)) {
    BOOST_LOG(error) << "[pyrowave] Session init failed"sv;
    return nullptr;
  }
  return s;
}

bool validate() {
  if (!Vulkan::Context::init_loader(nullptr)) return false;
  Vulkan::Context ctx;
  return ctx.init_instance_and_device(nullptr, 0, nullptr, 0, 0);
}

}  // namespace pyrowave
