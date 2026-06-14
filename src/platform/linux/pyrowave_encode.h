/**
 * @file src/platform/linux/pyrowave_encode.h
 * @brief PyroWave codec encode session for Sunshine.
 *
 * Creates a Granite Vulkan context alongside Sunshine's existing Vulkan stack.
 * DMA-BUF frames are imported, converted RGB→YCbCr via a raw compute dispatch
 * on Granite's VkDevice, then encoded by the PyroWave wavelet codec.
 */
#pragma once

#include "src/platform/common.h"
#include "src/video.h"

#include <vector>

namespace pyrowave {

  /**
   * @brief Full encode session: Granite device + PyroWave encoder.
   *
   * Inherits encode_session_t so it can be returned directly from
   * make_encode_session() without an extra wrapper.
   */
  class session_t: public video::encode_session_t {
  public:
    session_t();
    ~session_t() override;

    /**
     * @brief Initialize Granite context, YCbCr images, bitstream buffers,
     *        and the PyroWave encoder.
     * @return true on success.
     */
    bool init(int width, int height,
              const video::config_t &config,
              const video::sunshine_colorspace_t &colorspace);

    /**
     * @brief Convert a captured DMA-BUF frame.
     *
     * Pipeline:
     *   1. Import DMA-BUF → Granite VkImage (raw Vulkan external memory)
     *   2. RGB→YCbCr compute dispatch (rgb2ycbcr_planes.comp)
     *   3. PyroWave encode → bitstream in host-visible buffer
     *   4. Fence wait + packetize bitstream into pending_bitstream
     *
     * Dummy frames (sequence == 0) produce a zero-filled IDR packet so that
     * validate_config() can complete without a real captured frame.
     */
    int convert(platf::img_t &img) override;

    void request_idr_frame() override;
    void request_normal_frame() override;
    void invalidate_ref_frames(int64_t first_frame, int64_t last_frame) override;

    /**
     * @brief Move out the bitstream produced by the last convert() call.
     * @return Empty vector if convert() hasn't been called yet.
     */
    std::vector<uint8_t> take_bitstream();

    int64_t last_frame_idx = 0;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl;
  };

  /**
   * @brief Create a lightweight stub encode_device_t used only to carry the
   *        colorspace through make_encode_session(); no GPU resources.
   */
  std::unique_ptr<platf::pyrowave_encode_device_t> make_encode_device();

  /**
   * @brief Create and initialize a full pyrowave::session_t.
   * @return nullptr on failure (Granite unavailable, PyroWave init error, etc.)
   */
  std::unique_ptr<session_t> make_session(int width, int height,
                                          const video::config_t &config,
                                          const video::sunshine_colorspace_t &colorspace);

  /**
   * @brief Quick check: can we create a Granite/Vulkan context at all?
   * Used by validate_encoder() before running the full encode probe.
   */
  bool validate();

}  // namespace pyrowave
