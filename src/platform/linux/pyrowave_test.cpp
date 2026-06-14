/**
 * @file src/platform/linux/pyrowave_test.cpp
 * @brief Standalone encode test: capture one real KMS frame, encode it with PyroWave,
 *        report bitstream size and timing. Run via: sunshine pyrowave-test
 */
#ifdef SUNSHINE_BUILD_PYROWAVE

// Granite headers must come first (set up VK_NO_PROTOTYPES via volk)
#include "context.hpp"

#include <chrono>
#include <cstdio>
#include <memory>

#include "pyrowave_encode.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/video.h"
#include "src/video_colorspace.h"

namespace pyrowave {

int run_capture_test() {
  std::printf("[pyrowave-test] Initialising platform...\n");
  std::fflush(stdout);

  auto platf_guard = platf::init();
  if (!platf_guard) {
    std::fprintf(stderr, "[pyrowave-test] Platform init failed (no capture source)\n");
    return 1;
  }

  video::config_t cfg {};
  cfg.width            = 1920;
  cfg.height           = 1080;
  cfg.framerate        = 60;
  cfg.bitrate          = 20000;  // 20 Mbps
  cfg.slicesPerFrame   = 1;
  cfg.numRefFrames     = 0;
  cfg.encoderCscMode   = 0;
  cfg.videoFormat      = 0;  // H.264 slot (unused for pyrowave; just needs a valid config)
  cfg.dynamicRange     = 0;
  cfg.chromaSamplingType = 0;  // 4:2:0

  const std::string display_name;  // empty = first monitor

  std::printf("[pyrowave-test] Opening display (vulkan mem type)...\n");
  std::fflush(stdout);

  auto disp = platf::display(platf::mem_type_e::vulkan, display_name, cfg);
  if (!disp) {
    std::fprintf(stderr, "[pyrowave-test] platf::display() returned null\n");
    return 1;
  }
  std::printf("[pyrowave-test] Display opened: %dx%d\n", disp->width, disp->height);
  std::fflush(stdout);

  // SDR Rec.601 (matches the default for software encoder)
  video::sunshine_colorspace_t cs {
    video::colorspace_e::rec601,
    true,   // full_range
    8,
  };

  std::printf("[pyrowave-test] Creating PyroWave session...\n");
  std::fflush(stdout);

  auto session = make_session(disp->width, disp->height, cfg, cs);
  if (!session) {
    std::fprintf(stderr, "[pyrowave-test] make_session() failed\n");
    return 1;
  }
  std::printf("[pyrowave-test] Session ready.\n");
  std::fflush(stdout);

  // --- Capture one real frame via the display's async capture() API ---
  std::shared_ptr<platf::img_t> captured;
  bool done   = false;
  bool cursor = true;

  auto push_cb = [&](std::shared_ptr<platf::img_t> &&img, bool frame_captured) -> bool {
    if (frame_captured && !done) {
      captured = std::move(img);
      done     = true;
      return false;  // tell capture() to stop
    }
    return !done;
  };

  auto pull_cb = [&](std::shared_ptr<platf::img_t> &out) -> bool {
    out = disp->alloc_img();
    return out != nullptr;
  };

  std::printf("[pyrowave-test] Capturing one frame...\n");
  std::fflush(stdout);

  auto t0 = std::chrono::steady_clock::now();

  // capture() blocks until push_cb returns false or an error/timeout occurs
  auto status = disp->capture(push_cb, pull_cb, &cursor);

  auto t1 = std::chrono::steady_clock::now();

  if (!captured) {
    std::fprintf(stderr, "[pyrowave-test] No frame captured (capture status %d)\n",
                 static_cast<int>(status));
    return 1;
  }

  auto capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::printf("[pyrowave-test] Frame captured in %lld ms\n", (long long)capture_ms);
  std::fflush(stdout);

  // --- Encode ---
  std::printf("[pyrowave-test] Encoding frame...\n");
  std::fflush(stdout);

  auto t2 = std::chrono::steady_clock::now();

  if (session->convert(*captured) < 0) {
    std::fprintf(stderr, "[pyrowave-test] session->convert() failed\n");
    return 1;
  }

  auto t3 = std::chrono::steady_clock::now();

  auto bs   = session->take_bitstream();
  auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  if (bs.empty()) {
    std::fprintf(stderr, "[pyrowave-test] Encoder returned empty bitstream\n");
    return 1;
  }

  std::printf("[pyrowave-test] SUCCESS\n");
  std::printf("[pyrowave-test]   encode time : %lld ms\n", (long long)enc_ms);
  std::printf("[pyrowave-test]   bitstream   : %zu bytes (%.1f kB)\n",
              bs.size(), bs.size() / 1024.0);
  std::printf("[pyrowave-test]   effective br: %.1f Mbps (at 60 fps)\n",
              bs.size() * 8 * 60.0 / 1e6);
  std::fflush(stdout);

  return 0;
}

}  // namespace pyrowave

#endif  // SUNSHINE_BUILD_PYROWAVE
