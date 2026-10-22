// residency_probe: what Metal residency does on this Mac (scratch tool, not part of Kapitsa).
//
//   residency_probe watch <mode> <GB> <seconds>
//     Allocates GB of shared buffers (256 MB each, filled on the CPU), then samples memory every 250 ms:
//     system wired, this process's footprint, compressed, graphics ledger. Modes:
//       none     no residency set, no GPU work
//       request  a residency set, requestResidency once at 1 s
//       beat     requestResidency at 1 s and every 500 ms after
//       cb       no set: one command buffer at 1 s binds every buffer (one 1-thread dispatch each)
//       queue    the set on the command queue, requestResidency at 1 s, one tiny command buffer at 2 s
//       endres   requestResidency at 1 s, endResidency at 4 s
//
//   residency_probe stall <mode> <GB> <seconds> [inflight=2] [touch=4]
//     Allocates GB as above (the "weights"), then commits small command buffers for <seconds>: each binds a scratch
//     buffer and `touch` of the weight buffers (rotating) and reads one word of each. Up to `inflight` are in
//     flight. Reports each command buffer's start latency: from its commit, and from when it could start (its commit
//     or the end of the one before it, whichever is later). Modes:
//       queue    the set holding every weight buffer on the command queue (Kapitsa today)
//       wired    the set off the queue, requestResidency every 500 ms (the candidate change)
//       none     no residency set
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static double host_ms() {
  static const double scale = [] {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return double(tb.numer) / double(tb.denom) / 1e6;
  }();
  return double(mach_absolute_time()) * scale;
}

struct Mem {
  double sys_wired, footprint, compressed, gfx, gfx_comp, gfx_nofp, ri_wired;
};
static Mem mem() {
  Mem m{};
  vm_statistics64_data_t v;
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n);
  m.sys_wired = double(v.wire_count) * double(vm_page_size) / 1e9;
  task_vm_info_data_t ti;
  mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &c) == KERN_SUCCESS) {
    m.footprint = ti.phys_footprint / 1e9;
    m.compressed = ti.compressed / 1e9;
    m.gfx = ti.ledger_tag_graphics_footprint / 1e9;
    m.gfx_comp = ti.ledger_tag_graphics_footprint_compressed / 1e9;
    m.gfx_nofp = ti.ledger_tag_graphics_nofootprint / 1e9;
  }
  rusage_info_v4 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*)&ri) == 0) m.ri_wired = ri.ri_wired_size / 1e9;
  return m;
}
static void print_mem(double t, const char* tag) {
  Mem m = mem();
  printf("%7.2f s  sys_wired %6.2f GB  footprint %6.2f  compressed %5.2f  gfx %6.2f  gfx_comp %5.2f  gfx_nofp %5.2f  "
         "ri_wired %5.2f  %s\n", t, m.sys_wired, m.footprint, m.compressed, m.gfx, m.gfx_comp, m.gfx_nofp, m.ri_wired, tag);
  fflush(stdout);
}

static const char* kSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void touch1(device const uint* w [[buffer(0)]], device uint* out [[buffer(1)]],\n"
    "                   constant uint& at [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "  if (i == 0) out[0] += w[at];\n"
    "}\n";

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: residency_probe watch|stall <mode> <GB> <seconds> [inflight] [touch]\n");
    return 2;
  }
  const std::string what = argv[1], mode = argv[2];
  const double gb = atof(argv[3]), secs = atof(argv[4]);
  const int inflight = argc > 5 ? atoi(argv[5]) : 2, touch = argc > 6 ? atoi(argv[6]) : 4;
  if (gb <= 0 || gb > 30 || secs <= 0 || secs > 300) {
    fprintf(stderr, "refusing: GB in (0, 30], seconds in (0, 300]\n");
    return 2;
  }
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueueWithMaxCommandBufferCount:512];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch1"] error:&err];
    const size_t bsz = size_t(256) << 20;
    const int nb = int(std::lround(gb * 1e9 / double(bsz)));
    NSMutableArray<id<MTLBuffer>>* bufs = [NSMutableArray new];
    const double t_alloc = host_ms();
    for (int i = 0; i < nb; ++i) {
      @autoreleasepool {
        id<MTLBuffer> b = [dev newBufferWithLength:bsz
                                           options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!b) { fprintf(stderr, "alloc failed at %d\n", i); return 1; }
        uint8_t* p = (uint8_t*)b.contents;
        for (size_t o = 0; o < bsz; o += 16384) p[o] = uint8_t(i + o / 16384);  // populate every page
        [bufs addObject:b];
      }
    }
    id<MTLBuffer> scratch = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    printf("allocated %d x 256 MB = %.2f GB in %.0f ms (device allocated %.2f GB)\n", nb, nb * bsz / 1e9,
           host_ms() - t_alloc, dev.currentAllocatedSize / 1e9);
    id<MTLResidencySet> rs = nil;
    auto make_set = [&] {
      MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
      d.label = @"probe";
      rs = [dev newResidencySetWithDescriptor:d error:&err];
      for (id<MTLBuffer> b in bufs) [rs addAllocation:b];
      [rs commit];
    };
    const double t0 = host_ms();
    auto T = [&] { return (host_ms() - t0) / 1000.0; };

    if (what == "watch") {
      bool requested = false, cb_done = false, ended = false, queue_cb = false;
      double next_beat = 1.0;
      if (mode != "none" && mode != "cb") make_set();
      if (mode == "queue") [q addResidencySet:rs];
      print_mem(T(), "start");
      while (T() < secs) {
        const double t = T();
        std::string tag;
        if (!requested && t >= 1.0 && rs) {
          const double a = host_ms();
          [rs requestResidency];
          tag = "requestResidency (" + std::to_string(int(host_ms() - a)) + " ms)";
          requested = true;
        }
        if (mode == "beat" && t >= next_beat) {
          [rs requestResidency];
          next_beat += 0.5;
        }
        if (mode == "cb" && !cb_done && t >= 1.0) {
          const double a = host_ms();
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:scratch offset:0 atIndex:1];
          uint32_t at = 0;
          [e setBytes:&at length:4 atIndex:2];
          for (id<MTLBuffer> b in bufs) {
            [e setBuffer:b offset:0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          }
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          tag = "cb binding all: " + std::to_string(int(host_ms() - a)) + " ms, commit->start " +
                std::to_string(cb.GPUStartTime * 1000 - a) + " ms";
          cb_done = true;
        }
        if (mode == "queue" && !queue_cb && t >= 2.0) {
          const double a = host_ms();
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:scratch offset:0 atIndex:0];
          [e setBuffer:scratch offset:0 atIndex:1];
          uint32_t at = 0;
          [e setBytes:&at length:4 atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          tag = "tiny cb on the queue: " + std::to_string(int(host_ms() - a)) + " ms, commit->start " +
                std::to_string(cb.GPUStartTime * 1000 - a) + " ms";
          queue_cb = true;
        }
        if (mode == "endres" && !ended && t >= 4.0) {
          [rs endResidency];
          tag = "endResidency";
          ended = true;
        }
        print_mem(t, tag.c_str());
        usleep(250000);
      }
      print_mem(T(), "end");
      return 0;
    }

    // ---- stall: phases alternating between the modes listed in `mode` (comma-separated), `secs` each ----
    // The first phase starts after the delay in PROBE_DELAY_S (default 0) once allocated (time to start a memory hog).
    std::vector<std::string> phases;
    for (size_t p = 0; p <= mode.size();) {
      size_t c = mode.find(',', p);
      if (c == std::string::npos) c = mode.size();
      if (c > p) phases.push_back(mode.substr(p, c - p));
      p = c + 1;
    }
    make_set();
    [rs requestResidency];
    dispatch_queue_t bq = dispatch_queue_create("probe.residency", DISPATCH_QUEUE_SERIAL);
    dispatch_source_t beat = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, bq);
    dispatch_source_set_timer(beat, DISPATCH_TIME_FOREVER, 0, 0);
    static std::atomic<int> beats{0};
    static std::atomic<double> beat_ms_max{0};
    dispatch_source_set_event_handler(beat, ^{
      const double a = host_ms();
      [rs requestResidency];
      const double d = host_ms() - a;
      if (d > beat_ms_max.load()) beat_ms_max.store(d);
      beats++;
    });
    dispatch_activate(beat);
    const double delay = getenv("PROBE_DELAY_S") ? atof(getenv("PROBE_DELAY_S")) : 0;
    printf("ready; first phase in %.0f s\n", delay);
    fflush(stdout);
    usleep(useconds_t(delay * 1e6));
    struct Rec { double commit, start, end; };
    auto report = [&](const char* name, std::vector<double> v) {
      std::sort(v.begin(), v.end());
      auto pct = [&](double p) { return v[std::min(v.size() - 1, size_t(p * v.size()))]; };
      int over[5] = {0, 0, 0, 0, 0};
      const double th[5] = {1, 5, 20, 50, 100};
      for (double x : v)
        for (int i = 0; i < 5; ++i) over[i] += x > th[i];
      printf("  %s (%zu): p50 %.3f p90 %.3f p99 %.3f p99.9 %.3f max %.1f ms; > 1/5/20/50/100 ms: %d/%d/%d/%d/%d\n", name,
             v.size(), pct(0.5), pct(0.9), pct(0.99), pct(0.999), v.back(), over[0], over[1], over[2], over[3], over[4]);
    };
    bool on_queue = false;
    int k = 0;
    for (const std::string& ph : phases) {
      if (ph == "queue") {
        dispatch_source_set_timer(beat, DISPATCH_TIME_FOREVER, 0, 0);
        if (!on_queue) [q addResidencySet:rs];
        on_queue = true;
      } else {
        if (on_queue) [q removeResidencySet:rs];
        on_queue = false;
        if (ph == "wired")
          dispatch_source_set_timer(beat, dispatch_time(DISPATCH_TIME_NOW, 0), 500 * NSEC_PER_MSEC, 50 * NSEC_PER_MSEC);
        else
          dispatch_source_set_timer(beat, DISPATCH_TIME_FOREVER, 0, 0);  // "none": the set only requested at the start
      }
      beats = 0;
      beat_ms_max = 0;
      std::vector<Rec> recs;
      recs.reserve(1 << 20);
      std::vector<id<MTLCommandBuffer>> live;
      std::vector<double> live_commit;
      const double p0 = T();
      print_mem(p0, ("phase " + ph).c_str());
      while (T() - p0 < secs) {
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:scratch offset:0 atIndex:1];
          for (int j = 0; j < touch; ++j) {
            id<MTLBuffer> b = bufs[size_t(k++ % nb)];
            uint32_t at = uint32_t((k * 7919) % (bsz / 4));
            [e setBuffer:b offset:0 atIndex:0];
            [e setBytes:&at length:4 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          }
          [e endEncoding];
          const double c = host_ms();
          [cb commit];
          live.push_back(cb);
          live_commit.push_back(c);
          while (int(live.size()) > inflight || (!live.empty() && T() - p0 >= secs)) {
            id<MTLCommandBuffer> f = live.front();
            [f waitUntilCompleted];
            recs.push_back({live_commit.front(), f.GPUStartTime * 1000, f.GPUEndTime * 1000});
            live.erase(live.begin());
            live_commit.erase(live_commit.begin());
          }
        }
      }
      std::vector<double> from_commit, from_ready;
      double prev_end = 0;
      for (auto& r : recs) {
        from_commit.push_back(r.start - r.commit);
        from_ready.push_back(std::max(0.0, r.start - std::max(r.commit, prev_end)));
        prev_end = std::max(prev_end, r.end);
      }
      printf("phase %s: %.2f GB in %d buffers, %zu command buffers in %.1f s (inflight %d, touch %d), %d heartbeat "
             "requests (longest %.2f ms)\n", ph.c_str(), nb * bsz / 1e9, nb, recs.size(), secs, inflight, touch,
             beats.load(), beat_ms_max.load());
      report("start - commit", from_commit);
      report("start - ready ", from_ready);
      fflush(stdout);
    }
    dispatch_source_cancel(beat);
    print_mem(T(), "stall end");
  }
  return 0;
}
