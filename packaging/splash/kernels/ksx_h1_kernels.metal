
// ---- KSX H1: the kernels patch 0002 dispatches (packaging/splash/README.md) ----
// gen_kernels.py puts Splash's repack helpers above this. ksx_moe_expert_load_c below is Splash's moe_expert_load
// (Apache License 2.0) with its staging read through system-coherent loads; the rest is Kapitsa's.
constexpr constant thread_scope ksx_sys = static_cast<thread_scope>(3);
#define KSX_SYS_FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, ksx_sys)
#define KSX_H1_OUT 5
#define KSX_H1_IN 4
// ctl: [0] the last step the GPU reached, [1] the last step the CPU released, [2] the last step whose spin timed out.
struct KsxH1Params {
  uint step;
  uint spin_limit;
  uint out_words[KSX_H1_OUT];
  uint in_words[KSX_H1_IN];
};
#define KSX_OUT_ARGS(k) device const uint *or##k [[buffer(1 + k)]], volatile coherent(system) device uint *ow##k [[buffer(6 + k)]]
#define KSX_IN_ARGS(k, Q) Q device const uint *is##k [[buffer(11 + k)]], device uint *id##k [[buffer(15 + k)]]
#define KSX_OUT(k) for (uint i = t; i < p.out_words[k]; i += n) ow##k[i] = or##k[i]
#define KSX_IN(k) for (uint i = t; i < p.in_words[k]; i += n) id##k[i] = is##k[i]

kernel void ksx_h1_step(volatile coherent(system) device uint *ctl [[buffer(0)]],
                        KSX_OUT_ARGS(0), KSX_OUT_ARGS(1), KSX_OUT_ARGS(2), KSX_OUT_ARGS(3), KSX_OUT_ARGS(4),
                        KSX_IN_ARGS(0, volatile coherent(system)), KSX_IN_ARGS(1, volatile coherent(system)),
                        KSX_IN_ARGS(2, volatile coherent(system)), KSX_IN_ARGS(3, volatile coherent(system)),
                        constant KsxH1Params &p [[buffer(19)]],
                        uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  KSX_OUT(0); KSX_OUT(1); KSX_OUT(2); KSX_OUT(3); KSX_OUT(4);
  KSX_SYS_FENCE();
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    KSX_SYS_FENCE();
    ctl[0] = p.step;
    KSX_SYS_FENCE();
    bool released = false;
    for (uint k = 0; k < p.spin_limit; ++k) {
      KSX_SYS_FENCE();
      if (as_type<int>(ctl[1] - p.step) >= 0) {
        released = true;
        break;
      }
    }
    if (!released) ctl[2] = p.step;
    KSX_SYS_FENCE();
  }
  threadgroup_barrier(mem_flags::mem_device);
  KSX_SYS_FENCE();
  KSX_IN(0); KSX_IN(1); KSX_IN(2); KSX_IN(3);
}

// SPLASH_KSX_H1=2 (HostStepGpu::InBufferEvent): ksx_h1_step's copy-out, flag and bounded spin, without the copy-in,
// which ksx_h1_copy does after the next buffer's event wait. A spin that reaches its bound counts into ctl[3] and
// ends: the event wait then holds the GPU until the runner signals.
kernel void ksx_h1_spin(volatile coherent(system) device uint *ctl [[buffer(0)]],
                        KSX_OUT_ARGS(0), KSX_OUT_ARGS(1), KSX_OUT_ARGS(2), KSX_OUT_ARGS(3), KSX_OUT_ARGS(4),
                        constant KsxH1Params &p [[buffer(19)]],
                        uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  KSX_OUT(0); KSX_OUT(1); KSX_OUT(2); KSX_OUT(3); KSX_OUT(4);
  KSX_SYS_FENCE();
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    KSX_SYS_FENCE();
    ctl[0] = p.step;
    KSX_SYS_FENCE();
    bool released = false;
    for (uint k = 0; k < p.spin_limit; ++k) {
      KSX_SYS_FENCE();
      if (as_type<int>(ctl[1] - p.step) >= 0) {
        released = true;
        break;
      }
    }
    if (!released) ctl[3] = ctl[3] + 1;
    KSX_SYS_FENCE();
  }
}

// SPLASH_KSX_H1=3 (HostStepGpu::InBufferGate): ksx_h1_step, and on a spin that reaches its bound, the gate. The
// backend dispatches every direct dispatch after the command's first such step from a slot of `gate` (4 words each,
// from word KSX_GATE_SLOTS), and binds at 21.. the indirect grids that dispatches after this one read (3 words each).
// A step that gave up writes its number into ctl[2], skips its copy-in, zeroes the slots after its own (q.slot) and
// those grids, saving the grids first into gate[0..3 * KSX_GATE_GRIDS): the rest of the command does nothing, and
// the backend restores the grids (ksx_h1_restore) and encodes it again from this dispatch.
#define KSX_GATE_GRIDS 8
#define KSX_GATE_SLOTS 32
struct KsxGateParams {
  uint slot;
  uint slots;
  uint grids;
};
#define KSX_GRID_ARGS volatile coherent(device) device uint *g0 [[buffer(21)]], \
  volatile coherent(device) device uint *g1 [[buffer(22)]], volatile coherent(device) device uint *g2 [[buffer(23)]], \
  volatile coherent(device) device uint *g3 [[buffer(24)]], volatile coherent(device) device uint *g4 [[buffer(25)]], \
  volatile coherent(device) device uint *g5 [[buffer(26)]], volatile coherent(device) device uint *g6 [[buffer(27)]], \
  volatile coherent(device) device uint *g7 [[buffer(28)]]
#define KSX_GRID(k) (k == 0 ? g0 : k == 1 ? g1 : k == 2 ? g2 : k == 3 ? g3 : k == 4 ? g4 : k == 5 ? g5 : k == 6 ? g6 : g7)

kernel void ksx_h1_gate_step(volatile coherent(system) device uint *ctl [[buffer(0)]],
                             KSX_OUT_ARGS(0), KSX_OUT_ARGS(1), KSX_OUT_ARGS(2), KSX_OUT_ARGS(3), KSX_OUT_ARGS(4),
                             KSX_IN_ARGS(0, volatile coherent(system)), KSX_IN_ARGS(1, volatile coherent(system)),
                             KSX_IN_ARGS(2, volatile coherent(system)), KSX_IN_ARGS(3, volatile coherent(system)),
                             constant KsxH1Params &p [[buffer(19)]],
                             volatile coherent(device) device uint *gate [[buffer(20)]], KSX_GRID_ARGS,
                             constant KsxGateParams &q [[buffer(29)]],
                             uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup uint releasedShared;
  KSX_OUT(0); KSX_OUT(1); KSX_OUT(2); KSX_OUT(3); KSX_OUT(4);
  KSX_SYS_FENCE();
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    KSX_SYS_FENCE();
    ctl[0] = p.step;
    KSX_SYS_FENCE();
    bool released = false;
    for (uint k = 0; k < p.spin_limit; ++k) {
      KSX_SYS_FENCE();
      if (as_type<int>(ctl[1] - p.step) >= 0) {
        released = true;
        break;
      }
    }
    if (!released) ctl[2] = p.step;
    KSX_SYS_FENCE();
    releasedShared = released ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  KSX_SYS_FENCE();
  if (releasedShared) {
    KSX_IN(0); KSX_IN(1); KSX_IN(2); KSX_IN(3);
    return;
  }
  for (uint i = q.slot + 1 + t; i < q.slots; i += n) {
    gate[KSX_GATE_SLOTS + 4 * i] = 0;
    gate[KSX_GATE_SLOTS + 4 * i + 1] = 0;
    gate[KSX_GATE_SLOTS + 4 * i + 2] = 0;
  }
  if (t < q.grids && t < KSX_GATE_GRIDS) {
    volatile coherent(device) device uint *g = KSX_GRID(t);
    for (uint w = 0; w < 3; ++w) {
      gate[3 * t + w] = g[w];
      g[w] = 0;
    }
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

// The replay's first dispatch: puts back the grids that a gate step that gave up saved and zeroed.
kernel void ksx_h1_restore(volatile coherent(device) device uint *gate [[buffer(20)]], KSX_GRID_ARGS,
                           constant KsxGateParams &q [[buffer(29)]], uint t [[thread_index_in_threadgroup]]) {
  if (t < q.grids && t < KSX_GATE_GRIDS) {
    volatile coherent(device) device uint *g = KSX_GRID(t);
    for (uint w = 0; w < 3; ++w) g[w] = gate[3 * t + w];
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

kernel void ksx_h1_copy(KSX_IN_ARGS(0, ), KSX_IN_ARGS(1, ), KSX_IN_ARGS(2, ), KSX_IN_ARGS(3, ),
                        constant KsxH1Params &p [[buffer(19)]],
                        uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  KSX_IN(0); KSX_IN(1); KSX_IN(2); KSX_IN(3);
}

// Splash's moe_expert_load (gguf_repack.metal), staging read through system-coherent loads.
kernel void ksx_moe_expert_load_c(KSXC uchar *staging [[buffer(0)]], device uchar *arena [[buffer(1)]],
                                  device const uint *slots [[buffer(2)]], device const ulong *sources [[buffer(3)]],
                                  constant MoeExpertLoadParams &p [[buffer(4)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint index [[thread_index_in_threadgroup]]) {
  constant MoeExpertLoadProjection &q = p.projections[group.z];
  const uint G = q.columns / 32, t = group.x * MOE_EXPERT_LOAD_THREADS + index;
  if (t >= q.rows * G) return;
  const uint n = t / G, g = t % G;
  const ulong slot = slots[group.y];
  gguf_repack_group(staging + sources[group.y * 3 + group.z] + ulong(n) * q.row_bytes, q.format, G, n, g,
                    arena + q.planes[0] + slot * q.slot_bytes[0], arena + q.planes[1] + slot * q.slot_bytes[1],
                    arena + q.planes[2] + slot * q.slot_bytes[2]);
}
