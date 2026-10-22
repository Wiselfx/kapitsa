import sys
p = sys.argv[1]
s = open(p).read()
start = s.index("  } else if (dfl_) {\n    // DFlash2, double-buffered:")
end = s.index("  } else if (dory_spec_ > 0) {")
new = '''  } else if (dfl_) {
    // DFlash2, double-buffered: each cycle's verify (8-row target forward with captures + on-GPU
    // acceptance) is committed, then right behind it the NEXT cycle's drafting half (ring commit of the
    // rows this acceptance keeps + drafting pass from the anchor it picks; both read the GPU-side state
    // the acceptance writes), and only then does the host wait and read the accepted count. The GPU never
    // waits for the host between cycles; the host encodes each verify while the drafting half runs.
    // With other requests waiting for the GPU a cycle commits no drafting half behind its verify: the request parks
    // (the ring commit of the kept rows alone) and starts the next cycle from the anchor once its turn comes back.
    // Structured output: once the reasoning is over (g_on) each verify runs under the grammar's masks. The host computes
    // every verify row's mask from the drafts (the drafting half has run by then) while the verify's layers run, and
    // signals the verify's sampler; the acceptance then reads the masked samples and distributions, so the reply is
    // masked plain decoding's. Reasoning first runs unmasked and ends its pass at </think> (that cycle's later tokens
    // were verified unmasked); a second pass goes on from it, masked. A masked request is not batched.
    DFlashSeq& d = *slot.dseq;
    const int K = DFlash2::verify_depth();
    int P = n;
    req_cycles = 0;
    bool go = handle(s.tokens()[n]);
    for (;;) {
      int m = 0, a_last = 0;
      const bool masked = gm && g_on;
      const int mrows = dfl_->tree() ? DFlash2::kB : K + 1;
      if (masked) {
        slot.decoding = false;  // (no batch waits for it)
        if (!mask_event_) mask_event_ = g.new_shared_event();
      }
      std::unique_ptr<Encoder> next_draft;  // committed drafting half of the coming cycle
      // The drafting half from the anchor P with nothing in flight (after the prefill, and when the request's turn
      // comes back after parking), with KV cache room for the first cycles (it grows between cycles below): nothing may
      // use the old buffers, so the deferred prompt work completes first
      auto start_cycles = [&] {
        if (P + 8 + 8 + DFlash2::kB + 3 > s.cap && s.cap < s.limit) {
          if (mtp_e) mtp_e->wait();
          grow_kv(std::min(s.limit, P + 8 + 8 + DFlash2::kB + 3), [] {});
        }
        if (P + 8 + DFlash2::kB + 2 < s.cap) {
          // host write of the drafter state: nothing in flight reads it (the prefill is done; its deferred
          // prompt commit, if still running, is host-driven; a request taking its turn has nothing in flight)
          dfl_->gpu_state_reset(P);
          next_draft = g.begin();
          dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P, sp);
          next_draft->commit();
        }
      };
      if (go) start_cycles();
      bool parked = false;     // the request parked after the last cycle (no drafting half committed)
      bool park_last = false;  // the last cycle committed no drafting half though there was room: it meant to park
      while (go) {
        if (parked) {
          parked = false;
          start_cycles();
        }
        if (!next_draft) {  // no room for another drafting pass
          finish = "length";
          res.context_full = true;
          break;
        }
        const auto draft = std::move(next_draft);  // this cycle's drafting half (the budget reads its GPU time)
        auto e2 = g.begin();
        uint64_t pending = 0;  // masked: the value the verify's sampler waits for
        if (masked) {
          s.mask_rows = mrows;
          s.mask_event = mask_event_;
          s.mask_value = pending = ++mask_value_;
        }
        dfl_->encode_cycle_verify(*e2, s, P, K, sp);
        s.mask_rows = 0;
        s.mask_event = nullptr;
        e2->commit();
        if (masked) {  // this cycle's masks, from its drafts, while the verify's layers run
          draft->wait();
          const double tm = now_ms();
          spec_masks(P, mrows, dfl_->tree());
          mask_ms += now_ms() - tm;
          mask_n += mrows;
          Gpu::signal_event_cpu(mask_event_, pending);
        }
        bool kv_wait = false;  // no KV room for the next cycle while other requests' caches hold the memory
        if (P + 8 + 8 + DFlash2::kB + 3 > s.cap && s.cap < s.limit) {
          // the KV cache grows here: the verify in flight (and the drafting half before it) complete first, so
          // this cycle's drafting half is not overlapped with it
          e2->wait();
          kv_wait = !reserve_kv(slot, std::min(s.limit, P + 8 + 8 + DFlash2::kB + 3), true) && active_ > 1;
        }
        // another request waits for the GPU, this one for memory (start_cycles waits for it), or it joins the batch:
        // no drafting half behind this verify, park after it
        const bool turn = contended();
        const bool join = !kv_wait && !masked && batch_wanted();  // (a request waiting for memory would leave at once)
        const bool park = turn || kv_wait || join;
        const bool room = P + 8 + 8 + DFlash2::kB + 2 < s.cap;  // the next anchor is at most P + 8
        park_last = park && room;
        if (!park && room) {
          next_draft = g.begin();
          dfl_->encode_cycle_draft_gpu(*next_draft, s, d, P + 8, sp);
          next_draft->commit();
        }
        e2->wait();
        // KFlash: the budget reads the cycle's outcome and its GPU time, the drafting half's (done before the verify
        // started) and the verify's, as ni_run measures it
        dfl_->observe(draft->gpu_ms() + e2->gpu_ms());
        const int a = dfl_->accepted();
        a_last = a;
        const int* tk = s.tokens();
        m = 0;  // tokens consumed before stopping = rows committed
        while (m < a + 1) {
          ++m;
          if (!handle(tk[P + m])) {
            go = false;
            break;
          }
        }
        ++spec_cycles_;
        ++req_cycles;
        spec_accepted_ += m;
        cycle_done(m);
        if (trace_path && dfl_->tree()) {  // the accepted path's verify rows (tree slots), in order
          const size_t V = size_t(model_->vocab());
          std::vector<float> rows(size_t(m) * V);
          for (int j = 0; j < m; ++j)
            memcpy(rows.data() + size_t(j) * V, q35_->logits() + size_t(s.pend_path.as<int>()[j]) * V, V * 4);
          trace_fn(rows.data(), m, tk + P + 1);
        } else if (trace_path) {
          trace_fn(q35_->logits(), m, tk + P + 1);
        }
        q35_->commit_rows(s, m);  // rows P .. P+m-1 are fed; tk[P+m] is the new anchor
        P += m;
        d.ctx = P;  // the committed drafting half (if any) puts these rows into the ring
        if (go && park) {
          // park: the rows this cycle kept go into the ring as the drafting half would have put them (from the GPU
          // state the acceptance wrote, reading the verify's captures before any other request's passes)
          auto ep = g.begin();
          dfl_->encode_commit(*ep, d, -1);
          ep->commit_and_wait();
          m = 0;
          park_last = false;
          if (join) {
            if (batch(P)) go = false;
            d.ctx = P;  // (the batch committed every row before P to the ring)
          } else if (turn) {
            yield_turn(res);
          }
          parked = true;
        }
      }
      // finish: the ring rows of the last cycle (unless the committed drafting half already wrote them),
      // then fold the tape into the recurrent state
      auto e3 = g.begin();
      if (!next_draft && m > 0) {
        if (park_last) {
          // the drafting half this cycle left out to park would have committed the acceptance's rows: so does this
          dfl_->encode_commit(*e3, d, -1);
          if (m < a_last + 1) d.valid_lo = std::max(d.valid_lo, P - m + a_last + 1 - DFlash2::kWin);
        } else {
          d.ctx = P - m;
          dfl_->encode_commit(*e3, d, m);
        }
      }
      // The committed drafting half wrote all a_last + 1 rows of the last acceptance (positions P - m ..
      // P - m + a_last) to the ring. When the host kept fewer (a stop token or the budget inside the
      // accepted chain), the rows past d.ctx overwrote the slots of the window's oldest positions, which a
      // later rewind (prefix reuse) would otherwise still treat as valid context.
      if (next_draft && m < a_last + 1) d.valid_lo = std::max(d.valid_lo, P - m + a_last + 1 - DFlash2::kWin);
      q35_->encode_flush(*e3, s);
      e3->commit_and_wait();
      fed_until = P;
      if (dfl_->tree() && dfl_->budget() && req_cycles > 0) {
        logi("%s", dfl_->budget()->summary().c_str());
        dfl_->budget()->save(KfBudget::kSaveEvery);
      }
      // the reasoning ended (</think>, tokens[P], handled): the reply goes on from it, masked
      if (!(g_break && finish.empty())) break;
      g_break = g_in_spec = false;
      go = true;
    }
'''
s = s[:start] + new + s[end:]
open(p, 'w').write(s)
print("patched")
