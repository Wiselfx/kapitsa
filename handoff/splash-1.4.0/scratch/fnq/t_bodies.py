import json
import sys

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab/bench/harness")
import bench  # noqa: E402

a = bench.parse_args(["--base-url", "x", "--model", "m", "--label", "t"])
bench.load_corpus()
out = open(sys.argv[1], "w")
for sz in (4000, 16000):
    ctx = bench.build_context(sz)
    msgs = bench.prefill_messages(ctx, "session 0123456789abcdef",
                                  "In one sentence, what is the first file in the repository context about?")
    out.write(json.dumps(bench.body_of(a, msgs, 256)) + "\n")
out.write(json.dumps(bench.body_of(a, bench.decode_messages(), 512)) + "\n")
