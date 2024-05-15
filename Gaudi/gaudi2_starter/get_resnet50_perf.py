import re
import numpy as np
import sys
console_log_path = sys.argv[1]
skip_step = int(sys.argv[2])
seconds_perf_step_pattern = re.compile(
    rf"TimeHistory:\s(\d+\.\d+)\sseconds,\s(\d+\.\d+) examples/second.*steps\s(\d+)"
)
with open(console_log_path, "r") as f:
    log_text = f.read()
    step_perf_matches = seconds_perf_step_pattern.findall(log_text)
partial_perf = []
for seconds, perf, step in step_perf_matches:
    if int(step) <= skip_step:
        continue
    partial_perf.append(float(perf))
throughput = (
    round(1 / np.mean(1 / np.asarray(partial_perf)), 7)
)
print(f"Throughput DPS: {throughput}")
