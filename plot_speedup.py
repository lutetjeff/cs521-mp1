# LLM was used to do the regexing
# i refuse to do regex, regex is a language for machines

import subprocess, re
import matplotlib.pyplot as plt


def run(n, run_o0):
    out = subprocess.run(["build/mp1_cpu_o0", str(n), str(n), str(n), str(run_o0)], capture_output=True, text=True).stdout
    times = {}
    for name, t in re.findall(r"CPU,(gemm_cpu_\w+)\): ([\d.]+)ms", out):
        times[name] = float(t)
    return times

def run_o4(n, run_o0):
    out = subprocess.run(["build/mp1_cpu", str(n), str(n), str(n), str(run_o0)], capture_output=True, text=True).stdout
    times = {}
    for name, t in re.findall(r"CPU,(gemm_cpu_\w+)\): ([\d.]+)ms", out):
        if name == "gemm_cpu_o3":
            times["gemm_cpu_o4"] = float(t)
    return times

names = ["gemm_cpu_o1", "gemm_cpu_o2", "gemm_cpu_o3", "gemm_cpu_o4"]

fig, ax = plt.subplots()

x = 0
for size in [100, 1000]:
    print(f"Running ablation for size {size}")
    times = run(size, 1)
    times_o4 = run_o4(size, 0)
    times = times | times_o4
    speedups = []
    for name in names:
        speedups.append(times["gemm_cpu_o0"] / times[name])
    positions = []
    for i in range(len(names)):
        positions.append(i + x * 0.35)
    ax.bar(positions, speedups, 0.35, label=str(size))
    x += 1
ax.set_xticks([0.175, 1.175, 2.175, 3.175])
ax.set_xticklabels(names)
ax.legend()
ax.set_title("Speedup vs gemm_cpu_o0")

plt.savefig("gemm_cpu_ablation.png")

sizes = []
runtimes = []
for size in [100, 500, 1000, 5000, 10000]:
    print(f"Running scalability for size {size}")
    times = run_o4(size, 0)
    sizes.append(size)
    runtimes.append(times["gemm_cpu_o4"])

fig, ax = plt.subplots()
ax.plot(sizes, runtimes, marker="o")
ax.set_title("gemm_cpu_o4 runtime vs matrix size")

plt.savefig("gemm_cpu_scalability.png")
plt.show()
