import subprocess, re
import matplotlib.pyplot as plt

def run(n, run_o0):
    out = subprocess.run(["build/mp1_gpu", str(n), str(n), str(n), str(run_o0)], capture_output=True, text=True).stdout
    times = {}
    for name, t in re.findall(r"GPU, (gemm_gpu_\w+)\): ([\d.]+)ms", out):
        times[name] = float(t)
    return times

names = ["gemm_gpu_o2", "gemm_gpu_o3", "gemm_gpu_cublas"]

fig, ax = plt.subplots()

x = 0
for size in [100, 1000]:
    print(f"Running ablation for size {size}")
    times = run(size, 0)
    speedups = []
    for name in names:
        speedups.append(times["gemm_gpu_o1"] / times[name])
    positions = []
    for i in range(len(names)):
        positions.append(i + x * 0.35)
    ax.bar(positions, speedups, 0.35, label=str(size))
    x += 1
ax.set_xticks([0.175, 1.175, 2.175])
ax.set_xticklabels(names)
ax.legend()
ax.set_title("GPU speedup vs gemm_gpu_o1")

plt.savefig("gemm_gpu_ablation.png")

sizes = []
runtimes_o3 = []
for size in [100, 500, 1000, 5000, 10000]:
    print(f"Running scalability for size {size}")
    times = run(size, -1)
    sizes.append(size)
    runtimes_o3.append(times["gemm_gpu_o3"])

fig, ax = plt.subplots()
ax.plot(sizes, runtimes_o3, marker="o", label="gemm_gpu_o3")
ax.legend()
ax.set_title("gemm_gpu_o3 runtime vs matrix size")

plt.savefig("gemm_gpu_scalability.png")
plt.show()
