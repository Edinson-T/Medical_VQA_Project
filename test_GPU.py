# 保存为 test_gpu.py

import torch
import sys

print("=" * 60)
print("GPU检查工具")
print("=" * 60)

# 检查1：PyTorch是否看到GPU
print("\n1️⃣ 检查GPU可用性...")
if torch.cuda.is_available():
    print("✅ GPU可用！")
else:
    print("❌ GPU不可用")
    print("原因可能是：")
    print("  - CUDA未正确安装")
    print("  - 显卡驱动过旧")
    print("  - PyTorch版本不对")
    sys.exit(1)

# 检查2：GPU型号和显存
print("\n2️⃣ GPU信息...")
device = torch.device("cuda")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")

gpu_props = torch.cuda.get_device_properties(0)
vram_gb = gpu_props.total_memory / 1e9
print(f"显存总量: {vram_gb:.1f}GB")

if vram_gb < 8:
    print(f"⚠️ 警告：显存可能不足")
elif vram_gb >= 8:
    print(f"✅ 显存足够")

# 检查3：当前显存占用
print("\n3️⃣ 当前显存占用...")
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved = torch.cuda.memory_reserved(0) / 1e9
print(f"已分配: {allocated:.2f}GB")
print(f"已预留: {reserved:.2f}GB")
print(f"可用: {vram_gb - reserved:.2f}GB")

# 检查4：简单的GPU计算测试
print("\n4️⃣ 测试GPU计算...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU计算正常")
except Exception as e:
    print(f"❌ GPU计算失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有检查通过！可以开始训练")
print("=" * 60)