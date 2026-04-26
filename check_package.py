# 保存为 check_packages.py

packages = {
    'torch': 'GPU深度学习框架',
    'transformers': '加载模型',
    'peft': 'LoRA微调',
    'datasets': '加载数据',
    'PIL': '处理图像',
    'tqdm': '进度条',
}

print("检查Python包...")
print("=" * 50)

all_ok = True
for package, description in packages.items():
    try:
        __import__(package)
        print(f"✅ {package:<15} - {description}")
    except ImportError:
        print(f"❌ {package:<15} - {description}")
        all_ok = False

print("=" * 50)

if all_ok:
    print("✅ 所有包都已安装！")
else:
    print("❌ 有些包缺失，请运行：")
    print("pip install torch transformers peft datasets pillow tqdm")