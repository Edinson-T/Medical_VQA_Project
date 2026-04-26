# 保存为 test_dataset.py

print("测试数据集下载...")
print("(第一次会下载2GB数据，需要5-10分钟)")

try:
    from datasets import load_dataset
    
    print("正在加载VQA-RAD数据集...")
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
    
    print("✅ 数据集加载成功！")
    print(f"  训练集大小: {len(vqa_rad['train'])} 样本")
    print(f"  验证集大小: {len(vqa_rad['test'])} 样本")
    
    # 查看一个样本
    sample = vqa_rad['train'][0]
    print(f"\n示例样本:")
    print(f"  问题: {sample['question']}")
    print(f"  答案: {sample['answer']}")
    print(f"  图像: {type(sample['image'])}")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("原因可能是网络问题，请重试")