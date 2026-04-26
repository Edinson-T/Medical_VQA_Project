# Diagnostic script: Check what fields Qwen2-VL processor returns
# test_processor_fields.py

from transformers import AutoProcessor
from datasets import load_dataset
from PIL import Image

print("=" * 80)
print("🔍 Diagnosing Qwen2-VL Processor Return Fields")
print("=" * 80)

# Load processor
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("✓ Processor loaded successfully")

# Load a sample
print("\nLoading VQA-RAD dataset...")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
sample = vqa_rad['train'][0]
print("✓ Dataset loaded successfully")

# Display sample content
print(f"\nSample content:")
print(f"  Question: {sample['question']}")
print(f"  Answer: {sample['answer']}")
print(f"  Image: {sample['image']}")

# Process the sample
print("\n" + "=" * 80)
print("Now processing the sample with the processor...")
print("=" * 80)

image = sample['image']
question = sample['question']
answer = sample['answer']

# Method 1: Simple processing (recommended)
print("\n【Method 1】Simple processing:")
try:
    inputs = processor(
        text=f"Question: {question}\nAnswer:",
        images=image,
        return_tensors="pt",
    )
    
    print("✓ Processing successful!")
    print(f"\nFields returned by processor:")
    for key in inputs.keys():
        print(f"  - {key}: {inputs[key].shape if hasattr(inputs[key], 'shape') else type(inputs[key])}")
    
except Exception as e:
    print(f"❌ Processing failed: {e}")

# Method 2: Chat template processing
print("\n" + "-" * 80)
print("【Method 2】Using chat template:")
try:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Generated text: {text[:100]}...")
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    
    print("✓ Processing successful!")
    print(f"\nFields returned by processor:")
    for key in inputs.keys():
        val = inputs[key]
        if hasattr(val, 'shape'):
            print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  - {key}: {type(val)}")
    
except Exception as e:
    print(f"❌ Processing failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnosis complete!")
print("=" * 80)