import torch

# 直接设置使用GPU 2
try:
    torch.cuda.set_device(2)
    print(f"Set device to GPU 2")
except:
    print("Cannot set device to GPU 2")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 简单GPU测试
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print(f"GPU test passed! Result shape: {z.shape}")