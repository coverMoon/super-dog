import torch

# 源文件路径
path = "/home/windnotebook/PROJECT/RoboCon/Dog/HIMLoco/legged_gym/logs/rough_black_dog/Dec21_18-41-40_/model_400.pt" # 请确保路径对

try:
    # 1. 加载模型
    checkpoint = torch.load(path, map_location='cpu')
    print(f"原记录 Iter: {checkpoint['iter']}")
    
    # 2. 强制修正(根据优化器步数推算的真实值)
    checkpoint['iter'] = 401
    print(f"修正后 Iter: {checkpoint['iter']}")
    
    # 3. 保存为新文件
    new_path = path.replace("400.pt", "400.pt")
    torch.save(checkpoint, new_path)
    print(f"已保存修复后的文件: {new_path}")

except Exception as e:
    print(f"修复失败: {e}")