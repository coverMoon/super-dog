import torch

# 你的文件路径
path = "/home/windnotebook/PROJECT/RoboCon/Dog/HIMLoco/legged_gym/logs/rough_black_dog/Dec21_18-41-40_/model_400.pt"

data = torch.load(path, map_location='cpu')
print(f"文件名: 400.pt")
print(f"内部记录的迭代次数 (iter): {data.get('iter', 'Key not found')}")
print(f"内部记录的迭代次数 (iteration): {data.get('iteration', 'Key not found')}")

# 还可以看一下优化器的步数，这通常是绝对真实的
if 'optimizer_state_dict' in data:
    # 打印优化器内部状态的一个参数来看看步数
    try:
        opt_state = data['optimizer_state_dict']['state']
        # 取第一个参数的 step
        first_param_key = list(opt_state.keys())[0]
        print(f"优化器记录的 Step: {opt_state[first_param_key]['step']}")
    except:
        print("无法解析优化器 Step")