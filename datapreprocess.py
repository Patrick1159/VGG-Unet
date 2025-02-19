from PIL import Image
from tqdm import tqdm
import os
import time
import numpy as np
# import torch

rgb_to_class = { # rgb映射表 
    (64, 128, 64): 0,    # Animal
    (192, 0, 128): 1,    # Archway
    (0, 128, 192): 2,    # Bicyclist
    (0, 128, 64): 3,     # Bridge
    (128, 0, 0): 4,      # Building
    (64, 0, 128): 5,     # Car
    (64, 0, 192): 6,     # CartLuggagePram
    (192, 128, 64): 7,   # Child
    (192, 192, 128): 8,  # Column_Pole
    (64, 64, 128): 9,    # Fence
    (128, 0, 192): 10,   # LaneMkgsDriv
    (192, 0, 64): 11,    # LaneMkgsNonDriv
    (128, 128, 64): 12,  # Misc_Text
    (192, 0, 192): 13,   # MotorcycleScooter
    (128, 64, 64): 14,   # OtherMoving
    (64, 192, 128): 15,  # ParkingBlock
    (64, 64, 0): 16,     # Pedestrian
    (128, 64, 128): 17,  # Road
    (128, 128, 192): 18, # RoadShoulder
    (0, 0, 192): 19,     # Sidewalk
    (192, 128, 128): 20, # SignSymbol
    (128, 128, 128): 21, # Sky
    (64, 128, 192): 22,  # SUVPickupTruck
    (0, 0, 64): 23,      # TrafficCone
    (0, 64, 64): 24,     # TrafficLight
    (192, 64, 128): 25,  # Train
    (128, 128, 0): 26,   # Tree
    (192, 128, 192): 27, # Truck_Bus
    (64, 0, 64): 28,     # Tunnel
    (192, 192, 0): 29,   # VegetationMisc
    (0, 0, 0): 30,       # Void
    (64, 192, 0): 31     # Wall
}

# 11 类映射表
rgb_to_class_11 = {
    (128, 128, 128): 0,  # Sky
    (128, 0, 0): 1,      # Building
    (192, 192, 128): 2,  # Column_Pole
    (128, 64, 128): 3,   # Road
    (0, 0, 192): 4,      # Sidewalk
    (128, 128, 0): 5,    # Tree
    (64, 64, 128): 6,    # Fence
    (64, 0, 128): 7,     # Car
    (64, 64, 0): 8,      # Pedestrian
    (0, 128, 192): 9,    # Bicyclist
    (192, 128, 128): 10  # SignSymbol
}

# 将原始 32 类映射到 11 类
mapping_32_to_11 = {
    0: 8,    # Animal -> Pedestrian
    1: 1,    # Archway -> Building
    2: 9,    # Bicyclist -> Bicyclist
    3: 1,    # Bridge -> Building
    4: 1,    # Building -> Building
    5: 7,    # Car -> Car
    6: 7,    # CartLuggagePram -> Car
    7: 8,    # Child -> Pedestrian
    8: 2,    # Column_Pole -> Column_Pole
    9: 6,    # Fence -> Fence
    10: 3,   # LaneMkgsDriv -> Road
    11: 3,   # LaneMkgsNonDriv -> Road
    12: 1,   # Misc_Text -> Building
    13: 7,   # MotorcycleScooter -> Car
    14: 7,   # OtherMoving -> Car
    15: 4,   # ParkingBlock -> Sidewalk
    16: 8,   # Pedestrian -> Pedestrian
    17: 3,   # Road -> Road
    18: 3,   # RoadShoulder -> Road
    19: 4,   # Sidewalk -> Sidewalk
    20: 10,  # SignSymbol -> SignSymbol
    21: 0,   # Sky -> Sky
    22: 7,   # SUVPickupTruck -> Car
    23: 10,  # TrafficCone -> SignSymbol
    24: 10,  # TrafficLight -> SignSymbol
    25: 7,   # Train -> Car
    26: 5,   # Tree -> Tree
    27: 7,   # Truck_Bus -> Car
    28: 1,   # Tunnel -> Building
    29: 5,   # VegetationMisc -> Tree
    30: 1,   # Void -> Building
    31: 1    # Wall -> Building
}

# 最终的 11 类映射表
rgb_to_class_11_final = {rgb: mapping_32_to_11[class_id] for rgb, class_id in rgb_to_class.items()}

print(rgb_to_class_11_final)
# dataset父路径：
mask_dir = r'/home/sugon/zpy/U-NET/U-net/unet_data/trainannot'
mask_files =  sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

print(mask_files) # image_path list
print(f"Found {len(mask_files)} masks.")

# 将mask由rgb转化为类别,储存为npy


# 将 class_mapping 转换为 NumPy 字典映射数组（faster）
mapping_array = np.zeros((256, 256, 256), dtype=np.uint8)
for rgb, label in rgb_to_class_11_final.items():
    mapping_array[rgb] = label
# print("RGB 192 192 0 is type: ", mapping_array[192][192][0])

time_start = time.time()
for i in tqdm(range(len(mask_files))): 
    mask_path = r'/home/sugon/zpy/U-NET/U-net/unet_data/trainannot/' + mask_files[i]
    mask = Image.open(mask_path).convert('RGB') # RGB方式读取image
    mask_data = np.array(mask)
    mask_height, mask_width = mask_data.shape[0], mask_data.shape[1]
    mask_flat = mask_data.reshape(-1, 3)  # 展平为 (H*W, 3) (691200, 3)
    label_flat = mapping_array[mask_flat[:, 0], mask_flat[:, 1], mask_flat[:, 2]] # rgb -> type
    label_data = label_flat.reshape(mask_height, mask_width)  # 恢复为原始形状
    save_path = r'/home/sugon/zpy/U-NET/U-net/unet_data/train_numpy/' + mask_files[i][0: len(mask_files[i])- 4] # Seq05VD_f00000_L类型，去掉.png
    np.save(save_path, label_data)

    # label_data = torch.tensor(label_data)
    # print(label_data.unique())


time_end = time.time()
print("Time cost: ", time_end - time_start)


# save_path = r'd:/CIUS_Docs/U-net/unet_data/train_numpy/' + mask_files[i][0: len(mask_files[i])- 4] # Seq05VD_f00000_L类型，去掉.png
# d:/CIUS_Docs/U-net/unet_data/train_numpy/Seq05VD_f00000_L.npy格式
# 加载npy文件
def load_mapped_mask_numpy(load_path):
    """d:/CIUS_Docs/U-net/unet_data/train_numpy/Seq05VD_f00000_L.npy格式"""
    mask = np.load(load_path)
    return mask

print(load_mapped_mask_numpy("/home/sugon/zpy/U-NET/U-net/unet_data/train_numpy/0001TP_006750_L.npy"))