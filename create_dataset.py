import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ================= 1. 本地绝对路径配置 =================
SRC_NORMAL_IMGS = Path("/home/doctor/tzh/datasets/normal_images")
SRC_NORMAL_COMP_MASKS = Path("/home/doctor/tzh/datasets/normal_component_masks")

SRC_ABNORMAL_IMGS = Path("/home/doctor/tzh/datasets/abnormal_images")
SRC_ABNORMAL_COMP_MASKS = Path("/home/doctor/tzh/datasets/abnormal_component_masks")
SRC_ABNORMAL_DEFECT_MASKS = Path("/home/doctor/tzh/datasets/abnormal_defect_masks")

WORKSPACE_DIR = Path("/home/doctor/tzh/MSD-Inpainting-V2/data/CCM-Defect")

# ================= 2. 参数与标准 Token 映射 =================
IMG_SIZE = 512
TRAIN_RATIO = 45 / 140  # 约 1:2 的比例
RANDOM_SEED = 42

# 原始字典
DEFECT_CLASSES = {
    1: "flexible_printed_circuit_crack",
    2: "foreign_particle",
    3: "end_face_scratch",
    4: "lens_scratch"
}
PART_CLASSES = {
    1: "flexible_printed_circuit", 
    2: "end_face", 
    3: "lens"                       
}

# 自动转换为尖括号 Token 格式
PART_TOKENS = {k: f"<{v}>" for k, v in PART_CLASSES.items()}
DEFECT_TOKENS = {k: f"<{v}>" for k, v in DEFECT_CLASSES.items()}

# ================= 3. 目录初始化 =================
# 每次运行前，建议先清理一下可能残留的旧 metadata，避免追加混乱
DIRS = {
    "normal": WORKSPACE_DIR / "normal_components",
    "train": WORKSPACE_DIR / "defect_train_concept",
    "test": WORKSPACE_DIR / "defect_test_holdout"
}

for d in DIRS.values():
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "component_masks").mkdir(parents=True, exist_ok=True)
    if d != DIRS["normal"]:
        (d / "defect_masks").mkdir(parents=True, exist_ok=True)
    
    # 清空旧的 jsonl
    jsonl_path = d / "metadata.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

def append_metadata(
    jsonl_path,
    image_name,
    prompt,
    component_mask_name=None,
    defect_mask_name=None,
    object_token=None,
    defect_token=None,
):
    record = {"image_path": f"images/{image_name}", "prompt": prompt}
    if component_mask_name is not None:
        record["component_mask_path"] = f"component_masks/{component_mask_name}"
    if defect_mask_name is not None:
        record["defect_mask_path"] = f"defect_masks/{defect_mask_name}"
    if object_token is not None:
        record["object_token"] = object_token
    if defect_token is not None:
        record["defect_token"] = defect_token
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ================= 4. 核心裁切逻辑 =================
def crop_and_resize(image, specific_comp_mask, defect_mask=None):
    """根据单个组件的二值化 mask 计算外接正方形并裁切，同时返回组件掩码与缺陷掩码"""
    coords = cv2.findNonZero(specific_comp_mask)
    if coords is None:
        return None, None, None
    x, y, w, h = cv2.boundingRect(coords)
    
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(image.shape[1], x1 + side), min(image.shape[0], y1 + side)
    
    cropped_img = image[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    cropped_comp_mask = specific_comp_mask[y1:y2, x1:x2]
    resized_comp_mask = cv2.resize(cropped_comp_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    
    resized_defect = None
    if defect_mask is not None:
        cropped_defect = defect_mask[y1:y2, x1:x2]
        # 缺陷掩码必须使用 INTER_NEAREST 保证边缘锐利，不产生过渡像素
        resized_defect = cv2.resize(cropped_defect, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
    return resized_img, resized_comp_mask, resized_defect

# ================= 5. 流水线执行与统计 =================
def process_normals():
    print("\n--- 正在处理正常样本 (Normal Images) ---")
    
    # 统计字典
    stats_normal = {token: 0 for token in PART_TOKENS.values()}
    total_processed = 0
    
    # 兼容多种格式
    img_files = list(SRC_NORMAL_IMGS.glob("*.png")) + list(SRC_NORMAL_IMGS.glob("*.jpg"))
    
    for img_path in img_files:
        image = cv2.imread(str(img_path))
        # 假设掩码后缀始终为 png
        comp_mask_path = SRC_NORMAL_COMP_MASKS / (img_path.stem + ".png")
        
        if not comp_mask_path.exists():
            continue
            
        comp_mask_full = cv2.imread(str(comp_mask_path), cv2.IMREAD_GRAYSCALE)
        
        for pixel_val, comp_token in PART_TOKENS.items():
            # 提取单个组件的纯净二值掩码 (0和255)
            binary_comp_mask = (comp_mask_full == pixel_val).astype(np.uint8) * 255
            
            # 如果该图像中不存在该组件，则跳过
            if cv2.countNonZero(binary_comp_mask) == 0:
                continue
                
            resized_img, resized_comp_mask, _ = crop_and_resize(image, binary_comp_mask)
            if resized_img is not None and resized_comp_mask is not None:
                img_name = f"normal_{img_path.stem}_{comp_token.strip('<>')}.png"
                component_mask_name = f"normal_{img_path.stem}_{comp_token.strip('<>')}_component_mask.png"
                cv2.imwrite(str(DIRS["normal"] / "images" / img_name), resized_img)
                cv2.imwrite(str(DIRS["normal"] / "component_masks" / component_mask_name), resized_comp_mask)
                
                prompt = f"a photo of {comp_token}"
                append_metadata(
                    DIRS["normal"] / "metadata.jsonl",
                    img_name,
                    prompt,
                    component_mask_name=component_mask_name,
                    object_token=comp_token,
                )
                
                stats_normal[comp_token] += 1
                total_processed += 1

    # 打印统计信息
    print(">>> 正常组件图裁切完成！统计如下:")
    for token, count in stats_normal.items():
        print(f"  - {token:25s}: {count} 张")
    print(f"  -> 总计生成正常组件图: {total_processed} 张\n")


def process_abnormals():
    print("--- 正在处理异常样本与数据集划分 (Abnormal Images) ---")
    
    # 1. 第一遍扫描：根据缺陷掩码的像素值，对图像进行编目分类
    defect_catalog = {token: [] for token in DEFECT_TOKENS.values()}
    
    img_files = list(SRC_ABNORMAL_IMGS.glob("*.png")) + list(SRC_ABNORMAL_IMGS.glob("*.jpg"))
    
    for img_path in img_files:
        defect_mask_path = SRC_ABNORMAL_DEFECT_MASKS / (img_path.stem + ".png")
        if not defect_mask_path.exists():
            continue
            
        defect_mask_full = cv2.imread(str(defect_mask_path), cv2.IMREAD_GRAYSCALE)
        present_pixels = np.unique(defect_mask_full) 
        
        for pixel_val in present_pixels:
            if pixel_val in DEFECT_TOKENS:
                defect_catalog[DEFECT_TOKENS[pixel_val]].append(img_path)

    # 统计嵌套字典：stats[defect_type][dataset_type] = count
    stats_abnormal = defaultdict(lambda: {"train": 0, "test": 0})

    # 2. 第二遍处理：按比例划分并执行物理裁切
    for defect_token, file_paths in defect_catalog.items():
        file_paths = sorted(file_paths, key=lambda p: str(p))
        random.shuffle(file_paths)
        split_idx = int(len(file_paths) * TRAIN_RATIO)
        splits = {"train": file_paths[:split_idx], "test": file_paths[split_idx:]}
        
        for dataset_type, subset_paths in splits.items():
            target_dir = DIRS[dataset_type]
            
            for img_path in subset_paths:
                image = cv2.imread(str(img_path))
                comp_mask_path = SRC_ABNORMAL_COMP_MASKS / (img_path.stem + ".png")
                defect_mask_path = SRC_ABNORMAL_DEFECT_MASKS / (img_path.stem + ".png")
                
                if not comp_mask_path.exists():
                    continue
                    
                comp_mask_full = cv2.imread(str(comp_mask_path), cv2.IMREAD_GRAYSCALE)
                defect_mask_full = cv2.imread(str(defect_mask_path), cv2.IMREAD_GRAYSCALE)
                
                defect_id = list(DEFECT_TOKENS.keys())[list(DEFECT_TOKENS.values()).index(defect_token)]
                binary_defect_mask = (defect_mask_full == defect_id).astype(np.uint8) * 255
                
                assigned_comp_token = None
                max_overlap = 0
                assigned_binary_comp_mask = None
                
                for comp_id, comp_token in PART_TOKENS.items():
                    binary_comp_mask = (comp_mask_full == comp_id).astype(np.uint8) * 255
                    overlap = cv2.bitwise_and(binary_defect_mask, binary_comp_mask).sum()
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_comp_token = comp_token
                        assigned_binary_comp_mask = binary_comp_mask
                
                if assigned_comp_token:
                    resized_img, resized_comp_mask, resized_defect_mask = crop_and_resize(
                        image,
                        assigned_binary_comp_mask,
                        binary_defect_mask,
                    )
                    if resized_img is not None and resized_comp_mask is not None and resized_defect_mask is not None:
                        base_name = f"{defect_token.strip('<>')}_{dataset_type}_{img_path.stem}"
                        img_name = f"{base_name}.png"
                        defect_mask_name = f"{base_name}_defect_mask.png"
                        component_mask_name = f"{base_name}_component_mask.png"
                        
                        cv2.imwrite(str(target_dir / "images" / img_name), resized_img)
                        cv2.imwrite(str(target_dir / "defect_masks" / defect_mask_name), resized_defect_mask)
                        cv2.imwrite(str(target_dir / "component_masks" / component_mask_name), resized_comp_mask)
                        
                        prompt = f"a photo of {assigned_comp_token} with {defect_token}"
                        append_metadata(
                            target_dir / "metadata.jsonl",
                            img_name,
                            prompt,
                            component_mask_name=component_mask_name,
                            defect_mask_name=defect_mask_name,
                            object_token=assigned_comp_token,
                            defect_token=defect_token,
                        )
                        
                        # 记录统计
                        stats_abnormal[defect_token][dataset_type] += 1

    # 打印统计信息
    print(">>> 异常组件图划分与裁切完成！统计如下:")
    total_train = 0
    total_test = 0
    for defect, counts in stats_abnormal.items():
        tr = counts['train']
        te = counts['test']
        total_train += tr
        total_test += te
        print(f"  - {defect:30s} | 训练集 (Train): {tr:>3d} 张 | 测试/保留集 (Test): {te:>3d} 张")
        
    print("-" * 65)
    print(f"  -> 总计分配至 DefectFill 训练集 : {total_train} 张")
    print(f"  -> 总计分配至 封存测试集       : {total_test} 张")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"🚀 开始构建数据集... 目标分配比例: {TRAIN_RATIO:.1%} (Train)")
    process_normals()
    process_abnormals()
    print(f"\n✅ 全部处理完成！")
    print(f"📁 工作区路径: {WORKSPACE_DIR}")
