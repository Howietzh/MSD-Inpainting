import yaml
import argparse
from pathlib import Path
from utils.config_overrides import apply_config_overrides
from utils.mask_ops import DefectMaskEngine
from utils.inference_pipeline import DefectFillPipeline

def main():
    # --- 新增：使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Defect Fill Inference with overriding paths")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml", help="Path to inference config.")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to specific epoch LoRA weights.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for this epoch.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted paths, e.g. --set inference.batch_size=4",
    )
    args = parser.parse_args()

    # 1. 读取 YAML 配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    apply_config_overrides(config, args.overrides)
        
    paths = config["paths"]
    infer_config = config["inference"]
    tasks = config["tasks"]

    # --- 新增：用命令行参数覆盖 YAML 里的配置 ---
    if args.lora_weights:
        paths["lora_weights"] = args.lora_weights
    if args.output_dir:
        paths["output_dir"] = args.output_dir

    # 2. 实例化缺陷掩码引擎
    mask_engine = DefectMaskEngine(
        train_dir=Path(paths["train_dir"]),
        cache_file=Path(paths["stats_cache"])
    )
    
    # 3. 实例化推理流水线
    pipeline = DefectFillPipeline(
        model_config_path=paths["model_config"],
        lora_dir=Path(paths["lora_weights"]),
        normal_dir=Path(paths["normal_dir"]),
        output_dir=Path(paths["output_dir"]),
        mask_engine=mask_engine,
        infer_config=infer_config
    )
    
    # 4. 执行基于 DataLoader 与多卡并行的批量生成任务
    pipeline.execute_tasks(tasks)

if __name__ == "__main__":
    main()
