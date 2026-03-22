import yaml
from pathlib import Path
from utils.mask_ops import DefectMaskEngine
from utils.inference_pipeline import DefectFillPipeline

def main():
    # 1. 读取 YAML 配置文件
    config_path = "configs/inference_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    paths = config["paths"]
    infer_config = config["inference"]
    tasks = config["tasks"]

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
