# Component ablation experiments

The component ablation uses five generated datasets:

- `component_ablation_full`: complete MSD-Inpainting.
- `component_ablation_no_dsl`: uniform reconstruction MSE without defect-sensitive weighting.
- `component_ablation_no_dmaa`: both defect- and component-attention alignment losses disabled.
- `component_ablation_no_cdme`: full model with elastically deformed real training masks instead of CDME; no extra diffusion training is required.
- `component_ablation_no_ti`: natural-language component/defect phrases without learned textual-inversion tokens.

All commands default to the two GPUs `0,1`. Override `CUDA_DEVICES`, `NUM_PROCESSES`, or `NUM_GPUS` when needed.

## 1. Train the diffusion ablations

```bash
cd /home/doctor/tzh/MSD-Inpainting
bash run_component_ablation_train.sh
```

This trains Full, w/o DSL, w/o DMAA, and w/o TI. The resolved configuration for every run is saved beside its weights. The w/o-CDME experiment reuses the Full weights because CDME operates only at inference.

## 2. Generate the ablation datasets

```bash
bash run_component_ablation_infer.sh
```

To reuse an existing full-model directory rather than training Full again:

```bash
FULL_WEIGHTS_NAME=increase_text_encoder_learning_rates \
  bash run_component_ablation_infer.sh
```

Legacy Full-model directories may not contain `resolved_train_config.yaml`. The
inference script automatically falls back to `configs/train_config.yaml` for
Full, w/o DSL, w/o DMAA, and w/o CDME because they retain textual-inversion
prompts. The w/o-TI variant cannot use this fallback and must be trained by the
new ablation training script.

The five output directories are created under `data/CCM-Defect/generated_defect_dataset`.

## 3. Evaluate global and local synthesis quality

```bash
bash run_component_ablation_eval.sh
```

This evaluates G-KID, L-KID, G-IC-LPIPS, L-IC-LPIPS, and mask validity from the saved generated images. It does not regenerate images.

## 4. Train and evaluate downstream segmentation

```bash
NUM_GPUS=2 bash run_component_ablation_segmentation.sh
```

Every generated dataset is balanced to the same per-task sample count. MobileViT-UNet is trained with seeds 42, 43, and 44, selected using foreground mIoU on the shared real validation split, and evaluated on the identical real holdout set.

## 5. Export the paper Table 4 rows

After segmentation finishes, rerun:

```bash
bash run_component_ablation_eval.sh
```

The final files are:

- `data/CCM-Defect/eval_reports/component_ablation/table4_component_ablation.csv`
- `data/CCM-Defect/eval_reports/component_ablation/table4_component_ablation_rows.tex`

The CSV retains the three-seed mean and sample standard deviation for foreground mIoU and recall. The LaTeX file contains rows ready to paste into Table 4.
