import argparse
from pathlib import Path

import gradio as gr
import yaml

from utils.gradio_inference import InteractiveDefectFillEngine


def parse_args():
    parser = argparse.ArgumentParser(description="DefectFill Gradio demo")
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--infer-config", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--lora-weights", type=str, default=None)
    parser.add_argument("--normal-dir", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None])
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_initial_values(args):
    infer_config = load_yaml(args.infer_config)
    paths = infer_config["paths"]
    inference = infer_config["inference"]
    return {
        "train_config": args.train_config,
        "infer_config": args.infer_config,
        "lora_weights": args.lora_weights or paths["lora_weights"],
        "normal_dir": args.normal_dir or paths["normal_dir"],
        "stats_cache": args.stats_cache or paths["stats_cache"],
        "device": args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
        "num_inference_steps": inference.get("num_inference_steps", 30),
        "guidance_scale": inference.get("guidance_scale", 7.5),
        "negative_prompt": inference.get("negative_prompt", "blurry, smooth, unrealistic, artifacts"),
        "base_seed": inference.get("base_seed", 42),
        "num_lfs_samples": inference.get("num_lfs_samples", 8),
    }


def slider_hidden():
    return gr.update(visible=False)


def build_param_updates(engine, defect_token):
    if engine is None or not defect_token:
        return (
            slider_hidden(),
            slider_hidden(),
            slider_hidden(),
            slider_hidden(),
            slider_hidden(),
            slider_hidden(),
            slider_hidden(),
        )

    spec = engine.get_param_spec(defect_token)
    kind = spec["kind"]

    length_update = slider_hidden()
    thickness_update = slider_hidden()
    width_update = slider_hidden()
    radius_update = slider_hidden()
    count_update = slider_hidden()
    curvature_min_update = slider_hidden()
    curvature_max_update = slider_hidden()

    if kind == "scratch":
        length_update = gr.update(
            visible=True,
            label=spec["length"]["label"],
            minimum=spec["length"]["minimum"],
            maximum=spec["length"]["maximum"],
            step=1,
            value=spec["length"]["value"],
        )
        thickness_update = gr.update(
            visible=True,
            label=spec["thickness"]["label"],
            minimum=spec["thickness"]["minimum"],
            maximum=spec["thickness"]["maximum"],
            step=1,
            value=spec["thickness"]["value"],
        )
        curvature_min_update = gr.update(
            visible=True,
            label=spec["curvature_min"]["label"],
            minimum=spec["curvature_min"]["minimum"],
            maximum=spec["curvature_min"]["maximum"],
            step=spec["curvature_min"]["step"],
            value=spec["curvature_min"]["value"],
        )
        curvature_max_update = gr.update(
            visible=True,
            label=spec["curvature_max"]["label"],
            minimum=spec["curvature_max"]["minimum"],
            maximum=spec["curvature_max"]["maximum"],
            step=spec["curvature_max"]["step"],
            value=spec["curvature_max"]["value"],
        )
    elif kind == "tear":
        length_update = gr.update(
            visible=True,
            label=spec["length"]["label"],
            minimum=spec["length"]["minimum"],
            maximum=spec["length"]["maximum"],
            step=1,
            value=spec["length"]["value"],
        )
        width_update = gr.update(
            visible=True,
            label=spec["width"]["label"],
            minimum=spec["width"]["minimum"],
            maximum=spec["width"]["maximum"],
            step=1,
            value=spec["width"]["value"],
        )
    else:
        radius_update = gr.update(
            visible=True,
            label=spec["radius"]["label"],
            minimum=spec["radius"]["minimum"],
            maximum=spec["radius"]["maximum"],
            step=1,
            value=spec["radius"]["value"],
        )
        count_update = gr.update(
            visible=True,
            label=spec["count"]["label"],
            minimum=spec["count"]["minimum"],
            maximum=spec["count"]["maximum"],
            step=1,
            value=spec["count"]["value"],
        )

    return (
        length_update,
        thickness_update,
        width_update,
        radius_update,
        count_update,
        curvature_min_update,
        curvature_max_update,
    )


def preview_record(engine, component_token, selected_image_path, use_random_image, base_seed):
    if engine is None:
        raise gr.Error("请先加载模型与数据。")
    preview_image, preview_mask, actual_image_path = engine.preview_record(
        component_token=component_token,
        selected_image_path=selected_image_path,
        use_random_image=use_random_image,
        base_seed=base_seed,
    )
    return gr.update(value=actual_image_path), preview_image, preview_mask


def on_component_change(engine, component_token, use_random_image, base_seed):
    if engine is None:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            None,
            None,
            *build_param_updates(None, None),
        )

    choices = engine.get_normal_image_choices(component_token)
    defect_choices = engine.get_defect_choices(component_token)
    selected_defect = defect_choices[0] if defect_choices else None
    param_updates = build_param_updates(engine, selected_defect)

    if not choices:
        return (
            gr.update(choices=defect_choices, value=selected_defect),
            gr.update(choices=[], value=None),
            None,
            None,
            *param_updates,
        )

    preview_image, preview_mask, actual_image_path = engine.preview_record(
        component_token=component_token,
        selected_image_path=choices[0],
        use_random_image=use_random_image,
        base_seed=base_seed,
    )
    return (
        gr.update(choices=defect_choices, value=selected_defect),
        gr.update(choices=choices, value=actual_image_path),
        preview_image,
        preview_mask,
        *param_updates,
    )


def clear_generation_state():
    return None, None, None, [], None, None, {"stage": "mask_invalidated", "message": "输入或参数已变化，请先重新生成 / 刷新 Mask。"}


def reset_mask_refresh_state():
    return 0


def load_engine(train_config, infer_config, lora_weights, normal_dir, stats_cache, device, base_seed):
    try:
        engine = InteractiveDefectFillEngine(
            train_config_path=train_config,
            infer_config_path=infer_config,
            lora_weights=lora_weights,
            normal_dir=normal_dir,
            stats_cache=stats_cache,
            device=device,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    component_choices = engine.get_component_choices()
    selected_component = component_choices[0] if component_choices else None
    defect_choices = engine.get_defect_choices(selected_component)
    selected_defect = defect_choices[0] if defect_choices else None

    image_choices = engine.get_normal_image_choices(selected_component) if selected_component else []
    selected_image = image_choices[0] if image_choices else None

    preview_image = None
    preview_mask = None
    if selected_component and selected_image:
        preview_image, preview_mask, selected_image = engine.preview_record(
            selected_component,
            selected_image,
            False,
            base_seed,
        )

    param_updates = build_param_updates(engine, selected_defect) if selected_defect else (
        slider_hidden(),
        slider_hidden(),
        slider_hidden(),
        slider_hidden(),
        slider_hidden(),
        slider_hidden(),
        slider_hidden(),
    )
    status = (
        f"已加载模型。device={engine.device.type}, "
        f"components={len(component_choices)}, defects={len(defect_choices)}, "
        f"normal_samples={sum(len(v) for v in engine.normal_records_by_component.values())}"
    )

    return (
        engine,
        None,
        0,
        status,
        gr.update(choices=component_choices, value=selected_component),
        gr.update(choices=defect_choices, value=selected_defect),
        gr.update(choices=image_choices, value=selected_image),
        preview_image,
        preview_mask,
        *param_updates,
    )


def generate_mask(
    engine,
    component_token,
    defect_token,
    selected_image_path,
    use_random_image,
    base_seed,
    mask_refresh_state,
    length,
    thickness,
    width,
    radius,
    count,
    curvature_min,
    curvature_max,
    random_use_cache_range,
):
    if engine is None:
        raise gr.Error("请先加载模型与数据。")

    result = engine.generate_mask_preview(
        component_token=component_token,
        defect_token=defect_token,
        selected_image_path=selected_image_path,
        use_random_image=use_random_image,
        base_seed=base_seed,
        refresh_index=mask_refresh_state,
        length=length,
        thickness=thickness,
        width=width,
        radius=radius,
        count=count,
        curvature_min=curvature_min,
        curvature_max=curvature_max,
        random_use_cache_range=random_use_cache_range,
    )
    return (
        result["mask_payload"],
        int(mask_refresh_state) + 1,
        gr.update(value=result["selected_image_path"]),
        result["original"],
        result["component_mask"],
        result["defect_mask"],
        result["overlay"],
        [],
        None,
        None,
        result["info"],
    )


def generate_result(
    engine,
    mask_state,
    num_inference_steps,
    guidance_scale,
    negative_prompt,
    num_lfs_samples,
):
    if engine is None:
        raise gr.Error("请先加载模型与数据。")

    result = engine.generate_from_mask(
        mask_payload=mask_state,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_lfs_samples=num_lfs_samples,
    )
    return (
        gr.update(value=result["selected_image_path"]),
        result["original"],
        result["component_mask"],
        result["defect_mask"],
        result["overlay"],
        result["candidates"],
        result["best"],
        result["triptych"],
        result["info"],
    )


def create_demo(initial_values):
    with gr.Blocks(title="DefectFill Gradio Demo") as demo:
        engine_state = gr.State(value=None)
        mask_state = gr.State(value=None)
        mask_refresh_state = gr.State(value=0)

        gr.Markdown("# DefectFill 单样本实时生成可视化")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 模型与数据设置")
                train_config = gr.Textbox(label="Train Config", value=initial_values["train_config"])
                infer_config = gr.Textbox(label="Infer Config", value=initial_values["infer_config"])
                lora_weights = gr.Textbox(label="LoRA Weights", value=initial_values["lora_weights"])
                normal_dir = gr.Textbox(label="Normal Dir", value=initial_values["normal_dir"])
                stats_cache = gr.Textbox(label="Stats Cache", value=initial_values["stats_cache"])
                device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value=initial_values["device"])
                load_button = gr.Button("加载模型")
                load_status = gr.Textbox(label="加载状态", interactive=False)

                gr.Markdown("## 生成条件")
                component_token = gr.Dropdown(label="组件类别", choices=[], value=None)
                defect_token = gr.Dropdown(label="缺陷类别", choices=[], value=None)
                use_random_image = gr.Checkbox(label="随机抽取正常图", value=False)
                normal_image_path = gr.Dropdown(label="正常图", choices=[], value=None)
                preview_button = gr.Button("预览当前样本")
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, step=1, value=initial_values["num_inference_steps"])
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.1, value=initial_values["guidance_scale"])
                negative_prompt = gr.Textbox(label="Negative Prompt", value=initial_values["negative_prompt"])
                base_seed = gr.Number(label="Base Seed", value=initial_values["base_seed"], precision=0)
                num_lfs_samples = gr.Slider(label="LFS Candidates", minimum=1, maximum=16, step=1, value=initial_values["num_lfs_samples"])

                gr.Markdown("## 缺陷参数控制")
                random_use_cache_range = gr.Checkbox(label="随机使用 cache 范围", value=False)
                length_slider = gr.Slider(label="Length", minimum=1, maximum=200, step=1, value=50, visible=False)
                thickness_slider = gr.Slider(label="Thickness", minimum=1, maximum=30, step=1, value=5, visible=False)
                width_slider = gr.Slider(label="Width", minimum=1, maximum=30, step=1, value=5, visible=False)
                radius_slider = gr.Slider(label="Radius", minimum=1, maximum=30, step=1, value=5, visible=False)
                count_slider = gr.Slider(label="Count", minimum=1, maximum=10, step=1, value=1, visible=False)
                curvature_min_slider = gr.Slider(label="Curvature Min", minimum=0.0, maximum=1.0, step=0.01, value=0.05, visible=False)
                curvature_max_slider = gr.Slider(label="Curvature Max", minimum=0.0, maximum=1.0, step=0.01, value=0.65, visible=False)
                generate_mask_button = gr.Button("生成 / 刷新 Mask")
                generate_button = gr.Button("开始缺陷生成", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 过程展示")
                with gr.Row():
                    original_image = gr.Image(label="原图", type="pil")
                    component_mask = gr.Image(label="组件 Mask", type="pil")
                with gr.Row():
                    defect_mask = gr.Image(label="Defect Mask", type="pil")
                    defect_overlay = gr.Image(label="Defect Overlay", type="pil")
                candidate_gallery = gr.Gallery(label="LFS 候选结果", columns=4, rows=2, height=420)
                with gr.Row():
                    best_image = gr.Image(label="最终最佳结果", type="pil")
                    triptych_image = gr.Image(label="三联图", type="pil")
                info_json = gr.JSON(label="本次生成信息")

        load_button.click(
            fn=load_engine,
            inputs=[train_config, infer_config, lora_weights, normal_dir, stats_cache, device, base_seed],
            outputs=[
                engine_state,
                mask_state,
                mask_refresh_state,
                load_status,
                component_token,
                defect_token,
                normal_image_path,
                original_image,
                component_mask,
                length_slider,
                thickness_slider,
                width_slider,
                radius_slider,
                count_slider,
                curvature_min_slider,
                curvature_max_slider,
            ],
        ).then(
            fn=clear_generation_state,
            outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
        ).then(
            fn=reset_mask_refresh_state,
            outputs=[mask_refresh_state],
        )

        component_token.change(
            fn=on_component_change,
            inputs=[engine_state, component_token, use_random_image, base_seed],
            outputs=[
                defect_token,
                normal_image_path,
                original_image,
                component_mask,
                length_slider,
                thickness_slider,
                width_slider,
                radius_slider,
                count_slider,
                curvature_min_slider,
                curvature_max_slider,
            ],
        ).then(
            fn=clear_generation_state,
            outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
        ).then(
            fn=reset_mask_refresh_state,
            outputs=[mask_refresh_state],
        )

        normal_image_path.change(
            fn=preview_record,
            inputs=[engine_state, component_token, normal_image_path, use_random_image, base_seed],
            outputs=[normal_image_path, original_image, component_mask],
        ).then(
            fn=clear_generation_state,
            outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
        ).then(
            fn=reset_mask_refresh_state,
            outputs=[mask_refresh_state],
        )

        preview_button.click(
            fn=preview_record,
            inputs=[engine_state, component_token, normal_image_path, use_random_image, base_seed],
            outputs=[normal_image_path, original_image, component_mask],
        ).then(
            fn=clear_generation_state,
            outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
        ).then(
            fn=reset_mask_refresh_state,
            outputs=[mask_refresh_state],
        )

        defect_token.change(
            fn=build_param_updates,
            inputs=[engine_state, defect_token],
            outputs=[
                length_slider,
                thickness_slider,
                width_slider,
                radius_slider,
                count_slider,
                curvature_min_slider,
                curvature_max_slider,
            ],
        ).then(
            fn=clear_generation_state,
            outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
        ).then(
            fn=reset_mask_refresh_state,
            outputs=[mask_refresh_state],
        )

        for control in [
            use_random_image,
            random_use_cache_range,
            base_seed,
            length_slider,
            thickness_slider,
            width_slider,
            radius_slider,
            count_slider,
            curvature_min_slider,
            curvature_max_slider,
        ]:
            control.change(
                fn=clear_generation_state,
                outputs=[mask_state, defect_mask, defect_overlay, candidate_gallery, best_image, triptych_image, info_json],
            ).then(
                fn=reset_mask_refresh_state,
                outputs=[mask_refresh_state],
            )

        generate_mask_button.click(
            fn=generate_mask,
            inputs=[
                engine_state,
                component_token,
                defect_token,
                normal_image_path,
                use_random_image,
                base_seed,
                mask_refresh_state,
                length_slider,
                thickness_slider,
                width_slider,
                radius_slider,
                count_slider,
                curvature_min_slider,
                curvature_max_slider,
                random_use_cache_range,
            ],
            outputs=[
                mask_state,
                mask_refresh_state,
                normal_image_path,
                original_image,
                component_mask,
                defect_mask,
                defect_overlay,
                candidate_gallery,
                best_image,
                triptych_image,
                info_json,
            ],
        )

        generate_button.click(
            fn=generate_result,
            inputs=[
                engine_state,
                mask_state,
                num_inference_steps,
                guidance_scale,
                negative_prompt,
                num_lfs_samples,
            ],
            outputs=[
                normal_image_path,
                original_image,
                component_mask,
                defect_mask,
                defect_overlay,
                candidate_gallery,
                best_image,
                triptych_image,
                info_json,
            ],
        )

    return demo


def main():
    args = parse_args()
    initial_values = build_initial_values(args)
    demo = create_demo(initial_values)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
