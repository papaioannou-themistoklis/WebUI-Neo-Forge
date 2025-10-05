import re

import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing, shared
from modules.modelloader import load_upscalers
from modules.processing import apply_color_correction, setup_color_correction
from modules.ui import switch_values_symbol
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow, InputAccordion, ToolButton

upscale_cache = {}


def limit_size_by_one_dimension(w: int, h: int, limit: int) -> tuple[int, int]:
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w > limit:
        h = limit * h // w
        w = limit

    return (int(w), int(h))


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def ui(self):
        selected_tab = gr.Number(value=0, visible=False)

        with InputAccordion(True, label="Upscale", elem_id="extras_upscale") as upscale_enabled:
            with FormRow():
                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.Tab("Scale by", elem_id="extras_scale_by_tab") as tab_scale_by:
                        with gr.Column():
                            upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4, elem_id="extras_upscaling_resize")
                            max_side_length = gr.Slider(minimum=0, maximum=8192, step=8, label="Max Side Length", value=0, elem_id="extras_upscale_max_side_length")

                    with gr.Tab("Scale to", elem_id="extras_scale_to_tab") as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id="upscaling_column_size", scale=4):
                                upscaling_resize_w = gr.Slider(minimum=64, maximum=8192, step=8, label="Width", value=1024, elem_id="extras_upscaling_resize_w")
                                upscaling_resize_h = gr.Slider(minimum=64, maximum=8192, step=8, label="Height", value=1024, elem_id="extras_upscaling_resize_h")
                            with gr.Column(elem_id="upscaling_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                upscaling_res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="upscaling_res_switch_btn", tooltip="Switch width/height")
                                upscaling_crop = gr.Checkbox(label="Crop to fit", value=True, elem_id="extras_upscaling_crop")

            with FormRow():
                extras_upscaler_1 = gr.Dropdown(label="Upscaler 1", elem_id="extras_upscaler_1", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
                extras_upscaler_2 = gr.Dropdown(label="Upscaler 2", elem_id="extras_upscaler_2", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
                create_refresh_button([extras_upscaler_1, extras_upscaler_2], load_upscalers, lambda: {"choices": [x.name for x in shared.sd_upscalers]}, "refresh_upscaler")

            with FormRow():
                extras_color_correction = gr.Checkbox(label="Color Correction", elem_id="extras_color_correction", value=False)
                extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Upscaler 2 visibility", value=0.0, elem_id="extras_upscaler_2_visibility")

        upscaling_res_switch_btn.click(lambda w, h: (h, w), inputs=[upscaling_resize_w, upscaling_resize_h], outputs=[upscaling_resize_w, upscaling_resize_h], show_progress="hidden")
        tab_scale_by.select(fn=lambda: 0, outputs=[selected_tab])
        tab_scale_to.select(fn=lambda: 1, outputs=[selected_tab])

        extras_upscaler_2_visibility.change(
            fn=lambda vis: gr.update(interactive=(vis > 0.04)),
            inputs=[extras_upscaler_2_visibility],
            outputs=[extras_upscaler_2],
            show_progress="hidden",
            queue=False,
        )

        if shared.opts.set_scale_by_when_changing_upscaler:

            def on_selected_upscale_method(upscale_method: str) -> int | None:
                match = re.search(r"(\d)[xX]|[xX](\d)", upscale_method)
                if not match:
                    return gr.update()

                return gr.update(value=int(match.group(1) or match.group(2)))

            extras_upscaler_1.change(on_selected_upscale_method, inputs=[extras_upscaler_1], outputs=[upscaling_resize], show_progress="hidden")

        return {
            "upscale_enabled": upscale_enabled,
            "upscale_cc": extras_color_correction,
            "upscale_mode": selected_tab,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        }

    def _upscale(
        self,
        image: Image.Image,
        info: dict,
        upscaler,
        upscale_mode: int,
        upscale_by: float,
        max_side_length: int,
        upscale_to_width: int,
        upscale_to_height: int,
        upscale_crop: bool,
    ):
        if upscale_mode == 1:
            upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
            info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
        else:
            info["Postprocess upscale by"] = upscale_by
            if max_side_length != 0 and max(*image.size) * upscale_by > max_side_length:
                upscale_mode = 1
                upscale_crop = False
                upscale_to_width, upscale_to_height = limit_size_by_one_dimension(image.width * upscale_by, image.height * upscale_by, max_side_length)
                upscale_by = max(upscale_to_width / image.width, upscale_to_height / image.height)
                info["Max side length"] = max_side_length

        cache_key = (hash(np.array(image.getdata()).tobytes()), upscaler.name, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
        cached_image = upscale_cache.pop(cache_key, None)

        if cached_image is not None:
            image = cached_image
        else:
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        upscale_cache[cache_key] = image
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        if upscale_mode == 1 and upscale_crop:
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
            image = cropped
            info["Postprocess crop to"] = f"{image.width}x{image.height}"

        return image

    def process_firstpass(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_enabled=True,
        upscale_cc=False,
        upscale_mode=1,
        upscale_by=2.0,
        max_side_length=0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        else:
            pp.shared.target_width = int(pp.image.width * upscale_by)
            pp.shared.target_height = int(pp.image.height * upscale_by)

            pp.shared.target_width, pp.shared.target_height = limit_size_by_one_dimension(pp.shared.target_width, pp.shared.target_height, max_side_length)

        if upscale_cc:
            upscale_cache["cc"] = setup_color_correction(pp.image)

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_enabled=True,
        upscale_cc=False,
        upscale_mode=1,
        upscale_by=2.0,
        max_side_length=0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if not upscale_enabled:
            return

        _upscaler_1_name = None if upscaler_1_name == "None" else upscaler_1_name

        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == _upscaler_1_name]), None)
        assert upscaler1 or (_upscaler_1_name is None), f"could not find upscaler named {_upscaler_1_name}"

        if not upscaler1:
            return

        _upscaler_2_name = None if upscaler_2_name == "None" else upscaler_2_name

        upscaler2 = next(iter([x for x in shared.sd_upscalers if x.name == _upscaler_2_name and x.name != "None"]), None)
        assert upscaler2 or (_upscaler_2_name is None), f"could not find upscaler named {_upscaler_2_name}"

        upscaled_image = self._upscale(pp.image, pp.info, upscaler1, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)
        pp.info["Postprocess upscaler"] = upscaler1.name

        if upscaler2 and upscaler_2_visibility > 0:
            second_upscale = self._upscale(pp.image, pp.info, upscaler2, upscale_mode, upscale_by, max_side_length, upscale_to_width, upscale_to_height, upscale_crop)
            if upscaled_image.mode != second_upscale.mode:
                second_upscale = second_upscale.convert(upscaled_image.mode)
            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)

            pp.info["Postprocess upscaler 2"] = upscaler2.name

        if upscale_cc and "cc" in upscale_cache:
            pp.image = apply_color_correction(upscale_cache["cc"], upscaled_image)
        else:
            pp.image = upscaled_image

    def image_changed(self):
        upscale_cache.clear()
