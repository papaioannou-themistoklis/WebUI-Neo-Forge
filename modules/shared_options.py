import os

import gradio as gr

from backend.text_processing import emphasis as sd_emphasis
from modules import localization, shared, shared_gradio_themes, shared_items, ui_components, util
from modules.options import OptionDiv, OptionHTML, OptionInfo, categories, options_section
from modules.paths_internal import data_path, default_output_dir
from modules.shared_cmd_options import cmd_opts
from modules_forge import presets as forge_presets
from modules_forge import shared_options as forge_shared_options

options_templates = {}
hide_dirs = shared.hide_dirs

restricted_opts = {
    "clean_temp_dir_at_start",
    "directories_filename_pattern",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_img2img_samples",
    "outdir_init_images",
    "outdir_samples",
    "outdir_save",
    "outdir_txt2img_grids",
    "outdir_txt2img_samples",
    "samples_filename_pattern",
    "temp_dir",
}

categories.register_category("saving", "Saving")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("presets", "Presets")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")
categories.register_category("svdq", "Nunchaku")

options_templates.update(
    options_section(
        ("saving-images", "Saving Images/Grids", "saving"),
        {
            "samples_save": OptionInfo(True, "Automatically save every generated image").info('if disabled, images will needed to be manually saved via the "Save Image" button'),
            "samples_format": OptionInfo("png", "Image Format", gr.Dropdown, {"choices": ("jpg", "jpeg", "png", "webp", "avif", "heif")}).info('"webp" is recommended if supported'),
            "samples_filename_pattern": OptionInfo("", "Filename pattern for saving images", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "save_images_add_number": OptionInfo(True, "Append an ascending number to the filename", component_args=hide_dirs),
            "save_images_replace_action": OptionInfo("Override", "Behavior when saving image to an existing filename", gr.Radio, {"choices": ("Override", "Number Suffix"), **hide_dirs}),
            "grid_save": OptionInfo(True, "Automatically save every generated image grid").info("<b>e.g.</b> for <b>X/Y/Z Plot</b>"),
            "grid_format": OptionInfo("jpg", "Image Format for Grids", gr.Dropdown, {"choices": ("jpg", "jpeg", "png", "webp", "avif", "heif")}),
            "grid_extended_filename": OptionInfo(False, "Append extended info (seed, prompt, etc.) to the filename when saving grids"),
            "grid_only_if_multiple": OptionInfo(True, "Do not save grids that contain only one image"),
            "grid_prevent_empty_spots": OptionInfo(True, "Prevent empty gaps within a grid"),
            "grid_zip_filename_pattern": OptionInfo("", "Filename pattern for saving .zip archives", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "n_rows": OptionInfo(-1, "Grid Row Count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}).info("-1 for autodetect; 0 for the same as batch size"),
            "grid_text_active_color": OptionInfo("#000000", "Text Color for image grids", ui_components.FormColorPicker, {}),
            "grid_text_inactive_color": OptionInfo("#999999", "Inactive Text Color for image grids", ui_components.FormColorPicker, {}),
            "grid_background_color": OptionInfo("#ffffff", "Background Color for image grids", ui_components.FormColorPicker, {}),
            "save_init_img": OptionInfo(False, "Save a copy of the init image before img2img"),
            "save_images_before_face_restoration": OptionInfo(False, "Save a copy of the image before face restoration"),
            "save_images_before_highres_fix": OptionInfo(False, "Save a copy of the image before Hires. fix"),
            "save_images_before_color_correction": OptionInfo(False, "Save a copy of the image before color correction"),
            "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
            "save_mask_composite": OptionInfo(False, "For inpainting, save the masked composite"),
            "jpeg_quality": OptionInfo(85, "JPEG Quality", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
            "webp_lossless": OptionInfo(False, "Lossless WebP"),
            "export_for_4chan": OptionInfo(True, "Save copies of large images as JPG").info("if the following limits are met"),
            "img_downscale_threshold": OptionInfo(4.0, "File Size limit for the above option", gr.Number).info("in MB"),
            "target_side_length": OptionInfo(4096, "Width/Height limit for the above option", gr.Number).info("in pixels"),
            "img_max_size_mp": OptionInfo(100, "Maximum Grid Size", gr.Number).info("in megapixels; only affect <b>X/Y/Z Plot</b>"),
            "use_original_name_batch": OptionInfo(True, "During batch process in Extras tab, use the input filename for output filename"),
            "save_selected_only": OptionInfo(True, 'When using the "Save" button, only save the selected image'),
            "save_write_log_csv": OptionInfo(True, 'Write the generation parameters to a log.csv when saving images using the "Save" button'),
            "temp_dir": OptionInfo(util.truncate_path(os.path.join(data_path, "tmp")), "Directory for temporary images; leave empty to use the system TEMP folder").info("only used for intermediate/interrupted images"),
            "clean_temp_dir_at_start": OptionInfo(True, "Clean up the temporary directory above when starting webui").info("only when the directory is not the system TEMP"),
            "save_incomplete_images": OptionInfo(False, "Save Interrupted Images"),
            "notification_audio": OptionInfo(True, "Play a notification sound after image generation").info('a "notification.mp3" file is required in the root directory').needs_reload_ui(),
            "notification_volume": OptionInfo(100, "Notification Volume", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),
        },
    )
)

options_templates.update(
    options_section(
        ("saving-videos", "Saving Videos", "saving"),
        {
            "video_save_frames": OptionInfo(False, "Save intermediate frames when generating video"),
            "video_player_auto": OptionInfo(True, "Play the generated video when done"),
            "video_player_loop": OptionInfo(False, "Make the video player loop the playback"),
            "video_explanation": OptionHTML(
                """
Parameters for encoding videos in <b>H.264</b> using <b>FFmpeg</b><br>
Refer to the <a href="https://trac.ffmpeg.org/wiki/Encode/H.264">Wiki</a> for what these parameters mean
                """
            ),
            "video_crf": OptionInfo(16, "CRF", gr.Slider, {"minimum": 0, "maximum": 51, "step": 1}),
            "video_preset": OptionInfo("fast", "Preset", gr.Dropdown, {"choices": ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")}),
            "video_profile": OptionInfo("main", "Profile", gr.Dropdown, {"choices": ("baseline", "main", "high")}),
            "video_container": OptionInfo("mp4", "Extension", gr.Radio, {"choices": ("mp4", "mkv")}),
        },
    )
)

options_templates.update(
    options_section(
        ("saving-paths", "Paths for Saving", "saving"),
        {
            "outdir_samples": OptionInfo("", "Output Directory", component_args=hide_dirs).info("if empty, default to the <b>four</b> folders below"),
            "outdir_txt2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "txt2img-images")), "Output Directory for txt2img Images", component_args=hide_dirs),
            "outdir_img2img_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "img2img-images")), "Output Directory for img2img Images", component_args=hide_dirs),
            "outdir_extras_samples": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "extras-images")), "Output Directory for Extras Images", component_args=hide_dirs),
            "outdir_videos": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "videos")), "Output Directory for Videos", component_args=hide_dirs),
            "div00": OptionDiv(),
            "outdir_grids": OptionInfo("", "Output Directory for Grids", component_args=hide_dirs).info("if empty, default to the <b>two</b> folders below"),
            "outdir_txt2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "txt2img-grids")), "Output Directory for txt2img Grids", component_args=hide_dirs),
            "outdir_img2img_grids": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "img2img-grids")), "Output Directory for img2img Grids", component_args=hide_dirs),
            "div01": OptionDiv(),
            "outdir_save": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "images")), 'Directory for manually saving images via the "Save" button', component_args=hide_dirs),
            "outdir_init_images": OptionInfo(util.truncate_path(os.path.join(default_output_dir, "init-images")), "Directory for saving img2img init images if enabled", component_args=hide_dirs),
        },
    )
)

options_templates.update(
    options_section(
        ("saving-to-dirs", "Saving to Subdirectory", "saving"),
        {
            "save_to_dirs": OptionInfo(True, "Save Images to Subdirectory"),
            "grid_save_to_dirs": OptionInfo(True, "Save Grids to Subdirectory"),
            "use_save_to_dirs_for_ui": OptionInfo(False, 'Save to subdirectory when manually saving images via the "Save" button'),
            "directories_filename_pattern": OptionInfo("[date]", "Folder name pattern for subdirectories", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
            "directories_max_prompt_words": OptionInfo(8, "Max length of prompts for the [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 32, "step": 1, **hide_dirs}),
        },
    )
)

options_templates.update(
    options_section(
        ("upscaling", "Upscaling", "postprocessing"),
        {
            "ESRGAN_tile": OptionInfo(256, "Tile Size for Upscalers", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
            "ESRGAN_tile_overlap": OptionInfo(16, "Tile Overlap for Upscalers", gr.Slider, {"minimum": 0, "maximum": 64, "step": 4}).info("low values = visible seam"),
            "composite_tiles_on_gpu": OptionInfo(False, "Composite the Tiles on GPU").info("improve performance and resource utilization"),
            "upscaler_for_img2img": OptionInfo("None", "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}).info("for resizing the input image if the image resolution is smaller than the generation resolution"),
            "upscaling_max_images_in_cache": OptionInfo(4, "Number of upscaled images to cache", gr.Slider, {"minimum": 0, "maximum": 8, "step": 1}),
            "set_scale_by_when_changing_upscaler": OptionInfo(True, 'Automatically set the "Scale by" factor based on the name of the selected Upscaler'),
            "prefer_fp16_upscalers": OptionInfo(False, "Prefer to load Upscaler in half precision").info("increase speed; reduce quality; will try <b>fp16</b>, then <b>bf16</b>, then fall back to <b>fp32</b> if not supported").needs_restart(),
        },
    )
)

options_templates.update(
    options_section(
        ("face-restoration", "Face Restoration", "postprocessing"),
        {
            "face_restoration": OptionInfo(False, "Restore Faces", infotext="Face restoration").info("after each generation, process the face(s) with a 3rd-party model"),
            "face_restoration_model": OptionInfo("CodeFormer", "Face Restoration Model", gr.Radio, lambda: {"choices": [x.name() for x in shared.face_restorers]}),
            "code_former_weight": OptionInfo(0.5, "CodeFormer Strength", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.05}).info("0 = max effect; 1 = min effect"),
            "face_restoration_unload": OptionInfo(False, "Move the model to CPU after restoration"),
        },
    )
)

options_templates.update(
    options_section(
        ("system", "System", "system"),
        {
            "auto_launch_browser": OptionInfo("Local", "Launch the webui in browser on startup", gr.Radio, {"choices": ("Disable", "Local", "Remote")}).info("Remote = always automatically start; Local = only when not sharing the server, such as <b>--share</b>"),
            "enable_console_prompts": OptionInfo(False, "Print the generation prompts to console"),
            "samples_log_stdout": OptionInfo(False, "Print the generation infotxt to console"),
            "show_warnings": OptionInfo(False, "Show warnings in console").needs_reload_ui(),
            "show_gradio_deprecation_warnings": OptionInfo(False, "Show gradio deprecation warnings in console").needs_reload_ui(),
            "memmon_poll_rate": OptionInfo(5, "VRAM usage polls per second during generation", gr.Slider, {"minimum": 0, "maximum": 50, "step": 1}).info("0 = disable"),
            "multiple_tqdm": OptionInfo(True, "Add an additional progress bar to the console to show the total progress of an entire job"),
            "enable_upscale_progressbar": OptionInfo(True, "Show a progress bar in the console for tiled upscaling"),
            "list_hidden_files": OptionInfo(True, "List the models/files under hidden directories").info('directory is hidden if its name starts with "."'),
            "dump_stacks_on_signal": OptionInfo(False, "Print the stack trace before terminating the webui via Ctrl + C"),
        },
    )
)

options_templates.update(
    options_section(
        ("profiler", "Profiler", "system"),
        {
            "profiling_explanation": OptionHTML(
                """
These settings allow you to enable PyTorch profiler during generation.<br>
Profiling allows you to see which code uses how much of the computer's resources.
Each generation writes its own profile to one file, overwriting previous ones.
The file can be viewed in <a href="chrome:tracing">Chrome</a> or on the <a href="https://ui.perfetto.dev/">Perfetto</a> website.
<br><b>Warning:</b> Writing profile can take up to 30 seconds, and the file itself can be around 500MB in size.
                """
            ),
            "profiling_enable": OptionInfo(False, "Enable Profiling"),
            "profiling_activities": OptionInfo(["CPU"], "Activities", gr.CheckboxGroup, {"choices": ["CPU", "CUDA"]}),
            "profiling_record_shapes": OptionInfo(True, "Record Shapes"),
            "profiling_profile_memory": OptionInfo(True, "Profile Memory"),
            "profiling_with_stack": OptionInfo(True, "Include Python Stack"),
            "profiling_filename": OptionInfo("trace.json", "Profile Filename"),
        },
    )
)

options_templates.update(
    options_section(
        ("API", "API", "system"),
        {
            "api_enable_requests": OptionInfo(True, 'Allow "http://" and "https://" URLs as input images', restrict_api=True),
            "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
            "api_useragent": OptionInfo("", "User Agent for Requests", restrict_api=True),
        },
    )
)

options_templates.update(
    options_section(
        ("sd", "Stable Diffusion", "sd"),
        {
            "sd_model_checkpoint": OptionInfo("", "(Managed by Forge)", gr.State, infotext="Model"),
            "sd_unet": OptionInfo("Automatic", "SD UNet", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list),
            "emphasis": OptionInfo("Original", "Emphasis Mode", gr.Radio, lambda: {"choices": [x.name for x in sd_emphasis.options]}, infotext="Emphasis").info("pay (more:1.1) or (less:0.9) attention to prompts").html(sd_emphasis.get_options_descriptions()),
            "CLIP_stop_at_last_layers": OptionInfo(1, "Clip Skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}, infotext="Clip skip").link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("1 = disable, 2 = skip one layer, etc."),
            "comma_padding_backtrack": OptionInfo(16, "Token Wrap Length", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("for prompts shorter than the threshold, move them to the next chunk of 75 tokens if they do not fit inside the current chunk"),
            "tiling": OptionInfo(False, "Tiling", infotext="Tiling").info("produce a tileable image"),
            "randn_source": OptionInfo("CPU", "Random Number Generator", gr.Radio, {"choices": ("CPU", "GPU", "NV")}, infotext="RNG").info("use <b>CPU</b> for the maximum recreatability across different systems"),
            "divxl": OptionDiv(),
            "sdxl_crop_top": OptionInfo(0, "[SDXL] Crop-Top Coordinate"),
            "sdxl_crop_left": OptionInfo(0, "[SDXL] Crop-Left Coordinate"),
            "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "[SDXL] Low Aesthetic Score", gr.Number),
            "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "[SDXL] High Aesthetic Score", gr.Number),
        },
    )
)

options_templates.update(
    options_section(
        ("vae", "VAE", "sd"),
        {
            "sd_vae_explanation": OptionHTML(
                """
<abbr title='Variational AutoEncoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='Red/Green/Blue'>RGB</abbr>
image to and from latent space representation. Latent space is what Stable Diffusion works on during generation. For txt2img, VAE is used
to create the resulting image after the sampling is finished. For img2img, VAE is additionally used to process user's input image before the sampling.
                """
            ),
            "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list, infotext="VAE").info("None = always use VAE from checkpoint; Automatic = use VAE with the same filename as checkpoint"),
            "sd_vae_overrides_per_model_preferences": OptionInfo(True, '"SD VAE" option overrides per-model preference'),
            "sd_vae_encode_method": OptionInfo("Full", "VAE for Encoding", gr.Radio, {"choices": ("Full", "TAESD")}, infotext="VAE Encoder").info("method to encode image to latent (img2img / Hires. fix / inpaint)"),
            "sd_vae_decode_method": OptionInfo("Full", "VAE for Decoding", gr.Radio, {"choices": ("Full", "TAESD")}, infotext="VAE Decoder").info("method to decode latent to image"),
        },
    )
)

options_templates.update(
    options_section(
        ("img2img", "img2img", "sd"),
        {
            "inpainting_mask_weight": OptionInfo(1.0, "Inpainting Conditioning Mask Strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, infotext="Conditional mask weight"),
            "initial_noise_multiplier": OptionInfo(1.0, "Noise Multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.05}, infotext="Noise multiplier"),
            "img2img_extra_noise": OptionInfo(0.0, "Extra Noise Multiplier for img2img and Hires. fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, infotext="Extra noise").info("0 = disabled; higher = more details in generation"),
            "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors"),
            "img2img_fix_steps": OptionInfo(False, "During img2img, do exactly the number of Steps the slider specifies").info("otherwise, only process <b>Sampling steps</b> x <b>Denoising strength</b> steps"),
            "img2img_background_color": OptionInfo("#808080", "For img2img, fill the transparent parts of the input image with this color", ui_components.FormColorPicker, {}),
            "img2img_sketch_default_brush_color": OptionInfo("#ff0000", "Initial Brush Color for Sketch", ui_components.FormColorPicker, {}).needs_reload_ui(),
            "img2img_inpaint_mask_brush_color": OptionInfo("#808080", "Brush Color for Inpaint Mask", ui_components.FormColorPicker, {}).needs_reload_ui(),
            "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ff0000", "Initial Brush Color for Inpaint Sketch", ui_components.FormColorPicker, {}).needs_reload_ui(),
            "img2img_inpaint_mask_high_contrast": OptionInfo(True, "Use high-contrast brush for inpainting").info("use a checkerboard pattern instead of a solid color").needs_reload_ui(),
            "img2img_inpaint_mask_scribble_alpha": OptionInfo(75, "Inpaint mask alpha (transparency)", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("only affects solid color brush").needs_reload_ui(),
            "return_mask": OptionInfo(False, "For inpainting, append the greyscale mask to results"),
            "return_mask_composite": OptionInfo(False, "For inpainting, append the masked composite to results"),
            "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch of img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 256, "step": 1}).info("0 = disable; -1 = show all; too many images causes severe lag"),
            "overlay_inpaint": OptionInfo(True, "For inpainting, overlay the resulting image back onto the original image").info('when using the "Only masked" option'),
            "img2img_autosize": OptionInfo(False, "Automatically update the Width and Height when uploading image to img2img input"),
            "img2img_batch_use_original_name": OptionInfo(False, "In img2img Batch, use the input filenames when saving").info("<b>Warning:</b> may override existing files"),
        },
    )
)

options_templates.update(
    options_section(
        ("optimizations", "Optimizations", "sd"),
        {
            "cross_attention_optimization": OptionInfo("Automatic", "Cross Attention Optimization", gr.Dropdown, {"choices": ("Automatic",), "interactive": False}),
            "persistent_cond_cache": OptionInfo(True, "Persistent Cond Cache").info("do not recalculate conds if the prompts and parameters have not changed since previous generation"),
            "skip_early_cond": OptionInfo(0.0, "Ignore Negative Prompt during Early Steps", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, infotext="Skip Early CFG").info("in percentage of total steps; 0 = disable; higher = faster"),
            "s_min_uncond": OptionInfo(0.0, "Skip Negative Prompt during Later Steps", gr.Slider, {"minimum": 0.0, "maximum": 8.0, "step": 0.05}).info('in "sigma"; 0 = disable; higher = faster'),
            "s_min_uncond_all": OptionInfo(False, "For the above option, skip every step", infotext="NGMS all steps").info("otherwise, only skip every other step"),
            "div_tome": OptionDiv(),
            "token_merging_explanation": OptionHTML(
                """
<b>Token Merging</b> speeds up the diffusion process by fusing "redundant" tokens together, but also reduces quality as a result.
[<a href="https://github.com/dbolya/tomesd">GitHub</a>] <br>
<b>Note:</b> Has no effect on SDXL when Max Downsample is set to 1
                """
            ),
            "token_merging_ratio": OptionInfo(0.0, "Token Merging Ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}, infotext="Token merging ratio").info("0 = disable; higher = faster"),
            "token_merging_ratio_img2img": OptionInfo(0.0, "Token Merging Ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}).info("overrides base ratio if non-zero"),
            "token_merging_ratio_hr": OptionInfo(0.0, "Token Merging Ratio for Hires. fix", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.05}, infotext="Token merging ratio hr").info("overrides base ratio if non-zero"),
            "token_merging_stride": OptionInfo(2, "Token Merging - Stride", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1}).info("higher = faster"),
            "token_merging_downsample": OptionInfo(1, "Token Merging - Max Downsample", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}).info("higher = faster"),
            "token_merging_no_rand": OptionInfo(False, "Token Merging - No Random").info("reduce randomness by always fusing the same regions"),
        },
    )
)

options_templates.update(
    options_section(
        ("compatibility", "Compatibility", "sd"),
        {
            "compatibility_explanation": OptionHTML("Don't touch these unless you know what you are doing..."),
            "forge_try_reproduce": OptionInfo("None", "Try to reproduce the results from external software", gr.Radio, lambda: {"choices": ["None", "Diffusers", "ComfyUI", "WebUI 1.5", "InvokeAI", "EasyDiffusion", "DrawThings"]}),
            "auto_backcompat": OptionInfo(True, "Automatic backward compatibility").info("automatically enable options for backwards compatibility when importing generation parameters from infotext that has program version."),
            "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
            "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
            "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
            "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
            "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; old: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"),
            "use_downcasted_alpha_bar": OptionInfo(False, "Downcast model alphas_cumprod to fp16 before sampling. For reproducing old seeds.", infotext="Downcast alphas_cumprod"),
        },
    )
)

options_templates.update(
    options_section(
        ("extra_networks", "Extra Networks", "sd"),
        {
            "extra_networks_tree_view_style": OptionInfo("Dirs", "Extra Networks UI Style", gr.Radio, {"choices": ("Tree", "Dirs")}).needs_reload_ui(),
            "extra_networks_hidden_models": OptionInfo("When searched", "Show the Extra Networks in hidden directories", gr.Radio, {"choices": ("Always", "When searched", "Never")}).info('"When searched" option will only show the item when the search string contains 4 characters or more'),
            "extra_networks_default_multiplier": OptionInfo(1.0, "Default Weight for Extra Networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.05}),
            "extra_networks_card_width": OptionInfo(0, "Card Width for Extra Networks").info("in pixels; 0 = auto"),
            "extra_networks_card_height": OptionInfo(0, "Card Height for Extra Networks").info("in pixels; 0 = auto"),
            "extra_networks_card_text_scale": OptionInfo(1.0, "Card Text Scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.05}).info("1 = original size"),
            "extra_networks_card_show_desc": OptionInfo(True, "Show description on cards"),
            "extra_networks_card_description_is_html": OptionInfo(False, "Treat description as raw HTML"),
            "extra_networks_card_order_field": OptionInfo("Path", "Default sorting method for Extra Networks cards", gr.Dropdown, {"choices": ("Path", "Name", "Date Created", "Date Modified")}).needs_reload_ui(),
            "extra_networks_card_order": OptionInfo("Ascending", "Default sorting order for Extra Networks cards", gr.Radio, {"choices": ("Ascending", "Descending")}).needs_reload_ui(),
            "extra_networks_add_text_separator": OptionInfo(" ", "Extra Networks Separator").info("additional text to insert before the Extra Networks syntax"),
            "ui_extra_networks_tab_reorder": OptionInfo("", "Extra Networks Tab Order").info('tab names separated by "," character; empty = default').needs_reload_ui(),
            "extra_tree_div": OptionDiv(),
            "extra_networks_tree_view_default_enabled": OptionInfo(True, "Show the Extra Networks Tree view by default").needs_reload_ui(),
            "extra_networks_tree_view_default_width": OptionInfo(180, "Default Width for the Tree view", gr.Number).needs_reload_ui(),
            "extra_dirs_div": OptionDiv(),
            "extra_networks_show_hidden_directories": OptionInfo(True, "Show Dir buttons of hidden directories").info('directory is hidden if its name starts with "."'),
            "extra_networks_dir_button_function": OptionInfo(False, 'Add a "/" to the Dir buttons').info("buttons will only display the contents of the directory, without acting as a search filter"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_prompt_editing", "Prompt Editing", "ui"),
        {
            "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl + Up/Down", gr.Slider, {"minimum": 0.05, "maximum": 0.25, "step": 0.05}),
            "keyedit_precision_extra": OptionInfo(0.05, "Precision for <lora:0.9> when editing the prompt with Ctrl + Up/Down", gr.Slider, {"minimum": 0.05, "maximum": 0.25, "step": 0.05}),
            "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "RegEx Delimiters when editing the prompt with Ctrl + Up/Down"),
            "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Whitespace Delimiters when editing the prompt with Ctrl + Up/Down", gr.CheckboxGroup, {"choices": ("Tab", "Carriage Return", "Line Feed")}),
            "keyedit_move": OptionInfo(True, "Alt + Left/Right moves prompt chunks"),
            "disable_token_counters": OptionInfo(False, "Disable Token Counter"),
            "include_styles_into_token_counters": OptionInfo(True, "Include enabled Styles in Token Count"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_gallery", "Gallery", "ui"),
        {
            "do_not_show_images": OptionInfo(False, "Do not show any image in gallery"),
            "gallery_height": OptionInfo("", "Gallery Height", gr.Textbox).info("in CSS value; <b>e.g.</b> 768px or 20em").needs_reload_ui(),
            "return_grid": OptionInfo(True, "Show Grids in gallery").info("<b>e.g.</b> for <b>X/Y/Z Plot</b>"),
            "js_modal_lightbox": OptionInfo(True, 'Enable "Lightbox"').info("Full Page Image Viewer"),
            "js_modal_lightbox_initially_zoomed": OptionInfo(True, "[Lightbox]: show images zoomed in by default"),
            "js_modal_lightbox_gamepad": OptionInfo(False, "[Lightbox]: navigate with gamepad"),
            "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "[Lightbox]: gamepad repeat period").info("in ms"),
            "sd_webui_modal_lightbox_icon_opacity": OptionInfo(1.0, "[Lightbox]: control icon unfocused opacity", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, onchange=shared.reload_gradio_theme).info("for mouse only").needs_reload_ui(),
            "sd_webui_modal_lightbox_toolbar_opacity": OptionInfo(0.9, "[Lightbox]: tool bar opacity", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, onchange=shared.reload_gradio_theme).info("for mouse only").needs_reload_ui(),
            "open_dir_button_choice": OptionInfo("Subdirectory", "What directory the [📂] button opens", gr.Radio, {"choices": ("Output Root", "Subdirectory", "Subdirectory (even temp dir)")}),
            "hires_button_gallery_insert": OptionInfo(False, "When using the [✨] button, insert the upscaled image to the gallery").info("otherwise replace the selected image in the gallery"),
            "hires_insert_index": OptionInfo(True, "When the above option is enabled, automatically select the upscaled image").info("otherwise select the original image"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui_alternatives", "UI Alternatives", "ui"),
        {
            "show_refiner": OptionInfo(False, "Display the Refiner Accordion").info('"deprecated" feature for SDXL').needs_reload_ui(),
            "show_rescale_cfg": OptionInfo(False, "Display the Rescale CFG Slider").info("feature for v-pred checkpoints").needs_reload_ui(),
            "show_mahiro": OptionInfo(False, "Display the MaHiRo Toggle").info('see <a href="https://huggingface.co/spaces/yoinked/blue-arxiv">blue-arxiv</a> - <b>id:</b> <ins>2024-1208.1</ins>').needs_reload_ui(),
            "paste_safe_guard": OptionInfo(False, 'Disable the "Read generation parameters" button (↙️) when negative prompt is not empty'),
            "ctrl_enter_interrupt": OptionInfo(False, "Revert [Ctrl + Enter] to only interrupt the generation").info('the current "intended" behavior is to interrupt the current generation then immediately start a new one'),
            "div_classic": OptionDiv(),
            "compact_prompt_box": OptionInfo(False, "Compact Prompt Layout").info("put prompts inside the Generate tab, leaving more space for the gallery").needs_reload_ui(),
            "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_reload_ui(),
            "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Show filenames without folder in the Checkpoint dropdown").info("if disabled, models under subdirectories will be listed like sdxl/anime.safetensors"),
            "hires_fix_show_sampler": OptionInfo(False, "[Hires. fix]: Show checkpoint, sampler, scheduler, and cfg options").needs_reload_ui(),
            "hires_fix_show_prompts": OptionInfo(False, "[Hires. fix]: Show prompt and negative prompt textboxes").needs_reload_ui(),
            "txt2img_settings_accordion": OptionInfo(False, "Put txt2img parameters under Accordion").needs_reload_ui(),
            "img2img_settings_accordion": OptionInfo(False, "Put img2img parameters under Accordion").needs_reload_ui(),
            "interrupt_after_current": OptionInfo(False, "Don't Interrupt in the middle").info("when using the Interrupt button, if generating more than one image, stop after the current generation of an image has finished instead of immediately"),
        },
    )
)

options_templates.update(
    options_section(
        ("ui", "User Interface", "ui"),
        {
            "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None", *localization.localizations.keys()]}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
            "quicksettings_list": OptionInfo([], "Quicksettings List", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("settings that appear at the top of the page <b>instead of</b> in the Settings tab").needs_reload_ui(),
            "ui_tab_order": OptionInfo([], "UI Tab Order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
            "hidden_tabs": OptionInfo([], "Hide UI Tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
            "ui_reorder_list": OptionInfo([], "Parameter order for txt2img / img2img", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
            "gradio_theme": OptionInfo("Default", "Gradio Theme", ui_components.DropdownEditable, lambda: {"choices": ["Default", *shared_gradio_themes.gradio_hf_hub_themes]}).needs_reload_ui(),
            "gradio_themes_cache": OptionInfo(True, "Cache selected theme locally"),
            "show_progress_in_title": OptionInfo(True, "Show generation progress in window title"),
            "send_seed": OptionInfo(True, 'Send the Seed information when using the "Send to" buttons'),
            "send_cfg": OptionInfo(True, 'Send the CFG information when using the "Send to" buttons'),
            "send_size": OptionInfo(True, 'Send the Resolution information when using the "Send to" buttons'),
            "enable_reloading_ui_scripts": OptionInfo(False, 'Additionally reload the "modules.ui" scripts when using "Reload UI"').info("for developing"),
        },
    )
)

options_templates.update(
    options_section(
        ("infotext", "Infotext", "ui"),
        {
            "infotext_explanation": OptionHTML("Infotext is what the webui calls the text that contains generation parameters, and can be used to generate the same image again."),
            "enable_pnginfo": OptionInfo(True, "Write infotext to metadata of generated images"),
            "stealth_pnginfo_option": OptionInfo("None", "Stealth Infotext Mode", gr.Radio, {"choices": ("Alpha", "RGB", "None")}),
            "save_txt": OptionInfo(False, "Write infotext to a text file next to every generated image"),
            "add_model_name_to_info": OptionInfo(True, "Add model name to infotext"),
            "add_model_hash_to_info": OptionInfo(True, "Add model hash to infotext"),
            "add_vae_name_to_info": OptionInfo(True, "Add VAE name to infotext"),
            "add_vae_hash_to_info": OptionInfo(True, "Add VAE hash to infotext"),
            "add_user_name_to_info": OptionInfo(False, "Add user name to infotext when authenticated"),
            "add_version_to_infotext": OptionInfo(True, "Add webui version to infotext"),
            "disable_weights_auto_swap": OptionInfo(True, "Ignore checkpoint information when reading infotext"),
            "infotext_skip_pasting": OptionInfo([], "Ignore fields when reading infotext", ui_components.DropdownMulti, lambda: {"choices": shared_items.get_infotext_names()}),
            "infotext_styles": OptionInfo("Apply if any", "Infer Styles when reading infotext", gr.Radio, {"choices": ("Ignore", "Apply", "Apply if any", "Discard")}).html(
                """
<ul style='margin-left: 1.5em'>
<li><b>Ignore:</b> keep prompt and styles dropdown as it is</li>
<li><b>Apply:</b> remove style text from prompt; always replace styles dropdown value with found styles (even if none was found)</li>
<li><b>Apply if any:</b> remove style text from prompt; if any styles are found in prompt, put them into styles dropdown, otherwise keep it as it is</li>
<li><b>Discard:</b> remove style text from prompt, keep styles dropdown as it is</li>
</ul>
                """
            ),
        },
    )
)

options_templates.update(
    options_section(
        ("preview", "Live Previews", "ui"),
        {
            "show_progressbar": OptionInfo(True, "Show Progress Bar"),
            "live_previews_enable": OptionInfo(True, "Show live previews of images during sampling"),
            "live_previews_image_format": OptionInfo("jpeg", "Live Preview Format", gr.Radio, {"choices": ("jpeg", "png", "webp")}),
            "show_progress_grid": OptionInfo(True, "Show previews of all images in a batch as a grid"),
            "show_progress_type": OptionInfo("RGB", "Live Preview Method", gr.Radio, {"choices": ("Approx NN", "RGB", "TAESD")})
            .info("<b>Approx NN</b> and <b>TAESD</b> will download additional model")
            .html(
                """
<ul style='margin-left: 1.5em'>
<li><b>Approx NN</b>: legacy preview method</li>
<li><b>RGB</b>: fast but low quality preview method</li>
<li><b>TAESD</b>: high quality preview method</li>
</ul>
                """
            ),
            "live_preview_fast_interrupt": OptionInfo(False, "Return image with the selected preview method on interruption").info("speed up interruption"),
            "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show the live previews in full page image viewer"),
            "show_progress_every_n_steps": OptionInfo(10, "Generate live preview every N step", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("-1 = only after completion of a batch"),
            "live_preview_refresh_period": OptionInfo(1000, "Progress Bar and Preview update interval").info("in ms"),
            "prevent_screen_sleep_during_generation": OptionInfo(True, "Force the screen to stay awake during generation"),
        },
    )
)

options_templates.update(
    options_section(
        ("sampler-params", "Sampler Parameters", "sd"),
        {
            "hide_samplers": OptionInfo([], "Hide Samplers", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
        },
    )
)

options_templates.update(
    options_section(
        ("sampler-params", "Sampler Parameters", "sd") if cmd_opts.adv_samplers else (None, "Sampler Parameters"),
        {
            "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Eta DDIM"),
            "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext="Eta"),
            "ddim_discretize": OptionInfo("uniform", "img2img DDIM discretize", gr.Radio, {"choices": ("uniform", "quad")}),
            "s_churn": OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext="Sigma churn"),
            "s_tmin": OptionInfo(0.0, "sigma tmin", gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext="Sigma tmin"),
            "s_tmax": OptionInfo(0.0, "sigma tmax", gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext="Sigma tmax"),
            "s_noise": OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext="Sigma noise"),
            "sigma_min": OptionInfo(0.0, "sigma min", gr.Number, infotext="Schedule min sigma"),
            "sigma_max": OptionInfo(0.0, "sigma max", gr.Number, infotext="Schedule max sigma"),
            "rho": OptionInfo(0.0, "rho", gr.Number, infotext="Schedule rho"),
            "eta_noise_seed_delta": OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext="ENSD"),
            "always_discard_next_to_last_sigma": OptionInfo(False, "Always discard next-to-last sigma", infotext="Discard penultimate sigma"),
            "sgm_noise_multiplier": OptionInfo(False, "SGM noise multiplier", infotext="SGM noise multiplier"),
            "uni_pc_variant": OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ("bh1", "bh2", "vary_coeff")}, infotext="UniPC variant"),
            "uni_pc_skip_type": OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ("time_uniform", "time_quadratic", "logSNR")}, infotext="UniPC skip type"),
            "uni_pc_order": OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, infotext="UniPC order"),
            "uni_pc_lower_order_final": OptionInfo(True, "UniPC lower order final", infotext="UniPC lower order final"),
            "sd_noise_schedule": OptionInfo("Default", "Noise schedule for sampling", gr.Radio, {"choices": ("Default", "Zero Terminal SNR")}, infotext="Noise Schedule"),
            "beta_dist_alpha": OptionInfo(0.6, "Beta scheduler - alpha", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext="Beta scheduler alpha"),
            "beta_dist_beta": OptionInfo(0.6, "Beta scheduler - beta", gr.Slider, {"minimum": 0.01, "maximum": 2.0, "step": 0.01}, infotext="Beta scheduler beta"),
        },
    )
)

options_templates.update(
    options_section(
        ("postprocessing", "Postprocessing", "postprocessing"),
        {
            "postprocessing_enable_in_main_ui": OptionInfo([], "Enable Postprocessing operations in txt2img and img2img", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
            "postprocessing_disable_in_extras": OptionInfo([], "Disable Postprocessing operations in Extras tab", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
            "postprocessing_operation_order": OptionInfo([], "Order of Postprocessing operations", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
        },
    )
)

options_templates.update(
    options_section(
        (None, "Hidden Options"),
        {
            "disabled_extensions": OptionInfo([], "Disable these extensions"),
            "disable_all_extensions": OptionInfo("none", "Disable all extensions", gr.Radio, {"choices": ("none", "extra", "all")}),
            "restore_config_state_file": OptionInfo("", 'Config state file to restore from, under "config-states/" folder'),
            "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
        },
    )
)

options_templates.update(
    options_section(
        ("svdq", "Nunchaku", "svdq"),
        {
            "svdq_cpu_offload": OptionInfo(True, "CPU Offload").info("recommended if the VRAM is less than 16 GB"),
            "svdq_flux_exp": OptionHTML("Flux"),
            "svdq_cache_threshold": OptionInfo(0.0, "Cache Threshold", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}).info("increasing the value enhances speed at the cost of quality; a typical value is 0.12; setting it to 0 disables the effect"),
            "svdq_attention": OptionInfo("nunchaku-fp16", "Attention", gr.Radio, {"choices": ["nunchaku-fp16", "flashattn2"]}).info("RTX 20s GPUs can only use nunchaku-fp16"),
            "svdq_qwen_exp": OptionHTML("Qwen"),
            "svdq_use_pin_memory": OptionInfo(False, "Use Pinned Memory").info("improve load speed at the cost of higher RAM usage"),
            "svdq_num_blocks_on_gpu": OptionInfo(1, "Blocks on GPU", gr.Slider, {"minimum": 1, "maximum": 60, "step": 1}).info("higher = more VRAM usage ; lower = more RAM usage"),
        },
    )
)

forge_shared_options.register(options_templates, options_section, OptionInfo)
forge_presets.register(options_templates, options_section, OptionInfo)
