from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import spandrel
import spandrel_extra_arches
import torch

from modules import shared
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone  # noqa
from modules.util import load_file_from_url  # noqa

spandrel_extra_arches.install()

logger = logging.getLogger(__name__)


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None, hash_prefix=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @param hash_prefix: the expected sha256 of the model_url
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, "experiments/pretrained_models")
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=places[0], file_name=download_name, hash_prefix=hash_prefix))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, _ = os.path.splitext(file)
    return model_name


def load_upscalers():
    from modules.esrgan_model import UpscalerESRGAN

    del shared.sd_upscalers

    commandline_model_path = shared.cmd_opts.esrgan_models_path
    upscaler = UpscalerESRGAN(commandline_model_path)
    upscaler.user_path = commandline_model_path
    upscaler.model_download_path = commandline_model_path or upscaler.model_path

    shared.sd_upscalers = [
        *UpscalerNone().scalers,
        *UpscalerLanczos().scalers,
        *UpscalerNearest().scalers,
        *sorted(upscaler.scalers, key=lambda s: s.name.lower()),
    ]


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    arch = model_descriptor.architecture
    logger.info(f'Loaded {arch.name} Model: "{os.path.basename(path)}"')

    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        elif model_descriptor.supports_bfloat16:
            model_descriptor.model.bfloat16()
            half = True
        else:
            logger.warning(f"Model {path} does not support half precision...")

    if dtype:
        model_descriptor.model.to(dtype=dtype)

    logger.debug("Loaded %s from %s (device=%s, half=%s, dtype=%s)", arch, path, device, half, dtype)

    model_descriptor.model.eval()
    return model_descriptor
