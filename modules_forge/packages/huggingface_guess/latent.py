# reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.52/comfy/latent_formats.py

import torch


class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3512,  0.2297,  0.3227],
            [ 0.3250,  0.4974,  0.2350],
            [-0.2829,  0.1762,  0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3651,  0.4232,  0.4341],
            [-0.2533, -0.0042,  0.1068],
            [ 0.1076,  0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ]
        self.latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]
        self.taesd_decoder_name = "taesdxl_decoder"


class Flux(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778],
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Wan21(LatentFormat):
    latent_channels = 16
    latent_dimensions = 3

    latent_rgb_factors = [
        [-0.1299, -0.1692,  0.2932],
        [ 0.0671,  0.0406,  0.0442],
        [ 0.3568,  0.2548,  0.1747],
        [ 0.0372,  0.2344,  0.1420],
        [ 0.0313,  0.0189, -0.0328],
        [ 0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [ 0.0166,  0.1902,  0.1975],
        [-0.0412,  0.0267, -0.1364],
        [-0.1293,  0.0740,  0.1636],
        [ 0.0680,  0.3019,  0.1128],
        [ 0.0032,  0.0581,  0.0639],
        [-0.1251,  0.0927,  0.1699],
        [ 0.0060, -0.0633,  0.0005],
        [ 0.3477,  0.2275,  0.2950],
        [ 0.1984,  0.0913,  0.1861],
    ]
    latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    def __init__(self):
        self.scale_factor = 1.0
        self.latents_mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, self.latent_channels, 1, 1, 1)

        self.taesd_decoder_name = None

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
