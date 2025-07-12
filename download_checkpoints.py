# download_checkpoints.py  (offline-build)

import os
import subprocess
import time
import torch

from diffusers.pipelines.controlnet import \
    StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline


from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_NAMES = [
    "lora_garden_architecture_Exterior_SDlife_Chiasedamme_V1.0.safetensors",
    "TS_ChineseTraditionGarden_V10.safetensors",
    "别墅的后花园_V1.safetensors"
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""
    hf_hub_download(
        repo_id="sintecs/interior",
        filename="ruyiGardenLandscapeDesign_v10.safetensors",
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
    )
    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras",
            local_dir_use_symlinks=False,
        )


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = [
        ControlNetModel.from_pretrained(
            "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        ),
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        # "SG161222/Realistic_Vision_V3.0_VAE",
        "checkpoints/ruyiGardenLandscapeDesign_v10.safetensors",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    MLSDdetector.from_pretrained("lllyasviel/Annotators")

    return pipe


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
