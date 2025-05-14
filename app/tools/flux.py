import os
import time
from typing import Any, Dict
import numpy as np
import torch
from PIL import Image, Image as PILImage
import cv2
import onnxruntime
from loguru import logger
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import FluxPipeline, AutoModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from huggingface_hub import login

HF_KEY: str | None = os.getenv("HF_API_KEY")
if HF_KEY is not None:
    login(HF_KEY)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device: str = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

image_with: int = 1088
image_height: int = 1920
guidance_scale: float = 5.0
num_inference_steps: int = 28

quant_config_transformer = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

quant_config_diffusers = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

text_encoder = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config_transformer,
    torch_dtype=torch.float16,
)

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config_diffusers,
    torch_dtype=torch.float16,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    text_encoder_2=text_encoder,
    torch_dtype=torch.float16,
)

for _, module in pipe.components.items():
    if hasattr(module, "to"):
        module.to(device)

def pre_process(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def post_process(img: np.ndarray) -> np.ndarray:
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
    return img


def inference(model_path: str, img_array: np.ndarray) -> np.ndarray:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(model_path, options)
    ort_inputs: Dict[str, np.ndarray] = {
        ort_session.get_inputs()[0].name: img_array
    }
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]


def convert_pil_to_cv2(image: PILImage.Image) -> np.ndarray:
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def upscale_image(pil_image: PILImage.Image) -> PILImage.Image:
    model_path = os.path.join("models", "modelx4.ort")
    img = convert_pil_to_cv2(pil_image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        alpha_output = post_process(inference(model_path, pre_process(alpha)))
        alpha_output = cv2.cvtColor(alpha_output, cv2.COLOR_BGR2GRAY)

        img = img[:, :, 0:3]
        image_output = post_process(inference(model_path, pre_process(img)))
        image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2BGRA)
        image_output[:, :, 3] = alpha_output

    elif img.shape[2] == 3:
        image_output = post_process(inference(model_path, pre_process(img)))

    return Image.fromarray(image_output)


def generate_image(prompt: str) -> str:
    os.makedirs("generated_images", exist_ok=True)

    timestamp = int(time.time())
    prompt_slug = prompt.lower().replace(' ', '_')[:20]
    upscaled_filename_base = f"{prompt_slug}_{timestamp}"

    generator = torch.Generator().manual_seed(0)
    base_image_width = int(image_with // 4)
    base_image_height = int(image_height // 4)

    pipe_args: Dict[str, Any] = {
        "prompt": prompt,
        "width": base_image_width,
        "height": base_image_height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "max_sequence_length": 512,
        "generator": generator,
    }

    logger.info("Génération de l'image de base en cours...")
    start_time = time.time()
    result = pipe(**pipe_args)
    total_time = time.time() - start_time
    logger.info(f"Image de base générée en {total_time:.2f}s")

    image = result.images[0]

    logger.info("Upscale en cours...")
    start_upscale_time = time.time()
    upscaled_image = upscale_image(image)
    upscale_time = time.time() - start_upscale_time
    logger.info(f"Upscale effectué en {upscale_time:.2f}s")

    upscaled_filename = f"{upscaled_filename_base}_final.png"
    upscaled_file_path = os.path.join("generated_images", upscaled_filename)

    upscaled_image.save(upscaled_file_path)

    return upscaled_file_path
