import torch
from peft import PeftModel
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from src.config import get_config

def get_processor() -> LightOnOcrProcessor:
    config = get_config()
    return LightOnOcrProcessor.from_pretrained(config.adapter_id)

def get_dtype():
    config = get_config()
    device = config.use_device
    return torch.bfloat16 if device == "cuda" else torch.float32

def get_model() -> LightOnOcrForConditionalGeneration:
    config = get_config()

    adapter_id = config.adapter_id
    base_model_id = config.base_model_id
    device = config.use_device

    dtype = get_dtype()

    base_model = LightOnOcrForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    return PeftModel.from_pretrained(base_model, adapter_id).to(device)

if __name__ == "__main__":
    processor = get_processor()
    model = get_model()
    model.eval()
