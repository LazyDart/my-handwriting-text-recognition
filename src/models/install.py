import torch
from peft import PeftModel
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from src.config import get_config

if __name__ == "__main__":
    config = get_config()
    adapter_id = config.adapter_id
    base_model_id = config.base_model_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = LightOnOcrProcessor.from_pretrained(adapter_id)
    base_model = LightOnOcrForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_id).to(device)
    model.eval()
