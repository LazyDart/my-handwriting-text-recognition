import argparse

import torch
from PIL import Image

from src.config import get_config
from src.models.install import get_dtype, get_model, get_processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path of the image file to process.")
    args = parser.parse_args()

    config = get_config()
    processor = get_processor()
    model = get_model()
    dtype = get_dtype()
    device = config.use_device

    image = Image.open(args.file_path).convert("RGB")
    prompt = (
        "Przepisz dokładnie tekst z obrazu. Zachowaj polskie znaki, "
        "interpunkcję i podział na wiersze."
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(
        messages,  # pyrefly: ignore
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text],
        images=[image],
        text_kwargs={
            "padding": True,
            "truncation": True,
            "max_length": 4096,
        },
        images_kwargs={"size": {"longest_edge": 1540}},
    )

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(  # pyrefly: ignore[bad-argument-type]
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    transcription = processor.decode(generated_ids, skip_special_tokens=True)
    print(transcription.strip())
