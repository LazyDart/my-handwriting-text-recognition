from pydantic import BaseModel


class Config(BaseModel):
    base_model_id: str = "lightonai/LightOnOCR-2-1B-base"
    adapter_id: str = "IHPAN/LightOnOCR-2-1B-htr-polish-1"
    use_device: str = "cuda"

config = Config()


def get_config() -> Config:
    return config
