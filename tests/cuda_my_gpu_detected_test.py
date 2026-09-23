import torch

def test_cuda():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 15858*1024*1024