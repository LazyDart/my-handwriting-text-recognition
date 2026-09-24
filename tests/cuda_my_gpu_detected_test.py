import torch


def test_cuda():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory >= 15858 * 1024 * 1024

def test_gpu_operation()
    torch.rand(2,3,device='cuda') @ torch.rand(3,2,device='cuda')