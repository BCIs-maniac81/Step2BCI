import torch


class DeviceSwitcher(object):
    def __init__(self, force_device: str = None):
        """
        A class to switch the device used by PyTorch tensors.

        Args:
            force_device (str, optional): The device to use, either 'cuda' (GPU) or 'cpu'.
                If not provided, it checks if a CUDA device is available and uses it if possible.
        """
        self.force_device = force_device
        if isinstance (self.force_device, str):
            if force_device is not None and force_device.lower() not in ['cuda', 'gpu', 'cpu']:
                raise ValueError("Invalid device selected. Allowed devices: 'cuda' (GPU) or 'cpu'.")
        else:
            pass
        self.device = torch.device(self.force_device) if self.force_device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def to_device(self, item, non_blocking: bool = True):
        if isinstance(item, (tuple, list)):
            return [self.to_device(i, non_blocking=non_blocking) for i in item]
        elif isinstance(item, torch.Tensor):
            if self.force_device:
                return item.to(self.force_device, non_blocking=non_blocking)
            else:
                return item
        else:
            return item
