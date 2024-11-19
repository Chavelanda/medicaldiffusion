import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def pad_to_multiple(x, divisors=(4, 4, 4)):
    """
    Pads a 3D input tensor along its last three spatial dimensions to make them divisible
    by the given divisors. Ensures symmetric padding where possible.
    
    Args:
        x (torch.Tensor): 3D input tensor of shape (..., D, H, W).
        divisors (tuple): A tuple of 3 integers specifying the divisors for the depth, height, and width.
    
    Returns:
        tuple: (padded_tensor, padding_sizes)
            - padded_tensor (torch.Tensor): Padded tensor.
            - padding_sizes (tuple): Tuple of 3 tuples representing the padding applied 
                                     for each dimension as (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right).
    """
    d, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    div_d, div_h, div_w = divisors

    # Compute padding for each dimension to make divisible
    pad_d = (div_d - d % div_d) % div_d
    pad_h = (div_h - h % div_h) % div_h
    pad_w = (div_w - w % div_w) % div_w

    # Distribute padding symmetrically, adding extra to the end if necessary
    pad_front, pad_back = pad_d // 2, pad_d - pad_d // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    # Apply padding using F.pad
    padded_x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
    
    # Return padded tensor and the padding sizes
    return padded_x, ((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right))


def crop_to_original(x, padding_sizes):
    """
    Crops a padded 3D tensor back to its original size based on the padding_sizes.
    
    Args:
        x (torch.Tensor): The padded 3D tensor of shape (..., D, H, W).
        padding_sizes (tuple): A tuple of 3 tuples representing the padding applied
                               for each dimension as (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right).
    
    Returns:
        torch.Tensor: Cropped tensor with the original size before padding.
    """
    (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right) = padding_sizes
    
    # Compute the cropping indices
    d_start = pad_front
    d_end = -pad_back if pad_back > 0 else None
    
    h_start = pad_top
    h_end = -pad_bottom if pad_bottom > 0 else None
    
    w_start = pad_left
    w_end = -pad_right if pad_right > 0 else None
    
    # Crop the tensor using slicing
    cropped_x = x[..., d_start:d_end, h_start:h_end, w_start:w_end]
    
    return cropped_x