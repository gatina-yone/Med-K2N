import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Any, List, Optional, Tuple, Union


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        img, mask = sample['img'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        # 若未提供，将在调用时根据通道数动态设置
        self.mean = mean
        self.std = std

    def _expand_stats(self, C: int, stats: Optional[List[float]], default_val: float) -> List[float]:
        """根据通道数扩展/适配均值或方差列表。
        规则：
        - 若传入 None，则返回 [default_val] * C
        - 若长度等于 C，按原样返回
        - 若长度为 3 且 C 是 3 的倍数，则复制扩展到 C
        - 其他不匹配情况，回退为 [default_val] * C
        """
        if stats is None:
            return [default_val] * C
        if len(stats) == C:
            return stats
        if len(stats) == 3 and C % 3 == 0:
            times = C // 3
            return stats * times
        return [default_val] * C

    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':
                # 目标为生成图像，保持在[0,1]范围内，不做归一化
                sample[k] = sample[k].float()
                continue
            elif k == 'img':
                sample[k] = sample[k].float()
                C = sample[k].shape[0]
                # 对于医学图像，保持在[0,1]范围内，使用identity normalization
                mean = self._expand_stats(C, self.mean, 0.0)
                std = self._expand_stats(C, self.std, 1.0)
                sample[k] = TF.normalize(sample[k], mean, std)
            else:
                sample[k] = sample[k].float()

        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            # 修复：更温和的颜色调整，适合医学图像
            brightness = random.uniform(0.9, 1.1)  # 减少亮度变化范围
            contrast = random.uniform(0.9, 1.1)    # 减少对比度变化范围  
            saturation = random.uniform(0.95, 1.05) # 最小化饱和度变化
            # Split img into 3-channel modals
            modals = torch.split(sample['img'], 3, dim=0)
            new_modals = []
            for modal in modals:
                modal = TF.adjust_brightness(modal, brightness)
                modal = TF.adjust_contrast(modal, contrast)
                modal = TF.adjust_saturation(modal, saturation)
                new_modals.append(modal)
            sample['img'] = torch.cat(new_modals, dim=0)
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            # Split img into 3-channel modals for sharpness adjustment
            modals = torch.split(sample['img'], 3, dim=0)
            new_modals = []
            for modal in modals:
                modal = TF.adjust_sharpness(modal, self.sharpness)
                new_modals.append(modal)
            sample['img'] = torch.cat(new_modals, dim=0)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            # Split img into 3-channel modals for autocontrast
            modals = torch.split(sample['img'], 3, dim=0)
            new_modals = []
            for modal in modals:
                modal = TF.autocontrast(modal)
                new_modals.append(modal)
            sample['img'] = torch.cat(new_modals, dim=0)
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: Union[int, Tuple[int, int], List[int]] = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            if isinstance(self.kernel_size, (list, tuple)):
                ks = list(self.kernel_size)
            else:
                ks = [int(self.kernel_size), int(self.kernel_size)]
            sample['img'] = TF.gaussian_blur(sample['img'], ks)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
            return sample
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, img.shape[0])  # Adjust for multi-channel
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        fill_img = None  # 不需要对 img 指定 fill
        fill_lbl = [float(self.seg_fill)] * int(label.shape[0])
        out_img = TF.affine(
            img,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            TF.InterpolationMode.BILINEAR,
        )
        out_lbl = TF.affine(
            label,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            TF.InterpolationMode.BILINEAR,
            fill=fill_lbl,
        )
        return out_img, out_lbl


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            for k, v in sample.items():
                fill_val = self.seg_fill if k == 'mask' else 0
                fill_list = [float(fill_val)] * int(v.shape[0])
                sample[k] = TF.rotate(
                    v,
                    random_angle,
                    TF.InterpolationMode.BILINEAR,
                    expand=self.expand,
                    fill=fill_list,
                )
        return sample
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int, int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        size = list(self.size) if isinstance(self.size, tuple) else self.size
        return TF.center_crop(img, size), TF.center_crop(mask, size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int, int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int, int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = [0, 0, self.size[1] - img.shape[2], self.size[0] - img.shape[1]]
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int, int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH / H, tW / W) if W > H else max(tH / H, tW / W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        img = TF.resize(img, [nH, nW], TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [nH, nW], TF.InterpolationMode.BILINEAR)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask


class Resize:
    def __init__(self, size: Union[int, Tuple[int, int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample: dict) -> dict:
        H, W = sample['img'].shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        for k, v in sample.items():
            sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.BILINEAR)
        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32

        for k, v in sample.items():
            sample[k] = TF.resize(v, [alignH, alignW], TF.InterpolationMode.BILINEAR)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int, int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        H, W = sample['img'].shape[1:]
        if isinstance(self.size, int):
            tH, tW = self.size, self.size
        else:
            tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        scale = int(tH*ratio), int(tW*ratio)  # Adjusted for square-ish, but original had *4, but for 256, fine
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        for k, v in sample.items():
            sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                sample[k] = TF.pad(v, padding, fill=self.seg_fill if k == 'mask' else 0)

        return sample



def get_train_augmentation(size: Union[int, Tuple[int, int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomColorJitter(p=0.1),  # 降低颜色抖动概率，医学图像对颜色敏感
        RandomHorizontalFlip(p=0.5), 
        RandomGaussianBlur((3, 3), p=0.1),  # 降低模糊概率，保持细节
        RandomResizedCrop(size, scale=(0.8, 1.2), seg_fill=seg_fill),  # 修复：更温和的裁剪范围
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Identity normalization to keep [0,1] range
    ])

def get_val_augmentation(size: Union[int, Tuple[int, int], List[int]]):
    return Compose([
        Resize(size),
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Identity normalization to keep [0,1] range
    ])


if __name__ == '__main__':
    h = 230
    w = 420
    sample = {}
    sample['img'] = torch.cat([torch.randn(3, h, w) for _ in range(3)], dim=0)  # 9ch
    sample['mask'] = torch.randn(3, h, w)  # 3ch target
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop((512, 512)),
        Resize((224, 224)),
        Normalize(),
    ])
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)