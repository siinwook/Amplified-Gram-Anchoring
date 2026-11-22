"""ImageNet streaming and DINO multi-crop augmentation from final training notebook."""
from functools import partial
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from dinov2.data import MaskingGenerator, collate_data_and_cast
from dinov2.data.transforms import GaussianBlur, make_normalize_transform
from .config import (
    HF_DATASET, HF_TOKEN, SHUFFLE_BUFFER, LOCAL_CROPS, GLOBAL_SIZE,
    LOCAL_SIZE, GRAM_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
)


class DINOAugmentation:
    """
    Same global/local augmentation as the previous notebook.
    For Gram/AGA only, also returns clean 448x448 crops sharing
    the exact global crop geometry.
    """
    def __init__(
        self,
        include_gram=False,
        global_scale=(0.32, 1.0),
        local_scale=(0.05, 0.32),
        local_crops_number=4,
        global_size=224,
        local_size=98,
        gram_size=448,
    ):
        self.include_gram = include_gram
        self.local_crops_number = local_crops_number

        self.global_geometry = transforms.Compose([
            transforms.RandomResizedCrop(
                gram_size,
                scale=global_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.resize_global = transforms.Resize(
            global_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        self.local_geometry = transforms.Compose([
            transforms.RandomResizedCrop(
                local_size,
                scale=local_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])
        self.normalize = normalize

        self.global1 = transforms.Compose([
            color_jitter,
            GaussianBlur(p=1.0),
            normalize,
        ])

        self.global2 = transforms.Compose([
            color_jitter,
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])

        self.local = transforms.Compose([
            color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        image = image.convert("RGB")

        # Always create the same global geometry.
        base1 = self.global_geometry(image)
        base2 = self.global_geometry(image)

        g1 = self.global1(self.resize_global(base1))
        g2 = self.global2(self.resize_global(base2))

        local_crops = [
            self.local(self.local_geometry(image))
            for _ in range(self.local_crops_number)
        ]

        out = {
            "global_crops": [g1, g2],
            "global_crops_teacher": [g1, g2],
            "local_crops": local_crops,
            "offsets": (),
        }

        # Deterministic extra work only for relation modes.
        if self.include_gram:
            out["gram_teacher_crops"] = [
                self.normalize(base1),
                self.normalize(base2),
            ]

        return out


class HFImageNetStream(IterableDataset):
    def __init__(self, hf_stream, transform):
        super().__init__()
        self.hf_stream = hf_stream
        self.transform = transform

    def __iter__(self):
        for sample in self.hf_stream:
            try:
                image = sample["image"].convert("RGB")
                yield self.transform(image), ()
            except Exception:
                continue


def collate_optional_gram(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens,
    mask_generator,
):
    batch = collate_data_and_cast(
        samples_list,
        mask_ratio_tuple=mask_ratio_tuple,
        mask_probability=mask_probability,
        dtype=dtype,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )

    first = samples_list[0][0]
    if "gram_teacher_crops" in first:
        n_gram = len(first["gram_teacher_crops"])
        batch["collated_gram_teacher_crops"] = torch.stack([
            sample[0]["gram_teacher_crops"][i]
            for i in range(n_gram)
            for sample in samples_list
        ])

    return batch


def build_stream_loader(mode, seed=SEED):
    include_gram = mode in {"gram", "aga"}

    stream = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    ).shuffle(
        seed=seed,
        buffer_size=SHUFFLE_BUFFER,
    )

    transform = DINOAugmentation(
        include_gram=include_gram,
        local_crops_number=LOCAL_CROPS,
        global_size=GLOBAL_SIZE,
        local_size=LOCAL_SIZE,
        gram_size=GRAM_SIZE,
    )

    dataset = HFImageNetStream(stream, transform)

    n_tokens = (GLOBAL_SIZE // 14) ** 2
    mask_generator = MaskingGenerator(
        input_size=(GLOBAL_SIZE // 14, GLOBAL_SIZE // 14),
        max_num_patches=int(0.5 * n_tokens),
    )

    collate_fn = partial(
        collate_optional_gram,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
