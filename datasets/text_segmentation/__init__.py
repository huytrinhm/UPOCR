from pathlib import Path
from . import transforms as T 

from .textseg import TextSegDataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose


def build(cfg, args):
    root = Path(cfg['DATA_ROOT'])
    dataset_names = cfg['DATASET_NAMES']
    transforms = make_data_transform(cfg['TRANSFORM'])

    datasets = []
    for dataset_name in dataset_names:
        data_root = root / 'Tampered-IC13'
        
        dataset = TextSegDataset(
            data_root=data_root,
            phase=dataset_name,
            transform=transforms
        )
        datasets.append(dataset)
    
    if len(datasets) > 0:
        dataset = ConcatDataset(datasets)
    
    return dataset


def make_data_transform(cfg):
    transforms = []

    if 'CONVERT_LABEL' in cfg:
        transforms.append(T.ConvertLabel(
            word_effect_value=cfg['CONVERT_LABEL']['WORD_EFFECT_VALUE'],
            ignore_value=cfg['CONVERT_LABEL']['IGNORE_VALUE'],
        ))
    if 'RANDOM_CROP' in cfg:
        transforms.append(T.RandomCrop(
            min_size_ratio=cfg['RANDOM_CROP']['MIN_SIZE_RATIO'],
            max_size_ratio=cfg['RANDOM_CROP']['MAX_SIZE_RATIO'],
            prob=cfg['RANDOM_CROP']['PROB'],
        ))
    if 'RANDOM_HORIZONTAL_FLIP' in cfg:
        transforms.append(T.RandomHorizontalFlip(
            prob=cfg['RANDOM_HORIZONTAL_FLIP']['PROB']
        ))
    if 'RANDOM_DISTORTION' in cfg:
        transforms.append(T.RandomDistortion(
            brightness=cfg['RANDOM_DISTORTION']['BRIGHTNESS'],
            contrast=cfg['RANDOM_DISTORTION']['CONTRAST'],
            saturation=cfg['RANDOM_DISTORTION']['SATURATION'],
            hue=cfg['RANDOM_DISTORTION']['HUE'],
            prob=cfg['RANDOM_DISTORTION']['PROB']
        ))
    if 'RANDOM_ROTATE' in cfg:
        transforms.append(T.RandomRotate(
            angle=cfg['RANDOM_ROTATE']['ANGLE'],
            prob=cfg['RANDOM_ROTATE']['PROB'],
        ))
    if 'RESIZE' in cfg:
        transforms.append(T.Resize(
            size=cfg['RESIZE']['SIZE'],
            phase=cfg['RESIZE']['PHASE']
        ))
    transforms.append(T.ToTensor())

    transforms = Compose(transforms)
    return transforms
