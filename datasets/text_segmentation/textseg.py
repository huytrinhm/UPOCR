import os 
import json  
from PIL import Image
from torch.utils.data import Dataset 

class TextSegDataset(Dataset):
    def __init__(self, data_root, phase, transform):
        self.data_root = data_root 
        self.phase = phase 
        self.transform = transform 
        self.task = 'text segmentation'

        self._get_paths(data_root)
        self._read_annos()


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        rects, labels = self.annos[index]
        segmap = self.draw_segmap(image, rects, labels)

        data = {
            'image': image,
            'label': segmap,
            'filepath': image_path,
            'task': self.task,
            'ignore_value': 128
        }

        if self.transform:
            data = self.transform(data)

        return data


    def __len__(self):
        return len(self.image_paths)


    def draw_segmap(self, image, rects, labels):
        width, height = image.size 
        segmap = np.zeros((height, width, 3), dtype=np.uint8)

        for rect, label in zip(rects, labels):
            if label == 2:
                cv2.rectangle(segmap, 
                    (rect[0], rect[1]), (rect[2], rect[3]), 
                    (255, 255, 255), thickness=-1)
            elif label == 1:
                pass
            else:
                raise ValueError

        segmap = Image.fromarray(segmap).convert('RGB')
        return segmap

    def _read_annos(self):
        self.annos = []
        for gt_path in self.gt_paths:
            lines = open(gt_path, 'r').read().splitlines()
            lines = [list(map(int, line.split(','))) for line in lines]
            lines = np.array(lines)
            rects = lines[:, :4]
            labels = lines[:, 4]
            self.annos.append([rects, labels])

    def _get_paths(self, data_root):
        image_folder = os.path.join(data_root, f'{self.phase}_img')
        gt_folder = os.path.join(data_root, f'{self.phase}_gt')

        image_names = os.listdir(image_folder)
        image_names.sort()
        self.image_paths = [os.path.join(image_folder, name) for name in image_names]
        self.gt_paths = [os.path.join(gt_folder, os.path.splitext(name)[0] + '.txt') for name in image_names]