from torch.utils.data import Dataset
import os
import cv2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, imgs_path, imgs=None, labels=None, transformer=None, target_transformer=None, imgs_subfolder=[]):
            super(ImageDataset, self).__init__()
            self.__imgs_path = imgs_path
            self.__labels = labels
            self.__imgs = imgs
            self.__transformer = transformer
            self.__target_transformer = target_transformer
            self.__imgs_subfolder = imgs_subfolder
            if self.__labels is None:
                self.__initialize_dataset()

    def __initialize_dataset(self):
        self.__imgs = []
        self.__labels = []
        label = 0
        for subfolder in self.__imgs_subfolder:
            path = os.path.join(self.__imgs_path, subfolder)
            print(path, "   ", label)
            files = os.listdir(path)
            self.__imgs.extend(files)
            self.__labels.extend(label for _ in range(len(files)))
            label += 1

    def __len__(self):
        return len(self.__imgs)

    def __getitem__(self, idx):
        label = self.__labels[idx]
        if len(self.__imgs_subfolder) > 0:
            img_path = os.path.join(self.__imgs_path, self.__imgs_subfolder[label], self.__imgs[idx])
        else:
            img_path = os.path.join(self.__imgs_path, self.__imgs[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        if self.__transformer:
            img = self.__transformer(img)
        if self.__target_transformer:
            label = self.__target_transformer(label)

        # return {"x": img, "y": label}
        return img, label