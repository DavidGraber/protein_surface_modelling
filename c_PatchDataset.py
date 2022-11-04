import numpy as np
import os
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, labels_file, data_dir, transform=None, target_transform=None):
        l_dict = np.load(labels_file, allow_pickle="TRUE").item()
        self.labels = [(key,l_dict[key]) for key in l_dict]
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.labels[idx][0]+'_patch.npy')
        image = np.load(path, allow_pickle=True)
        label = np.asarray(self.labels[idx][1], dtype=np.float64)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label