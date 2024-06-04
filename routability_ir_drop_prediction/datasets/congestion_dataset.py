import os.path as osp
import copy
import numpy as np
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor  # Add these imports
from .augmentation import Flip, Rotation

class CongestionDataset(object):
    def __init__(self, ann_file, dataroot, pipeline=None, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)),  # Resize to 224x224
            ToTensor(),
        ])

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        # print("results['feature']: ", results['feature'].shape)
        results['label'] = np.load(results['label_path'])

        results = self.pipeline(results) if self.pipeline else results
        
        feature =  self.transform(results['feature'])
        label = self.transform(results['label'])

        return feature, label, results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
