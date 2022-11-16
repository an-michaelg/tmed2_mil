#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader#, WeightedRandomSampler
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip

from random import seed

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

'''
Dataloader for fully supervised learning of TMED2
2 modes, loading on a patient-level and loading on an image-level
Performs data augmentation transforms
'''

DATA_ROOT = "D:\\Datasets\\TMED2\\approved_users_only"
CSV_NAME = 'DEV479\\TMED2_fold0_labeledpart.csv'
# SOCKEYE: DATA_ROOT = TBD
# LOCAL: DATA_ROOT = "D:\\Datasets\\TMED\\approved_users_only"

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 1},
    'mild_mod': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 2},
    'mod_severe': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 2},
    'four_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 3},
    'five_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 2, 'moderate_AS': 3, 'severe_AS': 4},
}

label_schemes_weights: Dict[str, List[float]] = {
    'binary': [2.368, 0.634],
    'mild_mod': [1.579, 1.165, 0.667],
    'mod_severe': [1.579, 0.649, 1.212],
    'four_class': [1.184, 0.874, 1.096, 0.909],
    'five_class': [0.947, 0.828, 4.5, 0.878, 0.727]}

#view_scheme = {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4}
view_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'three_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':2},
    'four_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':3},
    'five_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4},
}

#For human reference
class_labels: Dict[str, List[str]] = {
    'binary': ['No AS', 'AS'],
    'mild_mod': ['No AS', 'Early', 'Significant'],
    'mod_severe': ['No AS', 'Mild-mod', 'Severe'],
    'four_class': ['No AS', 'Mild', 'Moderate', 'Severe'],
    'five_class': ['No AS', 'Mild', 'Mild-mod', 'Moderate', 'Severe']
}

def get_label_weights(scheme):
    return label_schemes_weights[scheme]

def get_as_dataloader(config, batch_size, split, mode, min_frames=0, max_frames=128):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    batch_size: int, batch size
    split : string, 'train'/'val'/'test'/'all' for which section to obtain
    mode : string, 'train'/'test' for setting augmentation/batching ops
    view: string, PLAX/PSAX/PLAXPSAX/no_other/all

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    droot = DATA_ROOT
    
    tra = (mode == 'train')
    flip = (mode == 'train') * 0.5
    shuff = (mode == 'train')
    if mode == 'train': # zero-pad shorter bags in the batch
        zeropad = True
        bsize = batch_size
        info = False
    else: # batch size of 1, no need to zero-pad
        zeropad = False
        bsize = 1
        info = True
        
    dset = TMEDDataset(dataset_root=droot, 
                        split=split,
                        view=config['view'],
                        transform=tra,
                        normalize=True,
                        flip_rate=flip,
                        label_scheme_name=config['label_scheme_name'],
                        view_scheme_name=config['view_scheme_name'],
                        zero_pad_for_batching=zeropad,
                        min_frames=min_frames,
                        max_frames=max_frames,
                        show_info=info
                        )
    
    loader = DataLoader(dset, batch_size=bsize, shuffle=shuff)
    return loader


class TMEDDataset(Dataset):
    def __init__(self, 
                 dataset_root: str = '~/as',
                 view: str = 'PLAX', # PLAX/PSAX/PLAXPSAX/no_other/all
                 split: str = 'train', # train/val/test/'all'
                 transform: bool = True, 
                 normalize: bool = False, 
                 resolution: int = 112,
                 flip_rate: float = 0.5,  
                 label_scheme_name: str = 'five_class', # see label_schemes variable
                 view_scheme_name: str = 'five_class', # see view_schemes variable
                 zero_pad_for_batching: bool = False, # zero-pad short bags for batching
                 min_frames: int = 0, # minimum bag size accepted
                 max_frames: int = 128, # max bag size
                 show_info: bool = False, # also pull path/study information
                 **kwargs):
        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])
        self.dataset_root = dataset_root
        
        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(dataset_root, CSV_NAME))
        # append dataset root to each path in the dataframe
        dataset['path'] = dataset.apply(self.get_data_path_rowwise, axis=1)
        
        if view in ('PLAX', 'PSAX'):
            dataset = dataset[dataset['view_label'] == view]
        elif view == 'plaxpsax':
            dataset = dataset[dataset['view_label'].isin(['PLAX', 'PSAX'])]
        elif view == 'no_other':
            dataset = dataset[dataset['view_label'] != 'A4CorA2CorOther']
        elif view != 'all':
            raise ValueError(f'View should be PLAX/PSAX/PLAXPSAX/no_other/all, got {view}')
       
        self.view_scheme = view_schemes[view_scheme_name]
        self.scheme = label_schemes[label_scheme_name]
        # # remove unnecessary columns in 'diagnosis_label' based on label scheme
        # dataset = dataset[dataset['diagnosis_label'].isin( self.scheme.keys() )]

        # Take train/test/val, note that we don't take any of the view-label-only examples
        if split in ('train', 'val', 'test'):
            dataset = dataset[dataset['diagnosis_classifier_split'] == split]
        elif split != 'all':
            raise ValueError(f'View should be train/val/test/all, got {split}')

        # group the images by study and note how many images there are per study
        dataset['study'] = dataset['query_key'].map(self.get_study_id)
        self.dataset = dataset
        
        # if batching, filter the number of studies by the min and max allowable length
        self.zero_pad_for_batching = zero_pad_for_batching
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.studies = []
        self.study_counts = []
        for s in dataset['study'].unique():
            s_count = len(dataset[dataset['study']==s])
            if s_count > self.min_frames and s_count <= self.max_frames:
                self.studies.append(s)
                self.study_counts.append(s_count)
        
        self.resolution = (resolution, resolution)
        self.transform = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(0.8, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
        self.normalize = normalize
        self.show_info = show_info
        

    def __len__(self) -> int:
        return len(self.studies)

    # get a dataset path from the TMED2 CSV row
    def get_data_path_rowwise(self, pdrow):
        path = join(self.dataset_root, pdrow['SourceFolder'], pdrow['query_key'])
        return path
    
    # get the study ID from the query_keys
    @staticmethod
    def get_study_id(text):
        # remove everything after 's'
        ind = text.find('_')
        return text[:ind]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.061
        std = 0.140
        return (in_tensor-m)/std
    
    # returns 3xHxW zero padding
    def _get_zero_pad(self):
        return torch.tensor(np.zeros((3, *self.resolution)), dtype=torch.float32)
    
    def _get_image(self, data_info):
        '''
        General method to get an image and apply tensor transformation to it

        Parameters
        ----------
        data_info : df row of the item to retrieve

        Returns
        -------
        x: size 3xHxW tensor, stacking of N images belonging to study
        y_AS: size 1 tensor, one-hot encoded diagnosis label
        y_view: size 1 tensor, one-hot encoded view label

        '''
        img_original = plt.imread(data_info['path'])
        
        img = resize(img_original, self.resolution) # HxW
        x = torch.tensor(img).unsqueeze(0) # 1xHxW
        
        y_view = torch.tensor(self.view_scheme[data_info['view_label']])
        y_AS = torch.tensor(self.scheme[data_info['diagnosis_label']])

        if self.transform:
            x = self.transform(x)
        if self.normalize:
            x = self.bin_to_norm(x)

        x = self.gray_to_gray3(x)
        x = x.float() # 3xHxW
        
        return x, y_AS, y_view, data_info['path']

    def _get_study(self, study_id):
        '''
        Get all images pertaining to a study

        Parameters
        ----------
        study_id : String, the ID of the study, eg '1234s0'

        Returns
        -------
        x: size Nx3xHxW tensor, stacking of N images belonging to study
        y_AS: size 1 tensor, one-hot encoded diagnosis label
        y_view: size N tensor, one-hot encoded view labels for each image
        mask: size N tensor, binary for if input-label pair exists for given index

        '''
        study_data = self.dataset[self.dataset['study']==study_id]
        x = []
        y_AS = []
        y_view = []
        mask = []
        paths = []
        num_images = len(study_data)
        # fill up entries with images
        for i in range(num_images):
            image, y_AS, v, path = (self._get_image(study_data.iloc[i]))
            x.append(image)
            y_view.append(v)
            mask.append(torch.tensor(1.))
            paths.append(path)
        if self.zero_pad_for_batching:
            for j in range(self.max_frames - num_images):
                x.append(torch.zeros(3, *self.resolution))
                y_view.append(torch.tensor(0))
                mask.append(torch.tensor(0.))
                paths.append('N/A')

        if self.show_info:
            return x, y_AS, y_view, mask, study_id, paths
        else:
            return x, y_AS, y_view, mask
        
    def __getitem__(self, item):
        study_id = self.studies[item]
        return self._get_study(study_id)
    
def dataset_unit_test(i):
    dset = TMEDDataset(dataset_root=DATA_ROOT, 
                        split='train',
                        view='all',
                        transform=True,
                        normalize=True,
                        flip_rate=0.5,
                        label_scheme_name='five_class',
                        view_scheme_name='five_class',
                        zero_pad_for_batching=False,
                        min_frames=1,
                        max_frames=128
                        )
    x, y_AS, y_view, mask = dset[i]
    return x, y_AS, y_view, mask

def dataloader_unit_test():
    config = {'view':'all', 'label_scheme_name':'five_class', 'view_scheme_name':'five_class'}
    batch_size = 2
    split = 'val'
    mode = 'train'
    min_frames = 64
    max_frames = 128
    dl = get_as_dataloader(config, batch_size, split, mode, min_frames, max_frames)
    d = next(iter(dl))
    return d

if __name__ == '__main__':
    d = dataloader_unit_test()