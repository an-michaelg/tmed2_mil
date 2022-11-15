#import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation

from random import seed

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)
DATA_ROOT = 'D:/Datasets/CompCars/data'

'''
Dataloader for multi-view learning of CompCars parts
Will load a group of between 1-8 videos, with a "difficulty" metric approximated by
which & how many views are shown
'''

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)
        return dct

def get_car_views(): # zero-indexed, care when accessing folder
    parts = ['HeadLight', 'TailLight', 'FogLight', 'AirIntake', 
             'Console', 'Steering', 'Dashboard', 'GearLever']
    return parts

def get_car_types(): # zero-indexed, care when accessing folder
    types = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 
             'estate', 'pickup', 'hardtop convertible', 'sports', 
             'crossover', 'convertible']
    return types

def get_cars_dataloader(batch_size, split, taxonomy):
    droot = DATA_ROOT
    transform = (split == 'train')
    show_info = (batch_size == 1)
    dset = CarPartsDataset(dataset_root = droot,
                           split = split,
                           taxonomy = taxonomy,
                           transform = transform,
                           show_info = show_info)
    classind_to_gt = dset.classind_to_gt
    if split == 'train':
        w = dset.sampling_weights
        inverse_freq_sampler = WeightedRandomSampler(w, len(w))
        loader = DataLoader(dset, batch_size=batch_size, sampler=inverse_freq_sampler)
    else:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    return loader, classind_to_gt
    

class CarPartsDataset(Dataset):
    def __init__(self, 
                 dataset_root: str = '~/CompCars/data',
                 split: str = 'train', # train/test
                 taxonomy: str = 'model',
                 show_info: bool = False,
                 transform: bool = True, 
                 normalize: bool = True, 
                 resolution: int = 224,
                 rotation_degrees: int = 10,
                 **kwargs):
        
        #dataset_root = dataset_root.replace('~', os.environ['HOME'])
        self.dataset_root = dataset_root
        self.part_data_root = join(dataset_root, 'part')
        self.show_info = show_info
        self.split = split
        self.taxonomy = taxonomy
        # obtain a dictionary of entries from the pickle file
        if self.split == 'train':
            data_dict_path = 'train_test_split/part/tr_dict.pickle'
        else:
            data_dict_path = 'train_test_split/part/te_dict.pickle'
        self.data_dict = load_dict_from_pickle(join(dataset_root, data_dict_path))
        self.views = get_car_views()
        
        # map dictionary entries to a list of keys
        self.key_list = list(self.data_dict.keys())
        
        # determine which car models are in the dataset
        # there are two indexing systems for car models
        # the first is the folder indexing system for all car models in the dataset
        # the second one is the k-th class index specific to the deep learning task
        # which we will create
        self.gt_to_classind, self.classind_to_gt = self._init_class_indices()
        
        # get a list of sampling weights based on the inverse frequency of the class
        self.sampling_weights = self._init_sampling_weights()
        
        # define the input transformation for training and validation
        self.resolution = (resolution, resolution)
        self.transform = None
        if transform:
            self.transform = Compose(
                [RandomRotation(degrees=rotation_degrees), # plus/minus deg
                 RandomResizedCrop(size=self.resolution, scale=(0.8, 1))]
            )
        self.normalize = normalize
        # normalize via imagenet mean because encoder pretraining was done on imagenet
        # see https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
        #data_mean = [0.54018143, 0.5401322, 0.53991949]
        imagenet_mean = [0.485, 0.456, 0.406]
        self.data_mean = torch.tensor(imagenet_mean).unsqueeze(1).unsqueeze(2) #3x1x1
        # data_std = [0.18976531, 0.18984794, 0.1901707 ]
        imagenet_std = [0.229, 0.224, 0.225]
        self.data_std = torch.tensor(imagenet_std).unsqueeze(1).unsqueeze(2) #3x1x1
        
    def _init_class_indices(self):
        # collect a group of car types/makes/models
        gt_codes = []
        gt_to_classind = {}
        classind_to_gt = {}
        for k in self.key_list:
            gt_codes.append(self.data_dict[k][self.taxonomy])
        # only keep unique entries
        gt_codes_set = sorted(set(gt_codes))
        # make a bidirectional dictionary for entries
        for ind, m in enumerate(gt_codes_set):
            gt_to_classind[m] = ind
            classind_to_gt[ind] = m
        return gt_to_classind, classind_to_gt
    
    def _init_sampling_weights(self):
        classes = []
        for k in self.key_list:
            classes.append(self.data_dict[k][self.taxonomy])
        counts = [classes.count(i) for i in classes]
        inverse_counts = [1/i for i in counts]
        return inverse_counts
        

    def __len__(self) -> int:
        return len(self.key_list)
    
    def _get_image(self, indexing_tuple):
        '''
        General method to get a list of images and apply transformation to them

        Parameters
        ----------
        indexing_tuple : (make_id, model_id, year) tuple of the item to retrieve
        
        Returns
        -------
        car_parts : length-8 list of size 3xHxW normalized tensor representing 
            images belonging to each part, a zero image is used if unavailable. 
        car_class: the class index for the specified make/model/type (0..K)
        paths : length-8 list of str paths representing image used for each view
            (None if view is unavailable)
        misc: list containing metadata: make, model, year, car type

        '''

        # from one index, obtain the relevant dict entry
        data = self.data_dict[indexing_tuple]
        car_class = torch.tensor(self.gt_to_classind[data[self.taxonomy]])
        
        # process list of images, apply transform and normalization
        car_parts = []
        paths = []
        for i in range(len(self.views)):
            part_i_images = data[i+1]
            if part_i_images: # if there is a list of images pick a random image
                rand_index = np.random.randint(len(part_i_images))
                img_path = join(self.part_data_root, data[i+1][rand_index])
                img_original = plt.imread(img_path) #HxWx3
                
                img = resize(img_original, self.resolution) # RxRx3
                img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32) # 3xRxR
                if self.transform:
                    img = self.transform(img)
                if self.normalize:
                    img = self.apply_norm(img)
                    
                car_parts.append(img)
                paths.append(data[i+1][rand_index])
            else: # no image, fill with blanks
                img = torch.tensor(np.zeros((3,224,224)), dtype=torch.float32)
                car_parts.append(img)
                paths.append('N/A')
        
        if self.show_info:
            # gather metadata into list
            misc = [data['make'], data['model'], data['year'], data['type']]
            return car_parts, car_class, paths, misc
        else:
            return car_parts, car_class

    def __getitem__(self, item):
        indexing_tuple = self.key_list[item]
        return self._get_image(indexing_tuple)
    
    def num_classes(self):
        return len(self.model_to_classind)
    
    def gt_to_classind(self):
        return self.gt_to_classind
    
    def classind_to_gt(self):
        return self.classind_to_gt
    
    def apply_norm(self, img_tensor):
        # expects image tensor to be 3xHxW
        return (img_tensor - self.data_mean) / self.data_std
    
    def inverse_norm(self, img_tensor):
        # expects image tensor to be 3xHxW
        return img_tensor * self.data_std + self.data_mean

def visualize_data(x, car_class, path, misc):
    # print out the make, model, year
    make, model, year, cartype = misc
    title = 'Displaying {}: {} {}, class_ind = {}'.format(make, year, model, car_class)
    fig, axs = plt.subplots(2, 4, clear=True, figsize=(16,9))
    fig.suptitle(title)
    for r in range(2):
        for c in range(4):
            axs[r, c].imshow(x[4*r+c].permute(1,2,0))
            axs[r, c].axis('off')
            axs[r, c].title.set_text(path[4*r+c])
    plt.show()

def dataset_unit_test(idx):
    cpd = CarPartsDataset(DATA_ROOT, split='train', show_info=True, transform=True, normalize=True, taxonomy='type')
    x, y, xp, meta  = cpd[idx]
    visualize_data(x, y, xp, meta)
    return cpd

def dataloader_unit_test(): # should be list of float32 tensors accompanied by their labels
    cars_dl, code = get_cars_dataloader(batch_size=8, split='train', taxonomy='make')
    #_, m2 = get_cars_dataloader(batch_size=1, split='train')
    d = next(iter(cars_dl))
    return d, code#, m2
    
d, c = dataloader_unit_test()