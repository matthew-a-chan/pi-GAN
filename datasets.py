"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np
import json
import pyspng
import zipfile





class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob('/home/ericryanchan/data/celeba/img_align_celeba/*.jpg')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0

class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob('/home/marcorm/S-GAN/data/cats_bigger_than_128x128/*.jpg')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob('/home/ericryanchan/graf-beta/data/carla/carla/*.png')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3








class ShapeNet_Cars(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        metadata = {
            'dataset_path': '/home/ericryanchan/cars_train.zip',
            'latent_dim': '256',
        }


        self._type = 'zip'
        self._path = metadata['dataset_path']
        self._zipfile = None

        dataset_path = metadata['dataset_path']
        self._path = dataset_path
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')
        PIL.Image.init()
        image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        # TODO REMOVE
        # self._image_fnames = self._image_fnames[:1]

        with self._open_file('dataset.json') as f:
            jsonfile = json.load(f)
            poses, intrinsics = jsonfile['pose'], jsonfile['intrinsics']
            #self.poses = [np.array(p[1]).astype(np.float32) for p in self.poses]
            #self.intrinsics = [np.array(i[1]).astype(np.float32) for i in self.intrinsics]

         # pruning to only upper hemisphere
        self.poses = []
        self.intrinsics = []
        self._image_fnames = []
        for i in range(len(poses)):
            if poses[i][1][2][3] > 0:
                self.poses.append(np.array(poses[i][1]).astype(np.float32))
                self.intrinsics.append(np.array(intrinsics[i][1]).astype(np.float32))
                self._image_fnames.append(image_fnames[i])

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.metadata = metadata
        
        self._zipfile = None

    def __len__(self):
        return len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile
    
    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None
    
    def retrieve(self, index):
        fname = self._image_fnames[index]

        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())

                image = np.array(PIL.Image.fromarray(image).convert('RGB')) # TODO: FIX

        pose = self.poses[index].copy()
        intrinsics = self.intrinsics[index].copy()
        image = self.transform(image)

        # Normalize camera intrinsics by image size
        assert intrinsics.shape == (3, 3), intrinsics

        assert intrinsics[0,1] == 0
        assert intrinsics[2,2] == 1
        assert intrinsics[1,0] == 0
        assert intrinsics[2,0] == 0
        assert intrinsics[2,1] == 0
        
        # All values are in the range [-1, 1], masked pixels have value -1
        return {'images': image, 'pose': pose, 'intrinsics': intrinsics, 'noise': torch.randn((2, self.metadata['latent_dim'],)), 'regularization_noise': torch.randn((2, self.metadata['latent_dim'],)),}

    def __getitem__(self, index):
        if self.metadata.get('conditional', False):
            input = self.retrieve(index)
            target = self.retrieve(torch.randint(0, len(self)-1, (1,)))
            return {
                'images': target['images'],
                'pose': target['pose'],
                'intrinsics': target['intrinsics'],
                'input_images': input['images'],
                'input_pose': input['pose'],
                'input_intrinsics': input['intrinsics'],
            }
        else:
            return self.retrieve(index)['images'], 0

        
        
        
        
        
        
        
        
        
        
        
        
        

        
class Shapenet_Cars_Simple(Dataset):
    """Shapenet Cars Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob('/media/data6/ericryanchan/mafu/data/shapenet_cars/*/*.png')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
#         self.transform = transforms.Compose(
#                     [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index]).convert('RGB')
        X = self.transform(X)

        return X, 0
        
        
        
class FFHQ_Simple(Dataset):
    """FFHQ Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob('/media/data6/ericryanchan/mafu/data/FFHQ_256b/*/*.png')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
#         self.transform = transforms.Compose(
#                     [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0