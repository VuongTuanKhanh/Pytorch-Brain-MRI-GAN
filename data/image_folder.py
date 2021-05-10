"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import nibabel as nib
from shutil import rmtree
import numpy as np
import cv2
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = images[:min(max_dataset_size, len(images))]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_mri_images(file):
    img = nib.load(file)
    data = img.get_fdata()
    maxx = data.max()
    data = data/maxx
    
    return data, data.shape[-1]


def save_image_data(image_folder, target_folder):
    image_file_format = '{}{}_{}.nii.gz'
    file_path_t1 = image_file_format.format(image_folder, image_folder.split('/')[-2], 't1')
    file_path_t2 = image_file_format.format(image_folder, image_folder.split('/')[-2], 't1ce')
    
    t1_img, _ = get_mri_images(file_path_t1)
    t2_img, _ = get_mri_images(file_path_t2)
    
    file_name = image_folder.split('/')[-2].split('_')[-1]
    image_size = t1_img.shape[0]
    for i in range(30, 110):
        canvas = np.empty((image_size, image_size*2), np.uint8)
        canvas[:, :image_size] = (t1_img[:, :, i] * 255).astype('int')
        canvas[:, image_size:] = (t2_img[:, :, i] * 255).astype('int')
        cv2.imwrite(target_folder + file_name + '_' + str(i) + '.jpg', canvas)


def make_test_dataset(validation_folder, save_folder):
    all_images_folder = [validation_folder + f + '/' for f in os.listdir(validation_folder)][:-2]
    if os.path.isdir(save_folder):
        rmtree(save_folder)
    os.mkdir(save_folder)
    os.mkdir(save_folder + 'test/')
    rand_fld = np.random.choice(all_images_folder)
    save_image_data(rand_fld, save_folder + 'test/')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
