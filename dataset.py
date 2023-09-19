# from __future__ import print_function
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import zipfile
import copy

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, verify_str_arg, check_integrity


class MNIST_soft(VisionDataset):
    """ MNIST Dataset with soft targets.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt`` and  ``MNIST/processed/test.pt`` exist.
        targets: Soft targets.
        train (bool, optional): If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, targets_soft, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST_soft, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.targets_soft = targets_soft

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target_soft, target = self.data[index], self.targets_soft[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target_soft, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    
class SiameseMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, random_seed=None, num_classes=10):
        self.mnist_dataset = mnist_dataset
        self.grouped_examples = self.group_examples()
        self.num_classes = num_classes
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def group_examples(self):
        grouped_examples = {}
        for idx, (_, label) in enumerate(self.mnist_dataset):
            if label not in grouped_examples:
                grouped_examples[label] = []
            grouped_examples[label].append(idx)
        return grouped_examples

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        
        # get the anchor image
        anchor = self.mnist_dataset[index][0]
        anchor_class = self.mnist_dataset[index][1]
        
        # get same class image index
        random_index_2 = np.random.randint(0, len(self.grouped_examples[anchor_class])-1)
        
        # ensure that the index of the same class image is not the same as the anchor image
        while random_index_2 == index:
            random_index_2 = np.random.randint(0, len(self.grouped_examples[anchor_class])-1)
        
        # get same class image
        index_2 = self.grouped_examples[anchor_class][random_index_2]
        image_pos = self.mnist_dataset[index_2][0]
        
        # pick random class
        other_selected_class = np.random.randint(0, self.num_classes-1)
        
        # ensure that the second class image is not the same class as the anchor image
        while other_selected_class == anchor_class:
            other_selected_class = np.random.randint(0, self.num_classes-1)
        
        # get different class image index
        random_index_3 = np.random.randint(0, len(self.grouped_examples[other_selected_class])-1)
        
        # get different class image
        index_3 = self.grouped_examples[other_selected_class][random_index_3]
        image_neg = self.mnist_dataset[index_3][0]

        return anchor, image_pos, image_neg

# MNIST dataset with correction method (and be used for F-MNIST as well)
class SiameseMNISTCorrectionDataset(Dataset):
    def __init__(self, meta_dataset, base_dataset, student_model, teacher_model, random_seed=None, num_classes=10):
        self.meta_dataset = meta_dataset # Used to sample images for teacher model to evaluate on
        self.base_dataset = base_dataset
        self.grouped_examples = self.group_examples()
        self.num_classes = num_classes
        self.student_model = copy.deepcopy(student_model).cpu()
        self.teacher_model = copy.deepcopy(teacher_model).cpu()
        self.num_samples = 30
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def group_examples(self):
        grouped_examples = {}
        for idx, (_, label) in enumerate(self.meta_dataset):
            if label not in grouped_examples:
                grouped_examples[label] = []
            grouped_examples[label].append(idx)
        return grouped_examples
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        
        self.student_model.eval()
        self.teacher_model.eval()
        
        # get anchor image
        anchor = self.base_dataset[index][0].reshape(1,1,28,28)
        # get anchor image's given class
        given_class = self.base_dataset[index][1]

        # get anchor image's class predicted by student
        with torch.no_grad():
            student_class = self.student_model(anchor)
            student_class = torch.argmax(student_class, axis=1).to(torch.int64).cpu().item()

        # No need for correction if given class and student class are the same
        if given_class == student_class:
            return self.base_dataset[index][0], given_class

        # Teacher correction if given class and student class are different
        given_class_indices = np.random.choice(self.grouped_examples[given_class], size=self.num_samples, replace=False)
        student_class_indices = np.random.choice(self.grouped_examples[student_class], size=self.num_samples, replace=False)

        given_class_imgs = torch.vstack([self.meta_dataset[idx][0] for idx in given_class_indices]).reshape(-1,1,28,28)
        student_class_imgs = torch.vstack([self.meta_dataset[idx][0] for idx in student_class_indices]).reshape(-1,1,28,28)
        anchor_imgs = anchor.repeat(self.num_samples,1,1,1)

        with torch.no_grad():
            anc_out, given_out, student_out = self.teacher_model(anchor_imgs, given_class_imgs, student_class_imgs)
            given_score = torch.sum((anc_out - given_out)**2, dim=1)
            student_score = torch.sum((anc_out - student_out)**2, dim=1)
            total = (torch.sum(given_score < student_score) / given_score.shape[0]).item()

            if total >= 0.5:
                final_class = given_class
            else:
                final_class = student_class

        return self.base_dataset[index][0], final_class
    
class SiameseSVHNCorrectionDataset(Dataset):
    def __init__(self, meta_dataset, base_dataset, student_model, teacher_model, random_seed=None, num_classes=10):
        self.meta_dataset = meta_dataset # Used to sample images for teacher model to evaluate on
        self.base_dataset = base_dataset
        self.grouped_examples = self.group_examples()
        self.num_classes = num_classes
        self.student_model = copy.deepcopy(student_model).cpu()
        self.teacher_model = copy.deepcopy(teacher_model).cpu()
        self.num_samples = 30
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def group_examples(self):
        grouped_examples = {}
        for idx, (_, label) in enumerate(self.meta_dataset):
            if label not in grouped_examples:
                grouped_examples[label] = []
            grouped_examples[label].append(idx)
        return grouped_examples
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        
        self.student_model.eval()
        self.teacher_model.eval()
        
        # get anchor image
        anchor = self.base_dataset[index][0].reshape(1,3,32,32)
        # get anchor image's given class
        given_class = self.base_dataset[index][1]

        # get anchor image's class predicted by student
        with torch.no_grad():
            student_class = self.student_model(anchor)
            student_class = torch.argmax(student_class, axis=1).to(torch.int64).cpu().item()

        # No need for correction if given class and student class are the same
        if given_class == student_class:
            return self.base_dataset[index][0], given_class

        # Teacher correction if given class and student class are different
        given_class_indices = np.random.choice(self.grouped_examples[given_class], size=self.num_samples, replace=False)
        student_class_indices = np.random.choice(self.grouped_examples[student_class], size=self.num_samples, replace=False)

        given_class_imgs = torch.vstack([self.meta_dataset[idx][0] for idx in given_class_indices]).reshape(-1,3,32,32)
        student_class_imgs = torch.vstack([self.meta_dataset[idx][0] for idx in student_class_indices]).reshape(-1,3,32,32)
        anchor_imgs = anchor.repeat(self.num_samples,1,1,1)

        with torch.no_grad():
            anc_out, given_out, student_out = self.teacher_model(anchor_imgs, given_class_imgs, student_class_imgs)
            given_score = torch.sum((anc_out - given_out)**2, dim=1)
            student_score = torch.sum((anc_out - student_out)**2, dim=1)
            total = (torch.sum(given_score < student_score) / given_score.shape[0]).item()

            if total >= 0.5:
                final_class = given_class
            else:
                final_class = student_class

        return self.base_dataset[index][0], final_class
    
class CustomDataset(Dataset):
    def __init__(self, original_dataset, new_labels):
        self.original_dataset = original_dataset
        self.new_labels = new_labels

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]  # Get the image and original label
        new_label = self.new_labels[idx]  # Get the new label from y'
        return image, new_label



def collate_fn(batch):
    anchors = []
    img_pos = []
    img_neg = []
    for i in batch:
        anchors.append(i[0])
        img_pos.append(i[1])
        img_neg.append(i[2])
    return torch.stack(anchors), torch.stack(img_pos), torch.stack(img_neg)