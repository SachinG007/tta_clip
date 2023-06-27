import os
from .imagenet import ImageNet
from .common import ImageFolderWithPaths
import torch

class ImageNetSketch(ImageNet):

    def populate_train(self):
        traindir = os.path.join(self.location, 'ImageNetSketch', 'train')
        self.train_dataset = ImageFolderWithPaths(traindir,
                                                  transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

        if self.custom:
            print('Not implemented')
            exit(0)

    def get_test_path(self):
        return os.path.join(self.location, 'ImageNetSketch', 'train')
