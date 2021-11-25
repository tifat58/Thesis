import torch
import cv2
import os
import os.path as osp
import torch.utils.data
import pandas as pd


def scaleRadius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    #     print(x)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    #     print(r, s)
    return cv2.resize(img, (0, 0), fx=s, fy=s), r, s


def scaleRadius_mask(img, scale, r, s):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    #     print(x)
    #     r=(x>x.mean()/10).sum()/2
    #     s=scale * 1.0 / r
    #     print(r, s)
    img = cv2.resize(img, (0, 0), fx=s, fy=s)
    img[img > 0] = 255
    return img


class IDRIDDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, data_dir, length, train=True, transform=None):

        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        self.label_list = list()

        self.data_dir = data_dir # '/mnt/sda/haal02-data/IDRID/'
        self.image_dir = 'B. Disease Grading/1. Original Images'
        self.label_dir = 'B. Disease Grading/2. Groundtruths'
        self.dir = '/mnt/sda/haal02-data/IDRID/B. Disease Grading/1. Original Images/a. Training Set'

        if train:
            image_paths = os.path.join(self.data_dir, self.image_dir, 'a. Training Set')
            label_file_path = os.path.join(self.data_dir, self.label_dir,
                                           'a. IDRiD_Disease Grading_Training Labels.csv')


        else:
            image_paths = os.path.join(self.data_dir, self.image_dir, 'b. Testing Set')
            label_file_path = os.path.join(self.data_dir, self.label_dir,
                                           'b. IDRiD_Disease Grading_Testing Labels.csv')

        label_df = pd.read_csv(label_file_path)

        for idx, row in label_df.iterrows():
            self.img_list.append(os.path.join(image_paths, row[0] + '.jpg'))
            self.label_list.append(row[1])


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        scale = 500
        image = cv2.imread(self.img_list[idx])
        # image, r, s = scaleRadius(image1, scale)
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)

        label = self.label_list[idx]

        # label1 = cv2.imread(self.msk_list[idx])
        # label = scaleRadius_mask(label1, scale, r, s)
        # label = label[:, :, 2]

        if self.transform:
            image= self.transform(image)

        # print(image.shape, label.shape)
        return image, label

    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
