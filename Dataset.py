import torch
import cv2
import os.path as osp
import torch.utils.data
import numpy as np 
import os
import pandas as pd
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def scaleRadius(img, scale) :
    x=img[int(img.shape[0]/2),:,:].sum(1)
#     print(x)
    r=(x>x.mean()/10).sum()/2
    s=scale * 1.0 / r
#     print(r, s)
    return cv2.resize(img,(0,0), fx=s, fy=s), r, s

def scaleRadius_mask(img, scale, r, s) :
    x=img[int(img.shape[0]/2),:].sum(1)
#     print(x)
#     r=(x>x.mean()/10).sum()/2
#     s=scale * 1.0 / r
#     print(r, s)
    img = cv2.resize(img,(0,0), fx=s, fy=s)
    img[img > 0] = 255
    return img
    
class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None):
        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        dataset_name = 'IDRID'
        self.data_dir = data_dir
        if dataset == 'train':
            self.image_df = pd.read_csv('./data/IDRID/idrid_segmentation_file_label_train.csv')
        else:
            self.image_df = pd.read_csv('./data/IDRID/idrid_segmentation_file_label_test.csv')
            
        with open(osp.join(data_dir, dataset + '_sg.txt'), 'r') as lines:
            for line in lines:
                if dataset_name == 'IDRID':
                    line_arr = line.split(',')
                else:
                    line_arr = line.split()
                
                
                self.img_list.append(osp.join(data_dir, line_arr[0].strip()))
                self.msk_list.append(osp.join(data_dir, line_arr[1].strip('\n')))

    def __len__(self):
#         return len(self.img_list)
        return len(self.image_df)
    
    

    def __getitem__(self, idx):
#         if os.path.isfile(self.msk_list[idx]) == False:
#             print("No file")
        scale = 500
        
        
        img = cv2.imread(os.path.join(self.data_dir, self.image_df['image_path'][idx]))
#         img, r, s = scaleRadius(img, scale)
#         img = cv2.addWeighted(img , 4 , cv2.GaussianBlur( img , ( 0 , 0 ) , scale /30) , -4 , 128)
        
        try:
#             mask_1 = cv2.imread(os.path.join(self.data_dir, self.image_df['seg_he_path'][idx]))
            mask_1 = Image.open(os.path.join(self.data_dir, self.image_df['seg_he_path'][idx]))
            mask_1= np.array(mask_1)
        except:
#             mask_1 = cv2.imread(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_1 = Image.open(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_1= np.array(mask_1)[:,:,0]
        
        if mask_1.ndim > 2:
            mask_1 = mask_1[:,:,-1]
        
        try:
    #         mask_2 = cv2.imread(os.path.join(self.data_dir, self.image_df['seg_ex_path'][idx]))
            mask_2 = Image.open(os.path.join(self.data_dir, self.image_df['seg_ex_path'][idx]))
            mask_2= np.array(mask_2)
            
        except:
#             mask_2 = cv2.imread(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_2 = Image.open(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_2= np.array(mask_2)[:,:,0]
        
        if mask_2.ndim > 2:
#                 print(self.image_df['seg_ex_path'][idx])
            mask_2= mask_2[:,:,-1]
        
        try:
#             mask_3 = cv2.imread(os.path.join(self.data_dir, self.image_df['seg_se_path'][idx]))
            mask_3 = Image.open(os.path.join(self.data_dir, self.image_df['seg_se_path'][idx]))
            mask_3= np.array(mask_3)
        
        except:
#             mask_3 = cv2.imread(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_3 = Image.open(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_3= np.array(mask_3)[:,:,0]
#             print(self.image_df['seg_se_path'][idx])

        if mask_3.ndim > 2:
            mask_3 = mask_3[:,:,-1]

        try:
    #         mask_4 = cv2.imread(os.path.join(self.data_dir, self.image_df['seg_ma_path'][idx]))
            mask_4 = Image.open(os.path.join(self.data_dir, self.image_df['seg_ma_path'][idx]))
            mask_4= np.array(mask_4)
        except:
            mask_4 = Image.open(os.path.join(self.data_dir, 'blank_mask.tif'))
            mask_4= np.array(mask_4)[:,:,0]
            
        if mask_4.ndim > 2:
            mask_4 = mask_4[:,:,-1]
        
#         print("Mask_1 shape: ", mask_1.shape)
#         print("Mask_2 shape: ", mask_2.shape)
#         print("Mask_3 shape: ", mask_3.shape)
#         print("Mask_4 shape: ", mask_4.shape)
#         print('\n')

        
#         mask_1 = scaleRadius_mask(mask_1, scale, r, s)
#         mask_2 = scaleRadius_mask(mask_2, scale, r, s)
#         mask_3 = scaleRadius_mask(mask_3, scale, r, s)
#         mask_4 = scaleRadius_mask(mask_4, scale, r, s)
        
        # reading and scaling
        
        masks = [mask_1, mask_2, mask_3, mask_4]
#         return masks
        
        if self.transform:
            transformed = self.transform(image = img, masks = masks)
            
            transformed['masks'] = [torch.from_numpy(item).unsqueeze(0) for item in transformed['masks']]
            label = torch.cat(transformed['masks'])
            image = transformed['image']
            
            return image.float(), label.float()


#     def __getitem__(self, idx):
# #         if os.path.isfile(self.msk_list[idx]) == False:
# #             print("No file")
#         scale = 500
#         image1 = cv2.imread(self.img_list[idx])
#         image, r, s = scaleRadius(image1, scale)
#         image=cv2.addWeighted (image , 4 , cv2.GaussianBlur( image , ( 0 , 0 ) , scale /30) , -4 , 128)

        
#         label1 = cv2.imread(self.msk_list[idx])
#         label = scaleRadius_mask(label1, scale, r, s)
#         label = label[:, :, 2]
        
        
# #         print(np.amax(label), np.amin(label))
# #         exit()
#         if self.transform:
#             [image, label] = self.transform(image, label)
# #         print(image.shape, label.shape)
#         return image, image #testing 3d labels


    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
