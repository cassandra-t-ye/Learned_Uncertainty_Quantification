# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np
import torch


# +
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        #PUTTING IN TRACE
        #import ipdb ipdb.set_trace() 

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        #print('\nimg_bytes = ',img_bytes)
        try:
            #img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = torch.load(gt_path).float()
        except:
            #print("THIS IS IN THE EXCEPTION")
            #print('\nimg_bytes = ',img_bytes)
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = torch.load(lq_path).float()
            #img_lq = imfrombytes(img_bytes, float32=True)

        except:
            raise Exception("lq path {} not working".format(lq_path))

        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
#            # flip, rotation
#             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])
#             print('img_gt shape =', img_gt.shape)
#             print('img_lq shape =',img_lq.shape)
# # #           ONLY WHEN TRAINING WITH ORIGINAL ONE CHANNEL SIDD
#             num_of_noisy = np.random.randint(0, 4)  
#             color_channel = np.random.randint(0, 2)
            
#             img_lq = torch.tensor(img_lq)
#             img_gt = torch.tensor(img_gt)
            
# #             img_lq = img_lq.repeat((5, 1, 1))
# #             img_gt = img_gt.repeat((5, 1, 1))
#             #img_lq = (512, 512, 3)
    
#             R_img = img_lq[:, :, 0].unsqueeze(0)   #SHAPE: (512, 512)
#             G_img = img_lq[:, :, 1].unsqueeze(0)
#             B_img = img_lq[:, :, 2].unsqueeze(0)
            
#             R_gt = img_gt[:, :, 0].unsqueeze(0) 
#             G_gt = img_gt[:, :, 1].unsqueeze(0) 
#             B_gt = img_gt[:, :, 2].unsqueeze(0)

            
#            if num_of_noisy == 0:  #1 image repeated
#                 if color_channel == 0:  
#                     img_lq = R_img
#                     img_gt = R_gt
                    
#                 if color_channel == 1:
#                     img_lq = G_img
#                     img_gt = G_gt
                    
#                 if color_channel == 2:
#                     img_lq = B_img
#                     img_gt = B_gt
                
#                 img_lq = img_lq.repeat((5, 1, 1))
#                 img_gt = img_gt.repeat((5, 1, 1))
                                                                      
#             elif num_of_noisy == 1:  #2 Images 
#                 if color_channel == 0:
#                     img_lq = torch.cat((R_img, R_img, G_img, R_img, R_img), 0)
#                     img_gt = R_gt
                                  
#                 if color_channel == 1:
#                     img_lq = torch.cat((G_img, G_img, B_img, G_img, G_img), 0)
#                     img_gt = G_gt
               
#                 if color_channel == 2:
#                     img_lq = torch.cat((B_img, B_img, R_img, B_img, B_img), 0)
#                     img_gt = B_gt
                    
#                 img_gt = img_gt.repeat((5, 1, 1))
                                     
#             elif num_of_noisy == 2:  #3 Images 
#                 if color_channel == 0:
#                     img_lq = torch.cat((R_img, B_img, G_img, R_img, R_img), 0)
#                     img_gt = R_gt
                                  
#                 if color_channel == 1:
#                     img_lq = torch.cat((G_img, R_img, B_img, G_img, G_img), 0)
#                     img_gt = G_gt
               
#                 if color_channel == 2:
#                     img_lq = torch.cat((B_img, G_img, R_img, B_img, B_img), 0)
#                     img_gt = B_gt
                    
#                 img_gt = img_gt.repeat((5, 1, 1))
            
#             elif num_of_noisy == 3:
#                 if color_channel == 0:
#                     img_lq = torch.cat((R_img, B_img, G_img, B_img, R_img), 0)
#                     img_gt = R_gt
                                  
#                 if color_channel == 1:
#                     img_lq = torch.cat((G_img, R_img, B_img, R_img, G_img), 0)
#                     img_gt = G_gt
               
#                 if color_channel == 2:
#                     img_lq = torch.cat((B_img, G_img, R_img, G_img, B_img), 0)
#                     img_gt = B_gt
                    
#                 img_gt = img_gt.repeat((5, 1, 1))
            
#             elif num_of_noisy == 4:
#                 if color_channel == 0:
#                     img_lq = torch.cat((R_img, B_img, G_img, G_img, R_img), 0)
#                     img_gt = R_gt
                                  
#                 if color_channel == 1:
#                     img_lq = torch.cat((G_img, R_img, B_img, G_img, B_img), 0)
#                     img_gt = G_gt
               
#                 if color_channel == 2:
#                     img_lq = torch.cat((B_img, C_img, R_img, C_img, B_img), 0)
#                     img_gt = B_gt
                    
#                 img_gt = img_gt.repeat((5, 1, 1))
                
#             print('img_gt shape =', img_gt.shape)
#             print('img_lq shape =', img_lq.shape)
                                     
#Saw that from the beginning


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        
        #img_gt, img_lq = img2tensor([img_gt, img_lq],bgr2rgb=True,float32=True)


        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
#         print(self.paths)
        return len(self.paths)
