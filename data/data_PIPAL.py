import os
import torch
import numpy as np
import cv2
from ctypes import c_double
import numpy as np
from scipy.special import gamma


class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_file_name, transform, train_mode, scene_list, train_size=0.8):
        super(IQADataset, self).__init__()
        
        self.db_path = db_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size
        
        self.data_dict = IQADatalist(
            txt_file_name=self.txt_file_name,
            train_mode=self.train_mode,
            scene_list=self.scene_list,
            train_size=self.train_size
        ).load_data_dict()
        self.n_images = len(self.data_dict['d_img_list'])
    
    def __len__(self):
        return self.n_images

    def preprocess_image(self,img):
        if isinstance(img, str):
            if os.path.exists(img):
                return cv2.imread(img, 0).astype(np.float64)
            else:
                raise FileNotFoundError('The image is not found on your system.')
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                image = img
            elif len(img.shape) == 3:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError('The image shape is not correct.')

            return image.astype(np.float64)
        else:
            raise ValueError('You can only pass image to the constructor.')

    def _estimate_ggd_param(self,vec):
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))
        sigma_sq = np.mean(vec ** 2)
        sigma = np.sqrt(sigma_sq)
        E = np.mean(np.abs(vec))
        rho = sigma_sq / E ** 2
        differences = abs(rho - r_gam)
        array_position = np.argmin(differences)
        gamparam = gam[array_position]
        return gamparam, sigma

    def _estimate_aggd_param(self,vec):
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = ((gamma(2.0 / gam)) ** 2) / (gamma(1.0 / gam) * gamma(3.0 / gam))
        left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
        right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))
        gamma_hat = left_std / right_std
        rhat = (np.mean(np.abs(vec))) ** 2 / np.mean((vec) ** 2)
        rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)
        differences = (r_gam - rhat_norm) ** 2
        array_position = np.argmin(differences)
        alpha = gam[array_position]
        return alpha, left_std, right_std

    def get_feature(self,img):
        imdist = self.preprocess_image(img)
        scale_num = 2
        feat = np.array([])
        for itr_scale in range(scale_num):
            mu = cv2.GaussianBlur(imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            sigma = np.sqrt(abs((sigma - mu_sq)))
            structdis = (imdist - mu) / (sigma + 1)
            alpha, overallstd = self._estimate_ggd_param(structdis)
            feat = np.append(feat, [alpha, overallstd ** 2])
            shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]
            for shift in shifts:
                shifted_structdis = np.roll(np.roll(structdis, shift[0], axis=0), shift[1], axis=1)
                pair = np.ravel(structdis, order='F') * np.ravel(shifted_structdis, order='F')
                alpha, left_std, right_std = self._estimate_aggd_param(pair)
                const = np.sqrt(gamma(1 / alpha)) / np.sqrt(gamma(3 / alpha))
                mean_param = (right_std - left_std) * (gamma(2 / alpha) / gamma(1 / alpha)) * const
                feat = np.append(feat, [alpha, mean_param, left_std ** 2, right_std ** 2])
            imdist = cv2.resize(imdist, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        return feat
    def __getitem__(self, idx):
        # r_img: H x W x C -> C x H x W
        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = cv2.imread(os.path.join((self.db_path + "/Reference"), r_img_name), cv2.IMREAD_COLOR)
        mscn_r_img= self.get_feature(r_img)

        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = np.transpose(r_img, (2, 0, 1))

        # d_img: H x W x C -> C x H x W
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.db_path, d_img_name), cv2.IMREAD_COLOR)
        mscn_d_img = self.get_feature(d_img)

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]

        sample = {'r_img': r_img, 'd_img': d_img, 'score': score, 'mscn_r_img': mscn_r_img, 'mscn_d_img': mscn_d_img}
        if self.transform:
            sample = self.transform(sample)
        return sample


class IQADatalist():
    def __init__(self, txt_file_name, train_mode, scene_list, train_size=1):
        self.txt_file_name = txt_file_name
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list
        
    def load_data_dict(self):
        scn_idx_list, r_img_list, d_img_list, score_list = [], [], [], []

        # list append
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                scn_idx, ref, dis, score = line.split()                
                scn_idx = int(scn_idx)
                score = float(score)
                
                scene_list = self.scene_list

                # add items according to scene number
                if scn_idx in scene_list:
                    scn_idx_list.append(scn_idx)
                    r_img_list.append(ref)
                    d_img_list.append(dis)
                    score_list.append(score)
        # reshape score_list (1xn -> nx1)
        score_list = np.array(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'r_img_list': r_img_list, 'd_img_list': d_img_list, 'score_list': score_list}
        return data_dict
