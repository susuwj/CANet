import os
import cv2
import numpy as np

from torch.utils.data import Dataset

from datasets.data_io import *


class ETHDataset(Dataset):
    def __init__(self, root_dir, list_file, n_views, **kwargs):
        super(ETHDataset, self).__init__()

        self.root_dir = root_dir
        self.list_file = list_file
        self.n_views = n_views

        self.img_wh = kwargs.get("img_wh", (2432,1600))
        self.img_wh = [int(v) for v in self.img_wh.split(',')]
        self.metas = self.build_metas()


    def build_metas(self):
        metas = []

        scans = self.list_file

        for scan in scans:
            with open(os.path.join(self.root_dir, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        metas += [(scan, -1, ref_view, src_views)]
        return metas

   
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        return intrinsics, extrinsics, depth_min, depth_max


    def read_img(self, filename):
        #img = Image.open(filename)
        #np_img = np.array(img, dtype=np.float32) / 255.
        img = cv2.imread(filename)
        img = image_net_center(img)
        return img


    def scale_mvs_input(self, intrinsics, img):
        height, width = img.shape[:2]

        max_w, max_h = self.img_wh[0], self.img_wh[1]

        img = cv2.resize(img, (max_w, max_h))

        scale_w = 1.0 * max_w / width
        intrinsics[0, :] *= scale_w
        scale_h = 1.0 * max_h / height
        intrinsics[1, :] *= scale_h

        return intrinsics, img


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        scan, _, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.root_dir, scan, f'cams/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            intrinsics, img = self.scale_mvs_input(intrinsics, img)
            imgs.append(img.transpose(2,0,1))

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)

            intrinsics[:2,:] *= 0.125
            proj_mat_0[0,:4,:4] = extrinsics.copy()
            proj_mat_0[1,:3,:3] = intrinsics.copy()
            int_mat_0 = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_1[0,:4,:4] = extrinsics.copy()
            proj_mat_1[1,:3,:3] = intrinsics.copy()
            int_mat_1 = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_2[0,:4,:4] = extrinsics.copy()
            proj_mat_2[1,:3,:3] = intrinsics.copy()
            int_mat_2 = intrinsics.copy()


            proj_matrices_0.append(proj_mat_0)
            proj_matrices_1.append(proj_mat_1)
            proj_matrices_2.append(proj_mat_2)

            # reference view
            if i == 0:
                depth_min =  depth_min_
                depth_max = depth_max_

        proj={}
        proj['stage1'] = np.stack(proj_matrices_0)
        proj['stage2'] = np.stack(proj_matrices_1)
        proj['stage3'] = np.stack(proj_matrices_2)

        intrinsics_matrices = {
            "stage1": int_mat_0,
            "stage2": int_mat_1,
            "stage3": int_mat_2
        }

        sample = {
            "imgs": imgs,
            "proj_matrices": proj,
            "intrinsics_matrices": intrinsics_matrices,
            "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        }

        return sample
