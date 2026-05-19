import os, time, sys, gc, cv2, logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.data_io import *
from datasets.dtu_eval import DTUDataset
from datasets.tnt import TNTDataset
from datasets.eth3d import ETHDataset
from datasets.intermediate import INDataset
from datasets.advanced import ADDataset

from models.mvsnet import MVSNet
from models.utils import *
from models.utils.opts import get_opts


cudnn.benchmark = True

args = get_opts()


def build_final_confidence(confidence_list):
    final_h, final_w = confidence_list[-1].shape[-2:]
    final_confidence = np.ones((final_h, final_w), dtype=np.float32)
    for confidence in confidence_list:
        if confidence.shape[-2:] != (final_h, final_w):
            confidence = cv2.resize(confidence, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
        final_confidence *= confidence
    return final_confidence


def test():
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            # @Note MVSNet main
            outputs = model(
                sample_cuda["imgs"], 
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"], 
                sample["filename"]
            )

            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(args.levels-2)].numpy()
            imgs = sample["imgs"]
            logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))


            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]):
                img = img[0].numpy()    # ref view
                cam = cam[0]            # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                depth_img_filename = os.path.join(args.outdir, filenames[0].format('depth_img', '.png'))
                os.makedirs(depth_img_filename.rsplit('/', 1)[0], exist_ok=True)
                plt.imsave(depth_img_filename, depth_est, cmap="rainbow", vmin=depth_est.min(), vmax=depth_est.max())
                # save confidence maps
                confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1, args.levels+1)]
                first_confidence_filename = os.path.join(args.outdir, filename.format('confidence', '_stage1.pfm'))
                save_pfm(first_confidence_filename, confidence_list[0])
                save_pfm(confidence_filename, build_final_confidence(confidence_list))

                # save cams, img
                write_cam(cam_filename, cam)
                img = np.transpose(img, (1, 2, 0))
                img = image_net_center_inv(img)
                img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(img_filename, img)

    torch.cuda.empty_cache()
    gc.collect()
    return total_time, len(TestImgLoader)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

    if args.which_dataset == 'tnt':
        logfile = os.path.join(args.logdir, 'TNT-test-' + curTime + '.log')
    else:
        logfile = os.path.join(args.logdir, 'test-' + curTime + '.log')
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if not args.nolog:
        fileHandler = logging.FileHandler(logfile, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    # @Note MVSNet model
    model = MVSNet(
        levels=args.levels, 
        hypo_plane_num_stages=[int(n) for n in args.hypo_plane_num_stages.split(",")], 
        depth_interval_ratio_stages=[float(ir) for ir in args.depth_interval_ratio_stages.split(",")],
        feat_base_channel=args.feat_base_channel, 
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=[int(n) for n in args.group_cor_dim_stages.split(",")],
        qkv_dim_stages=[int(n) for n in args.qkv_dim_stages.split(",")],
        heads_stages=[int(n) for n in args.heads_stages.split(",")],
    )
    print(args.loadckpt)
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)

    model.cuda()
    model.eval()

    # dataset, dataloader
    if args.which_dataset == 'dtu':
        test_dataset = DTUDataset(args.testpath, [args.testlist], args.n_views, img_wh=args.img_wh)
    elif args.which_dataset == 'tnt':
        test_dataset = TNTDataset(args.testpath, args.testlist, split=args.split, n_views=args.n_views, img_wh=(-1, 1024), cam_mode=args.cam_mode, img_mode=args.img_mode)
    elif args.which_dataset == 'eth':
        test_dataset = ETHDataset(args.testpath, [args.testlist], n_views=args.n_views, img_wh=args.img_wh, img_mode=args.img_mode)
    elif args.which_dataset == 'intermediate':
        test_dataset = INDataset(args.testpath, [args.testlist], n_views=args.n_views, img_wh=args.img_wh, cam_mode=args.cam_mode, img_mode=args.img_mode, depth_interval=args.depth_interval)
    elif args.which_dataset == 'advanced':
        test_dataset = ADDataset(args.testpath, [args.testlist], n_views=args.n_views, img_wh=args.img_wh, img_mode=args.img_mode)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    test()
