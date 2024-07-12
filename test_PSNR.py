import torch
from utils import util
from tqdm import tqdm
import cv2, os
import models.modules.DuDoNet_Loupe as DuDoNet_Loupe
import time
import numpy as np
import matplotlib.pyplot as plt


def main():
    mode = 'IXI'
    dataset_opt = {}
    dataset_opt['task'] = 'rec'
    dataset_opt['scale'] = 10
    dataset_opt['hr_in'] = True
    dataset_opt['crop_size'] = 0
    dataset_opt['test_size'] = 256
    data1 = True
    joint_rec = False
    save_result = True
    #### create train and val dataloader
    if mode == 'IXI':
        from data.IXI_dataset import IXI_train as D
        dataset_opt['dataroot_GT'] = '~/dataset/IXI/MC_MRI/test/T2'
    elif mode == 'brain':
        from data.brain_dataset import brain_train as D
        dataset_opt['dataroot_GT'] = '~/dataset/BrainTS/MCSR/T2_test'
    elif mode == 'fastmri':
        from data.fastmri_dataset import SliceDataset as D
        dataset_opt['dataroot_GT'] = '~/dataset/fastmri/test/PDFS'
    
    val_set = D(dataset_opt, train=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=1,pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))

    model_path = '~/experiments/brain_DuDoNet_x10_retrain/models/~.pth'
    model = DuDoNet_Loupe.DuDoNet_Loupe().cuda()
    save_path = '~'

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        #### validation
        
        avg_psnr_im1 = 0.0
        avg_ssim_im1 = 0.0
        avg_rmse_im1 = 0.0
        idx = 0
        time_begin = time.time()
        for i,val_data in enumerate(tqdm(val_loader)): 
            im1_lr = val_data['im1_LQ'].cuda()
            im1_gt = val_data['im1_GT'].cuda()
            im2_lr = val_data['im2_LQ'].cuda()
            im2_gt = val_data['im2_GT'].cuda()
            mask = val_data['mask'].cuda()

            if data1:
                sr_img_1 = model(im1_lr, im2_gt, mask)
                sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
                im1_gt = im1_gt[0,0].cpu().detach().numpy()*255.
            
                # calculate PSNR
                cur_psnr_im1 = util.calculate_psnr(sr_img_1, im1_gt)
                avg_psnr_im1 += cur_psnr_im1
                cur_ssim_im1 = util.calculate_ssim(sr_img_1, im1_gt)
                avg_ssim_im1 += cur_ssim_im1
                cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt)
                avg_rmse_im1 += cur_rmse_im1

            # save image
            save_path_1 = save_path
            if not os.path.exists(save_path_1):
                os.makedirs(save_path_1)
            if save_result:
                cv2.imwrite(os.path.join(save_path_1, '{:08d}.png'.format(i+1)), sr_img_1)

            idx += 1  
            time_end = time.time()  
        avg_psnr_im1 = avg_psnr_im1 / idx
        avg_ssim_im1 = avg_ssim_im1 / idx
        avg_rmse_im1 = avg_rmse_im1 / idx

        # log
        print("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
        print("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
        print("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))
        print('Total time:', time_end-time_begin)


if __name__ == '__main__':
    main()
