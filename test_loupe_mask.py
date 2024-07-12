import torch
from utils import util
from tqdm import tqdm
import cv2, os
import models.modules.DuDoNet as DuDoNet
import models.modules.DuDoNet_Loupe as DuDoNet_Loupe
import models.modules.UNet_loupe as UNet_loupe
import models.modules.MDUNet_loupe as MDUNet_loupe
import time

def main():
    mode = 'IXI'
    dataset_opt = {}
    dataset_opt['task'] = 'rec'
    dataset_opt['scale'] = 4
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
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))

    # creat model   
    model_path = '~/IXI_DuDoN_x10_joint/models/Iter_~.pth'
    model = DuDoNet_Loupe.DuDoNet_Loupe().cuda()

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
                sr_img_1, mask = model(im1_gt, im2_gt)
                # print(mask.shape)
                sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
                im1_gt = im1_gt[0,0].cpu().detach().numpy()*255.

                # save your mask
                mask_soft = mask.data.cpu().detach().numpy()*255.
                mask_hard = torch.where(mask>=0.5, 1, 0)
                mask_point = torch.sum(mask_hard)
                mask = mask_hard.data.cpu().detach().numpy()*255.
              
                cv2.imwrite('./Loupe_mask/2d_mask_IXI_soft.png', mask_soft)
                cv2.imwrite('./Loupe_mask/2d_mask_IXI_hard.png', mask)
            
if __name__ == '__main__':
    main()
