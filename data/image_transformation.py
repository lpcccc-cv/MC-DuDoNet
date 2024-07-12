import cv2
import os
import numpy as np
import random
 
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    rotated_img = cv2.warpAffine(image, M, (w, h))

    return rotated_img

def move(image, m_h, m_w):
    (h, w) = image.shape[:2]
    M = np.float32([[1,0,m_w],[0,1,m_h]])
    moved_img = cv2.warpAffine(image, M, (h,w))

    return moved_img

def tranform(img):
    params = []
    ## 对reference进行非对齐变换
    angle = 5
    shift_h = random.uniform(-5,5)
    shift_w = random.uniform(-5,5)
    if random.random()<0.5:
        img = rotate_bound(img, angle)
        params.append(str(angle))
    else:
        params.append(str(0))
    if random.random()<0.5:
        img = move(img, 0, shift_h)
        params.append(str(shift_h))
    else:
        params.append(str(0))
    if random.random()<0.5:
        img = move(img, shift_w, 0)
        params.append(str(shift_w))
    else:
        params.append(str(0))
    return img, params


img_file = '/home/lpc/dataset/BrainTS/MCSR/T1_test'
save_file = '/home/lpc/dataset/BrainTS/MCSR/T1_noalign_test'
file_name_list = sorted(os.listdir(img_file))

for name in file_name_list:
    img = cv2.imread(os.path.join(img_file, name), cv2.IMREAD_UNCHANGED)
    img, params = tranform(img)
    # 保存图片
    cv2.imwrite(os.path.join(save_file, name), img)
    print(name)
    print(params)


    
