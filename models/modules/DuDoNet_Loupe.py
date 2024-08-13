import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_util import *
import functools
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.
    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=(-2,-1)):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.
    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)


def real_to_complex(img):
    if len(img.shape)==3:
        data = img.unsqueeze(0)
    else:
        data = img
    y = torch.fft.fftn(data, dim=(-2,-1))
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y_complex = torch.cat([y.real, y.imag], 1)  ## (1,2,h,w)
    if len(img.shape)==3:
        y_complex = y_complex[0]
    return y_complex

def complex_to_real(data):
    if len(data.shape)==3:
        data1 = data.unsqueeze(0)
    else:
        data1 = data
    h, w = data.shape[-2], data.shape[-1]
    y_real, y_imag = torch.chunk(data1, 2, dim=1)
    y = torch.complex(y_real, y_imag)
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y = torch.fft.ifftn(y,s=(h,w),dim=(-2,-1)).abs()
    if len(data.shape)==3:
        y = y[0]
    return y


#  This part was built based on LOUPE: https://github.com/cagladbahadir/LOUPE/.
class LoupeLayer(nn.Module):
    def __init__(self, image_dims=[240, 240], pmask_slope=5, sample_slope=12, sparsity=0.05, hard=False, mode="relax"):
        super().__init__()

        self.image_dims = image_dims
        self.pmask_slope = pmask_slope  
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.mode = mode
        self.eps = 0.01
        self.hard = hard

        # Mask Initial
        self.pmask = nn.Parameter(torch.FloatTensor(*self.image_dims))  # Mask is same dimension as image plus complex domain
        self.pmask.requires_grad = True
        self.pmask.data.uniform_(self.eps, 1 - self.eps)
        self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope

    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope * mask)

    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (r <= 1).float()
        return le * mask * r + (1 - le) * (1 - (1 - mask) * beta)

    def sigmoid_beta(self, mask):
        random_uniform = torch.empty(*self.image_dims).uniform_(0, 1).cuda()
        return torch.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self):

        # initialize and squash
        probmask = self.squash_mask(self.pmask)

        # Sparsify, control the sampling ratio
        sparse_mask = self.sparsify(probmask)

        # generate soft mask
        mask = self.sigmoid_beta(sparse_mask)

        # return soft mask
        return mask


class DCN_Align(nn.Module):
    def __init__(self, nf=32, groups=4):
        super(DCN_Align, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True) 
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1    
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv4_2 = nn.Conv2d(nf, 32, 3, 1, 1, bias=True)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                            deformable_groups=4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        offset = torch.cat([fea1, fea2], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset = self.offset_conv4_2(offset)
        # DCN warping
        aligned_fea = self.dcnpack(fea2, base_offset)

        return aligned_fea



class DuDoNet_Loupe(nn.Module):
    def __init__(self, opt):
        super(DuDoNet_Loupe, self).__init__()

        self.image_channel = opt['c_image']  

        self.channel_in = opt['nf']   # c_i, default=32
        self.channel_fea = opt['nf']  # c_f, default=32
        self.iter_num = opt['stages']   # iteration stages T, default=4

        image_size = opt['image_size']  
        acc_ratio = opt['sparsity']  
        
        # learnable mask from Loupe
        self.LoupeLayer = LoupeLayer(image_dims=image_size, sparsity=acc_ratio)
   
        # variable initialization
        basic_block1 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.init_x = make_layer(basic_block1, 10)
        
        basic_block2 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.map_y = make_layer(basic_block2, 10)

        ## DAS module for spatial transformation
        self.dcn_align = DCN_Align(nf=32, groups=4)
        basic_block3 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.extract_y = make_layer(basic_block3, 10)

        ## convolutional layers for feature transformation
        # map x to feature domain 
        self.trans_E_x = nn.Sequential(*[nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                   nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2)])
        # map x feature to image domain
        self.trans_D_x = nn.Sequential(*[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                   nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_in, kernel_size=3, padding=3 // 2)])
        # map y to feature domain                           
        self.trans_E_y = nn.Sequential(*[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                    nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=3 // 2), \
                                   nn.ReLU(), \
                                   nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_in, kernel_size=3, padding=3 // 2)])


        ## proximal networks
        basic_block_x = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_x = make_layer(basic_block_x, 12)
        basic_block_img = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_k_img = make_layer(basic_block_img, 12)
        basic_block_real = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_k_real = make_layer(basic_block_real, 12)

        ## hyper-parameters
        self.mu_x = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.mu_k = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.alpha = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.beta = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]

        ## Weighted Average Layer (WAL): map C_i images to single image
        self.WAL = nn.Conv2d(in_channels=self.channel_in*2, out_channels=self.image_channel, kernel_size=3, padding=3 // 2)
        
        
    def data_consistency_layer(self, generated, X_k, mask):

        gene_complex = real_to_complex(generated)
        output_complex = X_k + gene_complex * (1.0 - mask)
        output_img = complex_to_real(output_complex)
        
        return output_img
    
    def forward(self, x, y, mask=None):

        # x: target image (b,1,h,w)
        # y: fully-sample reference image (b,1,h,w)
        '''
        # predifined mask with size (1,2,h,w)
        # if mask == None, x would be fully sampled target image and we should learn mask using Loupe. 
        The undersampled target image will generate using the learned mask.
        # if mask != None, we use the predefined mask. 
        x would be undersampled target image using the predefined mask.
        '''

        # CE
        x = x.repeat(1,self.channel_in,1,1)
        y = y.repeat(1,self.channel_in,1,1)

        # for Loupe, generate mask and get the corresponding under-sampled image
        if mask == None:  
            b,c,h,w = x.shape
            x_k = real_to_complex(x)
            ## generate mask 
            mask = self.LoupeLayer()
            # If learned 1D random mask, we should expand 1D random mask to 2D
            if mask.shape[0] != mask.shape[1]:  
                mask = mask.unsqueeze(0).unsqueeze(0).repeat(b,1,h,1) 
            else:
                mask = mask.unsqueeze(0).unsqueeze(0).repeat(b,1,1,1)
            # Undersample
            x_k_under = x_k*mask
            # iFFT into image space
            x_under = complex_to_real(x_k_under)
        
        # for predefined mask
        else:  
            mask = mask.repeat(1,self.channel_in,1,1)
            x_under = x
            x_k_under = real_to_complex(x)
            
        # initialize variables
        x_t0 = self.init_x(x_under)
        K_t0 = real_to_complex(x_t0)
        # extract feature for reference
        y = self.map_y(y)

        for i in range(self.iter_num):
            
            ########### update k ##########
            temp_k = K_t0 - self.mu_k[i]*(self.beta[i]*(K_t0-real_to_complex(x_t0)))
            K_t1 = torch.cat([self.prox_k_real(temp_k[:,:32]),self.prox_k_img(temp_k[:,32:])], 1)
            K_t0 = K_t1

            ########## update x ###########
            # DC layer
            DC_x = self.data_consistency_layer(x_t0, x_k_under, mask)  ## DC
            # refine image
            refine_image = -self.alpha[i]*self.trans_D_x(self.trans_E_x(x_t0)-self.trans_E_y(self.dcn_align(x_t0, y)))
            # refine k-space 
            refine_k = complex_to_real(K_t0) - x_t0
            # proximal net for x 
            x_t1 = self.prox_x(DC_x + self.mu_x[i]*(refine_image + refine_k)) # update x    
            x_t0 = x_t1
            # extract deep feature for y
            y = self.extract_y(y)
 
        # reconstruct final image
        x_out = self.WAL(torch.cat([x_t0, complex_to_real(K_t0)],1))

        # return x_out, mask
        return x_out
