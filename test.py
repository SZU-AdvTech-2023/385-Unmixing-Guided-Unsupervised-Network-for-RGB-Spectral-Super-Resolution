import time
import torch
import torch.nn as nn
import numpy as np
from math import exp
import torch.nn.functional as F
import scipy.io as sio
import model
import layer
from load_data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from torch.nn.functional import upsample
import skimage.measure as skm
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import hdf5storage


def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()  # .resize_(C, H*W)
    Ifake = im_fake.clone()  # .resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)  # .resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


random_seed = 7
torch.manual_seed(random_seed)

# dataset: ICVL
SRF = sio.loadmat('SRF/P_N_V2.mat')
SRF = torch.from_numpy(SRF['P_20N'])
SRF = Variable(SRF.type(torch.FloatTensor))

# dataset: DFC2018 Houston
# SRF = sio.loadmat('SRF/SRF_50.mat')
# SRF = torch.from_numpy(SRF['SRF_50'])
# SRF = Variable(SRF.type(torch.FloatTensor))
# SRF = SRF.permute(1,0)

# dataset: TG1HRSSC
# SRF = sio.loadmat('SRF/SRF_54.mat')
# SRF = torch.from_numpy(SRF['SRF_54'])
# SRF = Variable(SRF.type(torch.FloatTensor))

PSNR = 0
SSIM = 0
SAM = 0

endmember = 20
band_hsi = 31  # dataset: ICVL
# band_hsi = 50 #dataset: DFC2018 Houston
# band_hsi = 54 #dataset: TG1HRSSC
band_RGB = 3

# load_path = 'data/ICVL/'
load_path = 'data/ICVL/testdata/'
# load_path = 'data/DFC2018 Houston/testdata/'
# load_path = 'data/DFC2018 Houston/Originaldata/'
save_path = 'save/ICVL/'
# save_path = 'save/DFC2018 Houston/'

f = open(save_path + 'test3.txt', 'w+')
f.write('The total loss is : \n\n\n')

net = model.Generator(endmember, band_hsi, band_RGB).eval()

testname = os.listdir(load_path)

# dataset: ICVL
# checkpoint_G = torch.load('pretrained_model/pretrained_model_ICVL.pth')
checkpoint_G = torch.load('save/ICVL/newmodel_499.pth')
# dataset: DFC2018 Houston
# checkpoint_G = torch.load('pretrained_model/pretrained_model_DFC2018 Houston.pth')
# checkpoint_G = torch.load('save/DFC2018 Houston/newmodel_399.pth')
# checkpoint_G = torch.load('save/DFC2018 Houston/model_499.pth')
# dataset: TG1HRSSC
# checkpoint_G = torch.load('pretrained_model/pretrained_model_TG1HRSSC.pth')
net.load_state_dict(checkpoint_G['model'])

for i in range(len(testname)):

    # img2 = sio.loadmat(load_path+testname[i])
    img2 = h5py.File(load_path + testname[i], 'r')
    img2 = img2['rad'][:] / 1.0
    # img2 = img2['houstonU'][:] / 1.0
    img2 = img2.astype(float)
    # img2 = sio.loadmat(load_path+testname[i])
    # img2 = img2['data']/1.0
    img2 = torch.from_numpy(img2)
    img2 = img2.type(torch.FloatTensor)
    img2 = img2 / torch.max(img2)
    # img2 = img2.permute(2,0,1)

    img2 = img2.unsqueeze(0)
    rgb1 = img2.reshape(1, img2.size()[1], -1)
    rgb1 = torch.matmul(SRF, rgb1)
    rgb = rgb1.reshape(1, rgb1.shape[1], img2.shape[2], img2.shape[3])
    rgb2 = np.array(rgb.squeeze(0).numpy())



    img1 = sio.loadmat('guidance_data/ICVL/Salinas_corrected_31.mat')  # dataset: ICVL
    # img1 = sio.loadmat('guidance_data/DFC2018 Houston/Salinas_corrected_50.mat') #dataset: DFC2018 Houston
    # img1 = sio.loadmat('guidance_data/ICVL/Salinas_corrected_54.mat') #dataset: TG1HRSSC
    img1 = img1['guidance_image'] / 1.0
    img1 = torch.from_numpy(img1)
    img1 = img1.type(torch.FloatTensor)
    img1 = img1.permute(2, 0, 1)
    img1 = img1 / torch.max(img1)
    img1 = img1.unsqueeze(0)

    Input1 = Variable(img1, requires_grad=False).type(torch.FloatTensor)
    Input2 = Variable(rgb, requires_grad=False).type(torch.FloatTensor)

    abundance_hsi, abundance_RGB, hrhsi1, RGB1, pred_hrhsi, pred_hrhsi_be, pred_hrhsi_af = net(Input1.permute(0, 2, 3, 1), Input2.permute(0, 2, 3, 1))
    psnr_g = []

    img2 = img2.squeeze()
    img2 = np.array(img2)
    pred_hrhsi = pred_hrhsi.squeeze()
    pred_hrhsi = np.array(pred_hrhsi.detach().numpy())

    text = testname[i].split(".")
    save_path_result = 'save/ICVL/result3/'
    # save_path_result = 'save/DFC2018 Houston/result3/'
    if not os.path.exists(save_path_result):
        os.makedirs(save_path_result)
    sio.savemat(save_path_result + text[0] + '_result3.mat', {'result': pred_hrhsi})
    # sio.savemat(save_path_result + text[0] + '_rgb.mat', {'data': rgb2})

    for j in range(band_hsi):
        psnr_g.append(compare_psnr(img2[j, :, :], pred_hrhsi[j, :, :]))
    psnr = np.mean(np.array(psnr_g))

    fout_0 = np.transpose(img2, (1, 2, 0))
    hsi_g_0 = np.transpose(pred_hrhsi, (1, 2, 0))
    ssim = compare_ssim(im1=fout_0, im2=hsi_g_0, K1=0.01, K2=0.03, multichannel=True, data_range=1)

    sam = SAM_GPU(255 * torch.from_numpy(img2), 255 * torch.from_numpy(pred_hrhsi))

    print('PNSR = {0:.8f}'.format(psnr))
    print('ssim = {0:.8f}'.format(ssim))
    print('SAM = {0:.8f}\n'.format(sam))
    PSNR = PSNR + psnr
    SAM = SAM + sam
    SSIM = SSIM + ssim

    f.write('The PSNR is {:.4f}\n SSIM is {:.4f}\n SAM is {:.4f}\n\n'.format(psnr, ssim, sam))

print('PNSR = {0:.8f}'.format(PSNR / len(testname)))
print('ssim = {0:.8f}'.format(SSIM / len(testname)))
print('SAM = {0:.8f}\n'.format(SAM / len(testname)))
f.write('The PSNR is {:.4f}\n SSIM is {:.4f}\n SAM is {:.4f}\n\n'.format(PSNR / len(testname), SSIM / len(testname),
                                                                         SAM / len(testname)))
f.close()
