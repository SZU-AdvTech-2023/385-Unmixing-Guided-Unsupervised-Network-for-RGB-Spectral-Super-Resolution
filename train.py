import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import model
import layer
from load_data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from visdom import Visdom
import os


device = torch.device("cuda:0")

# 将窗口类实例化
viz = Visdom(port=1234)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

lr_G = 0.005  # 生成器学习率
lr_D = 0.0005  # 判别器学习率

batchsize = 1
# The number of endmembers of guidance hyperspectral image
endmember = 20
# The number of channels of hyperspectral images after super-resolution
band_hsi = 31  # dataset: ICVL
# band_hsi = 50  # dataset: DFC2018 Houston
# band_hsi = 54 #dataset: TG1HRSSC
band_RGB = 3

# imageFolder1:file address of guidance image
# imageFolder2:file address of hyperspectral image
# save_path:save path
imageFolder1 = 'guidance_data/ICVL/'
imageFolder2 = 'data/ICVL/traindata/'
save_path = 'save/ICVL/'

# save_path:save path
# imageFolder1 = 'guidance_data/DFC2018 Houston/'
# imageFolder2 = 'data/DFC2018 Houston/traindata/'
# save_path = 'save/DFC2018 Houston/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
# load spectral response function

# dataset: ICVL
SRF = sio.loadmat('SRF/P_N_V2.mat')
SRF = torch.from_numpy(SRF['P_20N'])
SRF = Variable(SRF.type(torch.FloatTensor).to(device))


# dataset: DFC2018 Houston
# SRF = sio.loadmat('SRF/SRF_50.mat')
# SRF = torch.from_numpy(SRF['SRF_50'])
# SRF = Variable(SRF.type(torch.FloatTensor).to(device))
# SRF = SRF.permute(1, 0)


# dataset: TG1HRSSC
# SRF = sio.loadmat('SRF/SRF_54.mat')
# SRF = torch.from_numpy(SRF['SRF_54'])
# SRF = Variable(SRF.type(torch.cuda.FloatTensor))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


# model
G = model.Network(endmember, band_hsi, band_RGB)
G = G.to(device)
G.apply(weights_init)

# loss & optimizer
L1 = nn.L1Loss()
BCE_loss = nn.BCELoss()
addition = layer.TVLoss()
criterionSparse = layer.SparseKLloss().to(device)
ba_loss = layer.Loss_before_after()
channel_loss = layer.Loss_ssim_hyper()
criterion_train = layer.LossTrainCSS2()

optimizer_G = torch.optim.Adam(G.gene.parameters(), lr=lr_G)
optimizer_D = torch.optim.Adam(G.dis.parameters(), lr=lr_D)

# load pretrained weights

# dataset: ICVL
# checkpoint_G = torch.load('pretrained_model/pretrained_model_ICVL.pth')
# dataset: DFC2018 Houston
# checkpoint_G = torch.load('pretrained_model/pretrained_model_DFC2018 Houston.pth')
# dataset: TG1HRSSC
# checkpoint_G = torch.load('pretrained_model/pretrained_model_TG1HRSSC.pth')

# G.gene.load_state_dict(checkpoint_G['model'])
# optimizer_G.load_state_dict(checkpoint_G['optimizer'])

# initialization of decoder1
# dataset: ICVL
edm = sio.loadmat('endmember/Salinas_corrected_endmember31.mat')
# dataset: DFC2018 Houston
# edm = sio.loadmat('endmember/Salinas_corrected_endmember50.mat')
# dataset: TG1HRSSC
# edm = sio.loadmat('endmember/Salinas_corrected_endmember54.mat')
edm = edm['E']
edm = torch.from_numpy(edm)
edm = edm.type(torch.FloatTensor).to(device)

layer_name = 'gene.decoder_hsi.layer.weight'


def _load_endmember(edm, layer_name):
    model_dict = G.state_dict()
    model_dict[layer_name] = edm
    G.load_state_dict(model_dict)


# _load_endmember(edm, layer_name)

# Load Dataset
dataset = LoadDataset(imageFolder1, imageFolder2)
data_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=0)


# learning rate decay
def LR_Decay(optimizer, n, lr):
    lr_d = lr * (0.7 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d


n = 0

start_time = time.time()

f = open(save_path + 'loss3.txt', 'w+')
f.write('The total loss is : \n\n\n')

for epoch in range(500):
    print('*' * 10, 'The {}th epoch for training.'.format(epoch + 1), '*' * 10)

    running_loss_pre = 0

    for iteration, Data in enumerate(data_loader, 1):

        # guidance image
        guidance_hsi = Data["guidance"]
        guidance_hsi = guidance_hsi.type(torch.FloatTensor).to(device)
        # hyperspectral image
        srhsi = Data["srhsi"]
        srhsi = srhsi.type(torch.FloatTensor).to(device)
        # generate rgb image
        # srhsi1 = srhsi.reshape(batchsize, srhsi.size()[1], -1)
        srhsi1 = srhsi.reshape(srhsi.size()[0], srhsi.size()[1], -1)
        rgb = torch.matmul(SRF, srhsi1)
        # rgb = rgb.reshape(batchsize, rgb.shape[1], srhsi.shape[2], srhsi.shape[3])
        rgb = rgb.reshape(rgb.shape[0], rgb.shape[1], srhsi.shape[2], srhsi.shape[3])

        Input_1 = Variable(guidance_hsi, requires_grad=False).type(torch.FloatTensor).to(device)
        Input_2 = Variable(rgb, requires_grad=False).type(torch.FloatTensor).to(device)

        abundance_hsi, abundance_RGB, hrhsi, RGB, pred_hrhsi, dis_hsi2, dis_pre2, pred_hrhsi_be, pred_hrhsi_af = G(
            Input_1.permute(0, 2, 3, 1),
            Input_2.permute(0, 2, 3, 1))
        abundance_hsi1 = abundance_hsi.reshape(abundance_hsi.size()[0],
                                               abundance_hsi.size()[1] * abundance_hsi.size()[2],
                                               abundance_hsi.size()[3])
        abundance_RGB1 = abundance_RGB.reshape(abundance_RGB.size()[0],
                                               abundance_RGB.size()[1] * abundance_RGB.size()[2],
                                               abundance_RGB.size()[3])
        # pred_rgb = torch.matmul(SRF, pred_hrhsi.reshape(batchsize, pred_hrhsi.shape[1], -1))
        pred_rgb = torch.matmul(SRF, pred_hrhsi.reshape(pred_hrhsi.shape[0], pred_hrhsi.shape[1], -1))
        # pred_rgb = pred_rgb.reshape(batchsize, pred_rgb.shape[1], pred_hrhsi.shape[2], pred_hrhsi.shape[3])
        pred_rgb = pred_rgb.reshape(pred_rgb.shape[0], pred_rgb.shape[1], pred_hrhsi.shape[2], pred_hrhsi.shape[3])

        # TVloss
        loss_tv = addition(pred_hrhsi)
        # sparse loss of hsi unmixing branch
        loss_sparse_hsi = criterionSparse(abundance_hsi)
        # sparse loss of rgb unmixing branch
        loss_sparse_RGB = criterionSparse(abundance_RGB)
        # reconstruction loss of hsi unmixing branch
        loss_euc_hsi = L1(hrhsi, Input_1)
        # reconstruction loss of rgb unmixing branch
        loss_euc_RGB = L1(RGB, Input_2)
        # reconstruction loss of reconstruction branch
        # loss_pre = L1(pred_rgb, rgb)
        loss_pre = criterion_train(pred_rgb, rgb)
        # reconstruction loss of berfore and after
        loss_ba = ba_loss(pred_hrhsi_be, pred_hrhsi_af)
        # reconstructed hsi adjacent channel similarity
        loss_ad = channel_loss(pred_hrhsi)
        #


        # Adversarial loss
        # y_real_ = torch.ones(batchsize, 1).to(device)
        # y_fake_ = torch.zeros(batchsize, 1).to(device)
        y_real_ = torch.ones(pred_rgb.shape[0], 1).to(device)
        y_fake_ = torch.zeros(pred_rgb.shape[0], 1).to(device)

        real_loss_d2 = BCE_loss(dis_hsi2, y_real_)
        fake_loss_d2 = BCE_loss(dis_pre2, y_fake_)
        fake_loss_g2 = BCE_loss(dis_pre2, y_real_)
        GP_loss2 = model.calc_gradient_penalty(G.dis.discriminator, Input_1.detach(), pred_hrhsi.detach(), center=0,
                                               alpha=None, LAMBDA=10, device=pred_hrhsi.device)
        loss_hsi = 0.01 * loss_sparse_hsi + loss_euc_hsi
        loss_RGB = 0.01 * loss_sparse_RGB + loss_euc_RGB

        loss_g = loss_hsi + loss_RGB + 5 * loss_pre + 0.5 * loss_tv + 1 * fake_loss_g2 + 0.5 * loss_ba + 0.5 * loss_ad
        loss_d2 = 3 * real_loss_d2 + 3 * fake_loss_d2 + 0.1 * GP_loss2

        f = open(save_path + 'loss3.txt', 'a')

        if epoch % 5 != 4:
            optimizer_G.zero_grad()
            loss_g.backward(retain_graph=True)
            optimizer_G.step()
        else:
            optimizer_D.zero_grad()
            loss_d2.backward(retain_graph=True)
            optimizer_D.step()

        running_loss_pre += loss_pre.data.cpu()
    print(
        "Epoch [%02d], Iter[%06d], Train Loss: %.9f"
        % (epoch, iteration, running_loss_pre))
    # 更新监听的信息
    viz.line(X=[epoch], Y=[running_loss_pre.item()], name='running_loss_pre', win='lin1', update='append', opts={
             'showlegend': True,  # 显示网格
             'title': "running_loss_pre",
             'xlabel': "epoch",  # x轴标签
             'ylabel': "running loss pre",  # y轴标签
         })


    f = open(save_path + 'loss3.txt', 'a')
    f.write('epoch {} The loss_pre is  {:.4f}.\n'.format(epoch, running_loss_pre))

    if epoch % 30 == 0:
        LR_Decay(optimizer_G, n, lr_G)
        LR_Decay(optimizer_D, n, lr_D)
        n += 1

    if epoch % 20 == 0:
        state = {'model': G.gene.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
        torch.save(state, save_path + 'newmodel_' + str(int(epoch)) + '.pth')

state = {'model': G.gene.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
torch.save(state, save_path + 'newmodel_' + str(int(epoch)) + '.pth')

T = time.time() - start_time

print('Total training time is {}'.format(T))
f.write('Total training time is {}.\n'.format(T))

f.close()
