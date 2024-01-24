import random
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:6")


class encoder_hr_hsi(nn.Module):
    def __init__(self, endmember, band_hsi):
        super(encoder_hr_hsi, self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.layer1 = nn.Linear(self.band_hsi, 2 * self.band_hsi)
        self.layer2 = nn.Linear(2 * self.band_hsi, self.band_hsi)
        self.layer3 = nn.Linear(self.band_hsi, int(self.band_hsi / 2))
        self.layer4 = nn.Linear(int(self.band_hsi / 2), self.endmember)
        self.relu = nn.ReLU()

        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(self.band_hsi, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(self.band_hsi, 64, 5, 1, 2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv13 = nn.Sequential(
        #     nn.Conv2d(self.band_hsi, 64, 7, 1, 3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv1 = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        #
        # self.conv2 = nn.Conv2d(64, self.endmember, 3, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(self.endmember)

    def forward(self, x):
        self.out1 = self.relu(self.layer1(x))
        self.out2 = self.relu(self.layer2(self.out1))
        self.out3 = self.relu(self.layer3(self.out2))
        self.out4 = self.relu(self.layer4(self.out3))
        self.out = F.softmax(self.out4, dim=3)

        # x = x.permute(0, 3, 1, 2)
        # b, c, h, w = x.size()
        #
        # x_add1 = self.conv11(x)
        #
        # x_add2 = self.conv12(x)
        #
        # x_add3 = self.conv13(x)
        #
        # dim = x_add3.shape[1]
        #
        # num1 = x_add1.shape[1] // dim
        # num2 = x_add2.shape[1] // dim
        # num3 = x_add3.shape[1] // dim
        # x_out = torch.empty(x.shape[0], dim, h, w).to(device=x.device)
        # x_out = torch.empty(x.shape[0], dim, h, w)
        # for i in range(dim):
        #     x1_tmp = x_add1[:, i * num1:(i + 1) * num1, :, :]
        #     x2_tmp = x_add2[:, i * num2:(i + 1) * num2, :, :]
        #     x3_tmp = x_add3[:, i * num3:(i + 1) * num3, :, :]
        #
        #     x_tmp = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
        #     addout = x1_tmp + x2_tmp + x3_tmp
        #     avgout = torch.mean(x_tmp, dim=1, keepdim=True)
        #     maxout, _ = torch.max(x_tmp, dim=1, keepdim=True)
        #     x_tmp = torch.cat([addout, avgout, maxout], dim=1)
        #     x_tmp = self.conv1(x_tmp)
        #     x_out[:, i:i + 1, :, :] = x_tmp
        # self.out = self.bn(self.conv2(x_out))
        # self.out = self.out.permute(0, 2, 3, 1)
        # self.out = F.softmax(self.out, dim=3)
        return self.out


class decoder_hsi(nn.Module):
    def __init__(self, endmember, band_hsi):
        super(decoder_hsi, self).__init__()
        self.endmember = endmember
        self.band_hsi = band_hsi
        self.layer = nn.Linear(self.endmember, self.band_hsi, bias=False)
        # self.layer = nn.Conv2d(self.endmember, self.band_hsi, 1)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        self.out = self.layer(x)
        # self.out = self.out.permute(0, 2, 3, 1)
        return self.out


class encoder_RGB(nn.Module):
    def __init__(self, endmember, band_RGB):
        super(encoder_RGB, self).__init__()
        self.endmember = endmember
        self.band_RGB = band_RGB
        self.layer1 = nn.Linear(self.band_RGB, 2 * self.band_RGB)
        self.layer2 = nn.Linear(2 * self.band_RGB, 4 * self.band_RGB)
        self.layer3 = nn.Linear(4 * self.band_RGB, 8 * self.band_RGB)
        self.layer4 = nn.Linear(8 * self.band_RGB, self.endmember)
        self.relu = nn.ReLU()

        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(self.band_RGB, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(self.band_RGB, 64, 5, 1, 2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv13 = nn.Sequential(
        #     nn.Conv2d(self.band_RGB, 64, 7, 1, 3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),  # No effect on order
        # )
        #
        # self.conv1 = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        #
        # self.conv2 = nn.Conv2d(64, self.endmember, 3, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(self.endmember)

    def forward(self, x):
        self.out1 = self.relu(self.layer1(x))
        self.out2 = self.relu(self.layer2(self.out1))
        self.out3 = self.relu(self.layer3(self.out2))
        self.out4 = self.layer4(self.out3)
        self.out = F.softmax(self.out4, dim=3)
        #
        # x = x.permute(0, 3, 1, 2)
        # b, c, h, w = x.size()
        #
        # x_add1 = self.conv11(x)
        #
        # x_add2 = self.conv12(x)
        #
        # x_add3 = self.conv13(x)
        #
        # dim = x_add3.shape[1]
        #
        # num1 = x_add1.shape[1] // dim
        # num2 = x_add2.shape[1] // dim
        # num3 = x_add3.shape[1] // dim
        # x_out = torch.empty(x.shape[0], dim, h, w).to(device=x.device)
        # x_out = torch.empty(x.shape[0], dim, h, w)
        # for i in range(dim):
        #     x1_tmp = x_add1[:, i * num1:(i + 1) * num1, :, :]
        #     x2_tmp = x_add2[:, i * num2:(i + 1) * num2, :, :]
        #     x3_tmp = x_add3[:, i * num3:(i + 1) * num3, :, :]
        #
        #     x_tmp = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
        #     addout = x1_tmp + x2_tmp + x3_tmp
        #     avgout = torch.mean(x_tmp, dim=1, keepdim=True)
        #     maxout, _ = torch.max(x_tmp, dim=1, keepdim=True)
        #     x_tmp = torch.cat([addout, avgout, maxout], dim=1)
        #     x_tmp = self.conv1(x_tmp)
        #     x_out[:, i:i + 1, :, :] = x_tmp
        # self.out = self.bn(self.conv2(x_out))
        # self.out = self.out.permute(0, 2, 3, 1)
        # self.out = F.softmax(self.out, dim=3)
        return self.out


class decoder_RGB(nn.Module):
    def __init__(self, endmember, band_RGB):
        super(decoder_RGB, self).__init__()
        self.endmember = endmember
        self.band = band_RGB
        self.layer = nn.Linear(self.endmember, self.band, bias=False)

    def forward(self, x):
        self.out = self.layer(x)
        return self.out


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class random_masking2(nn.Module):

    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, input1):
        b, c, h, w = input1.shape
        x = input1.reshape(b, c, h * w)
        N, L, D = x.shape

        x_out = self.pic_channel(x, L, self.mask_ratio)
        x_masked = x_out.reshape(b, c, h, w)
        return x_masked

    def pic_channel(self, x, channel, mask_ratio):
        num = int(channel * mask_ratio)

        mask_band = []
        for j in range(num):
            mask_band.append(random.randint(0, 50))

        for i in range(channel):
            if i in mask_band:
                x[:, i, :] += torch.abs(torch.randn(x.shape[0], x.shape[2])).to(device=x.device)
                # x[:, i, :] += torch.abs(torch.randn(x.shape[0], x.shape[2]))

        return x


class decoder_adaption(nn.Module):
    def __init__(self, band_hsi):
        super(decoder_adaption, self).__init__()
        self.band_hsi = band_hsi
        self.layer1 = nn.Conv2d(self.band_hsi, int(self.band_hsi/2), 3,1,1)
        self.layer2 = nn.Conv2d(int(self.band_hsi/2), int(self.band_hsi/2), 3,1,1)
        self.layer3 = nn.Conv2d(int(self.band_hsi/2), int(self.band_hsi/2), 3,1,1)
        self.layer4 = nn.Conv2d(int(self.band_hsi/2), self.band_hsi, 3,1,1)
        self.relu = nn.ReLU()

        # self.reshape0 = nn.Sequential(
        #     Conv3x3(band_hsi, 31, 3, 1),
        #     nn.PReLU(),
        #     Conv3x3(31, self.band_hsi, 3, 1)
        # )
        # self.mask = nn.Sequential(
        #     random_masking2(0.3),
        #     Conv3x3(band_hsi, 31, 3, 1),
        #     nn.PReLU(),
        #     Conv3x3(31, self.band_hsi, 3, 1)
        # )
    #
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        self.out = self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))
        # x = x.permute(0, 3, 1, 2)
        # out = self.reshape0(x)
        # before = out.clone()
        # out = self.mask(out)
        # out = before + 0.1 * out
        # out = out.permute(0, 2, 3, 1)
        self.out = self.out.permute(0,2,3,1)
        # x = x.permute(0,2,3,1)
        return self.out
        # return x
        # return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)


class reshape1(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.size(0), -1, 1, 1)


class denselayer(nn.Module):
    def __init__(self, cin, cout=31, RELU=True, BN=True, kernel_size=3, stride=1, act=True, dropout=False):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1, cin=cin, cout=cout, RELU=RELU, BN=BN, spatial_norm=True, stride=1)
        self.act = act
        self.actlayer = BCR(kernel=kernel_size, cin=cout, cout=cout, group=cout, RELU=RELU,
                            padding=(kernel_size - 1) // 2, BN=BN, spatial_norm=True, stride=stride)
        if dropout == True:
            self.dropout = nn.Dropout2d(0.1)
        self.drop = dropout

    def forward(self, x):
        if self.drop:
            [B, C, H, W] = x.shape
            x = x.permute([0, 2, 3, 1]).reshape([B * H * W, C, 1, 1])
            x = self.dropout(x)
            x = x.reshape([B, H, W, C]).permute([0, 3, 1, 2])
        output = self.compressLayer(x)
        if self.act == True:
            output = self.actlayer(output)

        return output


class BCR(nn.Module):
    def __init__(self, kernel, cin, cout, group=1, stride=1, RELU=True, padding=0, BN=False, spatial_norm=False):
        super(BCR, self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group, stride=stride,
                                  padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=kernel, groups=group,
                                           stride=int(abs(stride)), padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()

        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.Swish
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Dis_stage(nn.Module):
    def __init__(self, cin=31, cout=64, down=True):
        super(Dis_stage, self).__init__()
        self.down = down
        if down:
            self.downsample = nn.Sequential(
                denselayer(cin=cin, cout=cout, RELU=True, kernel_size=3, stride=2, BN=True),
                denselayer(cin=cout, cout=cout, RELU=True, kernel_size=3, stride=2, BN=True),
            )
        else:
            self.downsample = nn.Sequential(
                denselayer(cin=cin, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True))
        self.denseconv = nn.Sequential(
            denselayer(cin=cout, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 2, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 3, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 4, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 5, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 6, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
            denselayer(cin=cout * 7, cout=cout, RELU=True, kernel_size=3, stride=1, BN=True),
        )

    def forward(self, MSI):
        if self.down:
            dfeature = self.downsample(MSI)
        else:
            dfeature = self.downsample(MSI)

        feature = [dfeature]

        for conv in self.denseconv:
            feature.append(conv(torch.cat(feature, dim=1)))

        return feature[-1] + dfeature


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Discriminator_HSI(nn.Module):
    def __init__(self, band_hsi):
        super(Discriminator_HSI, self).__init__()
        self.band_hsi = band_hsi
        self.stage1 = Dis_stage(cin=self.band_hsi, cout=64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            reshape1(),
            denselayer(cin=256, cout=1, RELU=False, BN=False, kernel_size=1, stride=1),
            Flatten(),
            nn.Sigmoid())

    def forward(self, HSI):
        feature = self.stage1(HSI)
        prod = self.classifier(feature)

        return prod


def kl_divergence(p, q):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))

    return s1 + s2


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


class Loss_before_after(nn.Module):
    def __init__(self):
        super(Loss_before_after, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label)
        rrmse = torch.mean(error.view(-1))
        return rrmse


def conv(in_channels, out_channels, kernel_size, bias=True, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class Loss_ssim_hyper(nn.Module):
    def __init__(self):
        super(Loss_ssim_hyper, self).__init__()

    def forward(self, hyper):
        ssim_list = []
        for i in range(30):
            ssim_1 = ssim(hyper[:, i:i + 1, :, :], hyper[:, i + 1:i + 2, :, :])
            ssim_list.append(ssim_1)
        ssim_tensor = torch.Tensor(ssim_list)
        ssim_all = torch.mean(ssim_tensor)
        loss_ssim_hyper = 1 - ssim_all
        return loss_ssim_hyper


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=50):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class LossTrainCSS2(nn.Module):
    def __init__(self):
        super(LossTrainCSS2, self).__init__()

    # TODO 可以在这里加入ssim
    def forward(self, outputs, rgb_label):

        error = torch.abs(outputs - rgb_label)
        mrae = torch.mean(error)
        return mrae
