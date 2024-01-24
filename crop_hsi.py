import numpy as np
import scipy.io as sio
import h5py

# 加载高光谱数据
# data = sio.loadmat('guidance_data/ICVL/Salinas_corrected_31.mat')['guidance_image']  # 根据实际数据文件路径加载高光谱数据
# data = sio.loadmat('data/DFC2018 Houston/HoustonU.mat')['guidance_image']  # 根据实际数据文件路径加载高光谱数据
data = h5py.File('data/DFC2018 Houston/HoustonU.mat')['houstonU'][:]  # 根据实际数据文件路径加载高光谱数据
# data = np.array(data).transpose(1, 2, 0)

# 定义裁剪区域的大小和步长
patch_size = 128  # 裁剪区域的大小
stride = 64  # 步长

# 获取高光谱数据的形状
data_shape = data.shape


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


# creat patches
patches_hyper = Im2Patch(data, patch_size, stride)
patch_num = 1
# add data ：重组patches
for j in range(patches_hyper.shape[3]):
    print("generate sample #%d" % patch_num)
    cropped_data = patches_hyper[:, :, :, j]
    # 输出裁剪结果的形状
    print("第" + str(patch_num) + "块裁剪区域")
    print("Cropped data shape:", cropped_data.shape)

    if patch_num < 200:
        sio.savemat('data/DFC2018 Houston/traindata/HoustonU_' + str(patch_num) + '.mat', {'rad': cropped_data})
    else:
        sio.savemat('data/DFC2018 Houston/testdata/HoustonU_' + str(patch_num) + '.mat', {'rad': cropped_data})
    patch_num += 1

# # 按步长滑动裁剪
# ii = 0
# for i in range(0, data_shape[0] - crop_size[0] + 1, step_size[0]):
#     for j in range(0, data_shape[1] - crop_size[1] + 1, step_size[1]):
#         # 获取当前裁剪区域的起始和结束索引
#         start_row = i
#         end_row = i + crop_size[0]
#         start_col = j
#         end_col = j + crop_size[1]
#
#         # 裁剪高光谱数据，并添加到裁剪结果列表
#         cropped_data = data[start_row:end_row, start_col:end_col, :]
#         # 将裁剪结果转换为NumPy数组
#         cropped_data = np.array(cropped_data)
#         # 输出裁剪结果的形状
#         print("Cropped data shape:", cropped_data.shape)
#         # cropped_data.append(data[start_row:end_row, start_col:end_col, :])
#         ii = ii + 1
#         print("第" + str(ii) + "块裁剪区域")
#         # if ii < 200:
#         #     sio.savemat('data/DFC2018 Houston/traindata/HoustonU_' + str(ii) + '.mat', {'rad': cropped_data})
#         # else:
#         #     sio.savemat('data/DFC2018 Houston/testdata/HoustonU_' + str(ii) + '.mat', {'rad': cropped_data})
#         sio.savemat('guidance_data/ICVL/Crop_HSI/Salinas_corrected_31_' + str(ii) + '.mat', {'guidance_image': cropped_data})
