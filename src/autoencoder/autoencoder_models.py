import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import pywt
import pywt.data
# from ksvd import KSVD

from skimage.metrics import structural_similarity as ssim

class DenoisingAutoencoderLinear(nn.Module):
    def __init__(self,
                 num_inputs=28 * 28,
                 hidden_dim1=500,
                 hidden_dim2=120,
                 hidden_dim3=40):
        super(DenoisingAutoencoderLinear, self).__init__()

        # Enconder
        self.linear_encoder1 = nn.Linear(num_inputs, hidden_dim1)
        self.linear_encoder2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear_encoder3 = nn.Linear(hidden_dim2, hidden_dim3)

        # Decoder
        self.linear_decoder1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_decoder2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_decoder3 = nn.Linear(hidden_dim1, num_inputs)

        self.ReLU = nn.ReLU()

    def encoder(self, x):
        x = self.ReLU(self.linear_encoder1(x))
        x = self.ReLU(self.linear_encoder2(x))
        x = self.ReLU(self.linear_encoder3(x))
        return x

    def decoder(self, x):
        x = self.ReLU(self.linear_decoder1(x))
        x = self.ReLU(self.linear_decoder2(x))
        x = self.ReLU(self.linear_decoder3(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAutoencoderCnnCentroids(nn.Module):
    def __init__(self,
                 num_inputs=(28, 28),
                 hidden_dim1=500,
                 hidden_dim2=120,
                 hidden_dim3=40):
        # conv3d
        # (N, CIN, DIN, HIN, WIN)
        # (N, COUT, DOUT, HOUT, WOUT)
        # DOUT=[DIN+2PADDING-DILATION*(KERNEL_SIZE-1)-1]/STRIDE + 1

        super(DenoisingAutoencoderCnnCentroids, self).__init__()
        # input (128, 1, 16, 16, 64)
        self.encoder1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)  # (128, 16, 16, 16, 64)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)  # (128, 16, 8, 8, 32)
        self.encoder2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)  # (128, 32, 8, 8, 32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)  # (128, 32, 4, 4, 16)
        self.encoder3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # (128, 64, 4, 4, 16)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)  # (128, 32, 2, 2, 8) 64*2*2*8

        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 128)

    def cnn(self, x):
        x = F.relu(self.encoder1(x))
        x = self.maxpool1(x)
        x = F.relu(self.encoder2(x))
        x = self.maxpool2(x)
        x = F.relu(self.encoder3(x))
        x = self.maxpool3(x)
        return x

    def feedforward(self, x):
        x = F.relu(self.linear1(x.view(-1, 2048)))
        x = self.linear2(x)
        return x

    def forward(self, x):
        x = self.cnn(x)
        x = self.feedforward(x)
        return x

class DenoisingAutoencoderCNN(nn.Module):
    def __init__(self, num_inputs=(28, 28), hidden_dim1=500, hidden_dim2=120, hidden_dim3=40):
        # conv3d
        # (N, CIN, DIN, HIN, WIN)
        # (N, COUT, DOUT, HOUT, WOUT)
        # DOUT=[DIN+2PADDING-DILATION*(KERNEL_SIZE-1)-1]/STRIDE + 1

        super(DenoisingAutoencoderCNN, self).__init__()

        self.encoder1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.encoder2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.encoder3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.decoder1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)
        #print("mmm")

    def encoder(self, x):
        x = F.relu(self.encoder1(x))
        x = self.maxpool1(x)
        x = F.relu(self.encoder2(x))
        x = self.maxpool2(x)
        x = self.encoder3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = self.decoder3(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAutoencoderCNN2DSingleSubapeture(nn.Module):
    def __init__(self,
                 criterion='MSE',
                 batch_norm=False):

        super(DenoisingAutoencoderCNN2DSingleSubapeture, self).__init__()
        # input (128, 1, 16, 16)
        self.encoder1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # (128, 128, 16, 16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (128, 16, 8, 8, 32)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # (128, 256, 8, 8, 32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (128, 32, 4, 4, 16)
        self.encoder3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (128, 512, 4, 4, 16)
        self.decoder1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.criterion = criterion
        self.batch_norm = batch_norm

        if batch_norm:
            self.encoder_bn1 = nn.BatchNorm2d(128)
            self.encoder_bn2 = nn.BatchNorm2d(256)
            self.encoder_bn3 = nn.BatchNorm2d(512)
            self.decoder_bn1 = nn.BatchNorm2d(256)
            self.decoder_bn2 = nn.BatchNorm2d(128)
        else:
            self.encoder_bn1 = None
            self.encoder_bn2 = None
            self.encoder_bn3 = None
            self.decoder_bn1 = None
            self.decoder_bn2 = None

    def encoder(self, x):

        if self.batch_norm:
            x = F.relu(self.encoder1(x))
            x = self.encoder_bn1(self.maxpool1(x))
            x = F.relu(self.encoder2(x))
            x = self.encoder_bn2(self.maxpool2(x))
            x = self.encoder_bn3(F.relu(self.encoder3(x)))
        else:
            x = F.relu(self.encoder1(x))
            x = self.maxpool1(x)
            x = F.relu(self.encoder2(x))
            x = self.maxpool2(x)
            x = F.relu(self.encoder3(x))

        return x

    def decoder(self, x):
        # print(x.shape)
        if self.batch_norm:
            x = self.decoder_bn1(F.relu(self.decoder1(x)))
            x = self.decoder_bn2(F.relu(self.decoder2(x)))
        else:
            # print(x.shape)
            x = F.relu(self.decoder1(x))
            # print(x.shape)
            x = F.relu(self.decoder2(x))
        x = self.decoder3(x)
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.criterion == "BCE":
            x = torch.sigmoid(x)
        return x

class Autoencoder:
    def __init__(self, config):
        self.type = config.autoencoder['type'].lower()

        if self.type == "cnn":
            self.model = DenoisingAutoencoderCNN(num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40) # num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)
        elif self.type == "cnn_single_subaperture":
            self.model = DenoisingAutoencoderCNN2DSingleSubapeture()  # num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)
        elif self.type == "cnn_centroids":
            self.model = DenoisingAutoencoderCnnCentroids()
        else:
            self.model = DenoisingAutoencoderLinear(num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.autoencoder['path'] is not None:
            autoencoder_path = config.autoencoder['path']
            if torch.cuda.is_available():
                 self.model.load_state_dict(torch.load(autoencoder_path))
            else:
                self.model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))
            self.model.to(device=self.device)
            self.model.eval()

    def predict(self, noisy_tensor, only_inference_time=False):  # noise_image
        # noisy_tensor = torch.tensor(noisy_image).to(device=self.device)
        if self.type == "cnn":
            with torch.no_grad():
                predicted = self.model(noisy_tensor.view(1,1,16,16,-1)).cpu().numpy().reshape(16,16,-1)
        elif self.type == "cnn_single_subaperture":
            predicted = self.traditional_denoise(noisy_tensor)
            # with torch.no_grad():
            #     if only_inference_time:
            #         predicted = self.model(noisy_tensor)
            #     else:
            #         predicted = self.model(noisy_tensor.view(-1,1,16,16)).cpu().numpy().reshape(-1,16,16)
        elif self.type == "cnn_centroids":
            with torch.no_grad():
                predicted = self.model(noisy_tensor.view(1,1,16,16,-1)).cpu().numpy()
        else:
            with torch.no_grad():
                # print(noisy_tensor.shape)
                predicted = self.model(noisy_tensor.reshape(-1,16*16*64)).cpu().numpy().reshape(16,16,-1)
        return predicted

    def traditional_denoise(self,noisy_tesor):
        shap = noisy_tesor.shape
        denoisy_image = []

        for i in range(shap[0]):
            image = noisy_tesor[i]
            # clear_img = predicted[i].cpu().detach().numpy()
            image = image.cpu().detach().numpy()
            de_image = self.gaussionBlur(image)
            denoisy_image.append(de_image)
            # unnoise_filter_diff, diff = ssim(de_image, clear_img, full=True,
            #                                  data_range=clear_img - clear_img)
            #
            # noise_filter_diff, diff = ssim(image, clear_img, full=True,
            #                                  data_range=clear_img - clear_img)
            #
            # plt.figure()
            # plt.suptitle("ksvd filter denoising")
            # plt.subplot(1, 3, 1)
            # plt.title(f"noisy_image ")
            # plt.imshow(image)
            #
            # plt.subplot(1, 3, 2)
            # plt.title(f"denoisy_image ssim{unnoise_filter_diff:.2f}")
            # plt.imshow(de_image)
            #
            # plt.subplot(1, 3, 3)
            # plt.title(f"model_image")
            # plt.imshow(clear_img)
            # plt.show()
        denoisy_image = np.array(denoisy_image)
        return  denoisy_image


    #高斯去噪
    def gaussionBlur(self, noisy_image):
        denoised_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
        return denoised_image

    #傅里叶变换
    def fft_filter(self,noisy_image):
        # 进行傅里叶变换，转换到频域
        f = np.fft.fft2(noisy_image)
        fshift = np.fft.fftshift(f)  # 将频谱的低频移到中心

        # 生成低通滤波器（设置阈值，去除高频部分）
        rows, cols = noisy_image.shape
        crow, ccol = rows // 2, cols // 2  # 频谱中心
        radius = 30  # 滤波器的半径，控制去噪程度
        mask = np.zeros((rows, cols), np.uint8)
        center = [crow, ccol]
        x, y = np.fft.fftfreq(rows), np.fft.fftfreq(cols)
        dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        mask[dist < radius] = 1  # 保留低频部分

        # 应用低通滤波器
        fshift = fshift * mask

        # 进行逆傅里叶变换，回到空间域
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)  # 获取图像的绝对值作为最终结果
        return img_back
    #bm3d
    def bm3d_filter(self,noisy_image):
        # 使用BM3D去噪
        denoised_image = bm3d(noisy_image, sigma_psd=25 / 255.0)  # sigma_psd 是噪声的标准差（归一化）
        return denoised_image

    def svd_denoising(self, noisy_image, k=5):
        # SVD分解
        U, S, Vt = np.linalg.svd(noisy_image, full_matrices=False)

        # 仅保留前k个奇异值
        S[k:] = 0

        # 重建去噪矩阵
        denoised_matrix = np.dot(U, np.dot(np.diag(S), Vt))
        return denoised_matrix






# autoencoder = autoencoder({"type":"linear", "path":"autoencoder"})
# image2predict = np.load("noise3_image_scao_sh_10x10_16pix_2m_gs9_noise3_delay0.npy")
# real_image2predict = image2predict[0,:,:]
# tensor_image = torch.Tensor(real_image2predict)
# with torch.no_grad():
#    predicted = autoencoder.predict(tensor_image.view(-1,160*160)).cpu().numpy().reshape(160,160)
# plt.imshow(predicted)
# plt.show()