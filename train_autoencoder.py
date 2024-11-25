import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch import  optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from src.autoencoder.autoencoder_models import DenoisingAutoencoderCNN2DSingleSubapeture
from  data_loader import EncoderDataset
import tqdm
from hcipy import FFMpegWriter
import  os
from skimage import io

from PIL import Image
from skimage.metrics import structural_similarity as ssim

batchsize = 256
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cup")
lr = 1e-2
epoch_len = 100
class Train_class():
    def __init__(self):

      self.model = DenoisingAutoencoderCNN2DSingleSubapeture().to(device)
      self.file_dict = "output/autoencoder/train_gs4_3layer/"
      if not os.path.exists(self.file_dict):
          os.makedirs(self.file_dict)

      self.criteon = nn.MSELoss(reduction="mean").to(device)

    def plot_img(self,tain_loss_list,val_loss_list):
        plt.figure()
        plt.title("loss")
        plt.plot(tain_loss_list,label='train loss')
        plt.plot(val_loss_list,label = 'val loss')

        plt.show()
        plt.savefig(self.file_dict +"loss_img.png",)

    # def plt_mp4(self,oriImg,transImg,labImg):
    #     plt.figure()
    #     anim = FFMpegWriter(os.path.join("output/autoencoder/", 'animation.mp4'), framerate=10)
    #     images = np.asarray(images)
    #     images = images.transpose(2, 0, 1)
    #     for i in range(len(images)):
    #         plt.clf()
    #         plt.imshow(oriImg)
    #         plt.title('bincube image')
    #         plt.legend()
    #
    #         anim.add_frame()
    #     plt.close()
    #     anim.close()




    def predict(self):
        # dict = torch.load(self.file_dict+ "save_model.pth")
        dict = torch.load("output/autoencoder/save_model1.pth")

        self.model.load_state_dict(dict)
        self.model.eval()
        test_db = EncoderDataset("test")
        test_loader = DataLoader(test_db, batch_size=1,shuffle = True)
        losses = []
        # anim = FFMpegWriter(self.file_dict+'testanimation_GS9.mp4', framerate=10)
        anim = FFMpegWriter('output/autoencoder/testanimation_GS4.mp4', framerate=10)

        plt.figure()


        #t图片xiangsixing
        sim_ori_modelresult = []
        sim_modelresult_lab = []


        with torch.no_grad():
            for rep in tqdm.tqdm(range(100)):
                dummy_input, label = next(iter(test_loader))
                lab = label.to(device)
                dummy_input = dummy_input.to(device)
                dummy_input = dummy_input.unsqueeze(0)
                lab = lab.unsqueeze(0)
                refer_labs = self.model(dummy_input)

                loss = self.criteon(lab, refer_labs)
                losses.append(loss.item())
                dummy_input = dummy_input.squeeze().cpu().numpy()
                refer_labs = refer_labs.squeeze().cpu().numpy()
                lab = lab.squeeze().cpu().numpy()
                # for i in range(len(rtuefer_labs)):

                original_diff_value, diff = ssim(dummy_input, lab, full=True,data_range = dummy_input.max()-dummy_input.min())
                sim_ori_modelresult.append(original_diff_value)

                lab_diff_value, diff = ssim(lab, refer_labs, full=True,data_range = lab.max()-lab.min())
                sim_modelresult_lab.append(lab_diff_value)


                plt.clf()
                plt.figure(figsize=(15, 5))
                plt.subplots_adjust(hspace=0.5)
                plt.subplot(1, 3, 1)
                plt.title(f"orginal image(original and label\n {original_diff_value:.3f})")
                plt.imshow(dummy_input)


                plt.subplot(1, 3, 2)
                plt.title(f"predict\n(predict and label{lab_diff_value:.3f})")
                plt.imshow(refer_labs)

                plt.subplot(1, 3, 3)
                plt.title("lab")
                plt.imshow(lab)

                anim.add_frame()

        plt.close()
        anim.close()


        # fig, (ax1, ax2,ax3) = plt.subplots(3)
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(1,3,1)
        plt.title("test loss ")
        plt.plot(losses,label = "test loss")

        plt.subplot(1,3,2)
        plt.title("original compare lab ssim")
        plt.plot(sim_ori_modelresult,label = "ssim")

        plt.subplot(1, 3, 3)
        plt.title("predict compare lab ssim")
        plt.plot(sim_modelresult_lab, label="ssim")
        plt.show()
        # plt.savefig(self.file_dict +"test_loss.png")
        plt.savefig("output/autoencoder/GS4_test_loss.png")



    def train(self):
        best_file_name = self.file_dict + "save_model.pth"

        best_train_loss=10000
        best_val_loss = 10000
        train_loss = []
        val_loss = []
        train_db = EncoderDataset("train")
        val_db = EncoderDataset("val")
        train_dataloader = DataLoader(train_db, batch_size=batchsize,drop_last=True,shuffle = True)
        val_dataloader = DataLoader(val_db, batch_size=batchsize,drop_last=True,shuffle = True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.5, last_epoch=-1)
        model = self.model.train()
        for epoch in range(epoch_len):
           for batchidx, (image,label) in enumerate(train_dataloader):

             image ,label = image.to(device),label.to(device)
             image = image.unsqueeze(2).view(-1,1,16,16)
             label = label.unsqueeze(2).view(-1,1,16,16)
             # image = image.unsqueeze(1)
             predict = self.model(image)
             loss = self.criteon(predict,label)
             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()
           if best_train_loss > loss:
               best_train_loss = loss
           self.scheduler.step()
           train_loss.append(loss.item())
           print("epoch---------", epoch, "btrain_loss---------------", loss.item())

           # 模型验证
           model.eval()
           with torch.no_grad():
               for batchhidx, (image, label) in enumerate(val_dataloader):
                   image, label = image.to(device), label.to(device)
                   image = image.unsqueeze(2).view(-1, 1, 16, 16)
                   label = label.unsqueeze(2).view(-1, 1, 16, 16)

                   # image = image.unsqueeze(1)
                   logits = model(image)
                   # label = label.to(torch.float32)
                   loss = self.criteon(logits, label)
               if best_val_loss > loss:
                   best_val_loss = loss
                   torch.save(model.state_dict(), best_file_name)

               val_loss.append(loss.item())

           print("val_loss---------------", loss.item())
        tain_path = self.file_dict + "train_loss.npy"
        val_path = self.file_dict +"val_loss.npy"
        np.save(tain_path, train_loss)
        np.save(val_path, val_loss)
        self.plot_img(train_loss,val_loss)



if __name__ == "__main__":
    # auto_class = Train_class()
    # auto_class.train()
    # auto_class.predict()
    train_loss = np.load("output/autoencoder/train_loss2.npy")
    val_loss = np.load("output/autoencoder/val_loss2.npy")
    min = train_loss.min()
    val_min = val_loss.min()
    print(f"train_min,{min} val_min{val_min}")

