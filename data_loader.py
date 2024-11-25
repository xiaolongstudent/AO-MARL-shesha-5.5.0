import numpy as np
import torch
import os
from torch.utils.data import Dataset
import  random
import matplotlib.pyplot  as plt
data_path_list = []
lab_path_list = []



class EncoderDataset(Dataset):
    def __init__(self,type):
        root_path = "output/autoencoder/output_dataset_autoencoder/"
        global data_path_list
        global lab_path_list

        self.data = []
        self.target = []

        if len(data_path_list)==0:
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    # file_path = os.path.join(root, file)
                    if "noise3" in file:
                        data_path_list.append(file)
                    else:
                        lab_path_list.append(file)

            random.shuffle(data_path_list)
        data_len = len(data_path_list)
        if type == "train":
            file_list = data_path_list[:int(data_len*0.8)]

        elif type == "val":
            file_list = data_path_list[int(data_len*0.8):int(data_len*0.9)]
        else:
            file_list = data_path_list[int(data_len*0.9):]
        
        name_len = len("noise3_image_production_sh_10x10_2m_small")

        for path in file_list[:int(len(file_list)*0.01)]:
            file_num = path[name_len:]
            lab_path = "noiseminus1_image_production_sh_10x10_2m_small"+file_num


            if lab_path in lab_path_list:
                lab_path = os.path.join(root_path, lab_path)
                file_path = os.path.join(root_path, path)

                load_data = np.load(file_path)
                load_data = np.swapaxes(load_data,1,3)
                load_data = np.swapaxes(load_data, 2, 3)
                load_data = load_data.reshape(-1,16,16)


                self.data.extend(load_data)
                load_target_data = np.load(lab_path)
                load_target_data = np.swapaxes(load_target_data, 1, 3)
                load_target_data = np.swapaxes(load_target_data, 2, 3)
                load_target_data = load_target_data.reshape(-1, 16, 16)
                self.target.extend(load_target_data)
        print("加载数据完成！")


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        origin_data = self.data[idx]
        lab_data = self.target[idx]
        # origin_data = origin_data.transpose(2,0,1)
        # lab_data = lab_data.transpose(2,0,1)


        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow(origin_data[0])
        #
        # plt.subplot(2,1,2)
        # plt.imshow(lab_data[0])
        # plt.show()




        return origin_data,lab_data


# if __name__ == "__main__":
#     loader = encoderDataset("train")








