import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# import tensorflow as tf
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.summary.summary_iterator import summary_iterator

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hcipy import FFMpegWriter

def read_tensorboard_logs(log_dir):
    # 初始化 EventAccumulator，指定加载标量数据
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # 获取所有记录的标量标签
    tags = event_acc.Tags().get('scalars', [])

    data = []
    for tag in tags:
        # 逐步读取指定标签的数据
        #Training_Reward/Average Reward of last 10 episodes
        #"Evaluation_Strehl_SE/RL_SR_SE":
        if tag == "Evaluation_Strehl_SE/RL_SR_SE":

            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                data.append(
                    event.value
                )



    return data


def plot_train():
    # 使用示例
    # log_dir = 'path/to/your/log/dir'

    # logdir = 'outputgain_0.4_noice3_layer3_GM4_para0.16_train0.16_no_auencoder_worker4_hidden32_criticpolicy_kan/runs/performance/performance_training'
   #outputgain_0.4_noice3_layer1_GM4_para0.1_train0.1_no_auencoder_worker4_hidden32_criticpolicy_kan
    #outputgain_0.4_noice3_layer3_GM4_para0.16_train0.16_no_auencoder_worker4_hidden32_criticpolicy_kan_test
    logdir = 'outputgain_0.4_noice3_layer1_GM4_para0.1_train0.1_no_auencoder_worker4_hidden32_criticpolicy_kan/runs/performance/performance_training'
    # logdir = 'outputgain_0.4_noice3_layer3_GM4_para0.16_train0.16_no_auencoder_worker4_hidden32_criticpolicy_kan_test/runs/performance/performance_training'
    # logdir = 'outputgain_0.4_noice3_layer1_GM4_para0.05_train0.05_no_auencoder_worker4_hidden32_criticpolicy_kan_1/runs/performance/performance_training'

    list_encoder1 = read_tensorboard_logs(logdir)
    # list_encoder2 = read_tensorboard_logs(logdir2)
    # list_encoder3 = read_tensorboard_logs(logdir3)
    # list_encoder4 = read_tensorboard_logs(logdir4)


    plt.title(f"r0=0.1 train sr")
    # plt.plot(delay0[20:120], label="integrator")
    # plt.plot(delay_clamp1[1:], label="10*10_delay2_r0=0.1_clamp1")
    # plt.plot(delay_clamp5[1:], label="10*10_delay2_r0=0.1_clamp5")
    plt.plot(list_encoder1[:],  label="free model RL")
    # plt.plot(list_encoder2[:],  label="init_xavier")
    # plt.plot(list_encoder3[:],  label="critic_kan_hidden64")
    # plt.plot(list_encoder4[:],  label="linear_hindden64")

    plt.legend(loc='lower right', fontsize=10)
    # plt.legend()
    plt.show()




def plot_mp4():
    actor1_sr = np.load("test_result/compare_sr/se_list_loss_actor1.npy")
    actor2_sr = np.load("test_result/compare_sr/se_list_loss_actor2.npy")
    actor3_sr = np.load("test_result/compare_sr/se_list_loss_actor3.npy")
    actor4_sr = np.load("test_result/compare_sr/se_list_loss_actor4.npy")
    actor_full_sr = np.load("test_result/compare_sr/se_list_full.npy")
    dm_phase_loss_actor1 = np.load("test_result/compare_sr/dm_phase_loss_actor1.npy")
    dm_phase_loss_actor2 = np.load("test_result/compare_sr/dm_phase_loss_actor2.npy")
    dm_phase_loss_actor3 = np.load("test_result/compare_sr/dm_phase_loss_actor3.npy")
    dm_phase_loss_actor4 = np.load("test_result/compare_sr/dm_phase_loss_actor4.npy")
    dm_phase_full = np.load("test_result/compare_sr/dm_phase_full.npy")

    plt.figure()
    anim = FFMpegWriter(os.path.join("test_result/compare_sr/", 'dm_image_loss_actor_or_full.mp4'), framerate=10)

    max_val= max(dm_phase_loss_actor1.max(),dm_phase_loss_actor2.max(),dm_phase_loss_actor3.max(),dm_phase_loss_actor4.max(),dm_phase_full.max())
    min_val= min(dm_phase_loss_actor1.min(),dm_phase_loss_actor2.min(),dm_phase_loss_actor3.min(),dm_phase_loss_actor4.min(),dm_phase_full.min())
    norm_range = matplotlib.colors.Normalize(vmin=min_val,vmax=max_val)
    for i in range(len(actor1_sr)):

        plt.clf()
        plt.figure(figsize=(10, 15))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.subplot(3,2,1)
        plt.imshow(dm_phase_full[i], cmap='hot',norm=norm_range)
        plt.title(f"actor_full sr{actor_full_sr[i]:.2}")
        plt.colorbar(shrink=0.5)

        plt.subplot(3,2,2)
        plt.imshow(dm_phase_loss_actor1[i],cmap='hot',norm=norm_range)
        plt.title(f"loss actor1 sr{actor1_sr[i]:.2}")
        plt.colorbar(shrink=0.5)

        plt.subplot(3, 2, 3)
        plt.imshow(dm_phase_loss_actor2[i], cmap='hot',norm=norm_range)
        plt.title(f"loss actor2 sr{actor2_sr[i]:.2}")
        plt.colorbar(shrink=0.5)

        plt.subplot(3, 2, 4)
        plt.imshow(dm_phase_loss_actor3[i],cmap='hot',norm=norm_range)
        plt.title(f"loss actor3 sr{actor3_sr[i]:.2}")
        plt.colorbar(shrink=0.5)

        plt.subplot(3, 2, 5)
        plt.imshow(dm_phase_loss_actor4[i],cmap='hot',norm=norm_range)
        plt.title(f"loss actor4 sr{actor4_sr[i]:.2}")
        plt.colorbar(shrink=0.5)

        plt.subplot(3, 2, 6)
        plt.title(f"loss actor and full  compare sr")
        plt.plot(actor_full_sr,label="actor full")
        plt.plot(actor1_sr,label="loss actor1")
        plt.plot(actor2_sr,label="loss actor2")
        plt.plot(actor3_sr,label="loss actor3")
        plt.plot(actor4_sr,label="loss actor4")
        plt.legend()


        anim.add_frame()
    plt.close()
    anim.close()


if __name__ == "__main__":
    plot_train()



#
# tag_name = 'Evaluation_Strehl_LE/RL_SR_LE'  # 替换为你的图像数据的标签名
#
# # 获取图像数据的事件
# items = ea.scalars.Items()
# img_arrays = [event.image for event in img_events]
#
# #
# for i, img_array in enumerate(img_arrays):
#
#     print("dushuju")