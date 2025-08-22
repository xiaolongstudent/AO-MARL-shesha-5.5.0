import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator



rl_list = np.zeros((5, 500))
for i in range(3):
    if i == 0:
        filename = "test_model_sr/layer3_train_0.16_test0.16_se_list_kan"
    elif i == 1:
        filename = "test_model_sr/layer3_train_0.16_test0.1_se_list_kan"
    else:
        filename = "test_model_sr/layer3_train_0.16_test0.05_se_list_kan"

    for j in range(5):
        delay_sr = np.load(filename+f"{j}.npy")
        if i == 0:
            rl_list[j,:200]=delay_sr[:200]
            # rl_list[j,:] = delay_sr[:500]
        elif i ==1:
            rl_list[j,200:350]=delay_sr[200:350]
        else:
            rl_list[j,350:500] =delay_sr[350:500]
inte_list = np.zeros((5, 500))
for i in range(3):
    if i == 0:
        filename = "test_model_sr/layer3_train_0.16_test0.16_sl_list_inte"
    elif i == 1:
        filename = "test_model_sr/layer3_train_0.16_test0.1_sl_list_inte"
    else:
        filename = "test_model_sr/layer3_train_0.16_test0.05_sl_list_inte"

    for j in range(5):
        delay_sr = np.load(filename+f"{j}.npy")
        if i == 0:
            inte_list[j,:200] = delay_sr[:200]
            # inte_list[j,:] = delay_sr[:500]
        elif i ==1:
            inte_list[j,200:350] = delay_sr[200:350]
        else:
            inte_list[j,350:500] = delay_sr[350:500]

# delay_clamp0 = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan0.npy",allow_pickle=True)
# delay_clamp1 = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan1.npy",allow_pickle=True)
# delay_clamp2 = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan2.npy",allow_pickle=True)
# delay_clamp3 = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan3.npy",allow_pickle=True)
# delay_clamp4 = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan4.npy",allow_pickle=True)
#
# delay_1_clamp1 = np.load("test_model_sr/layer3_train_0.16_test0.1_se_list_kan0.npy",allow_pickle=True)
# delay_1_clamp2 = np.load("test_model_sr/layer3_train_0.16_test0.1_se_list_kan1.npy",allow_pickle=True)
# delay_1_clamp3 = np.load("test_model_sr/layer3_train_0.16_test0.1_se_list_kan2.npy",allow_pickle=True)
# delay_1_clamp4 = np.load("test_model_sr/layer3_train_0.16_test0.1_se_list_kan3.npy",allow_pickle=True)
# delay_1_clamp5 = np.load("test_model_sr/layer3_train_0.16_test0.1_se_list_kan4.npy",allow_pickle=True)
#
# delay_2_clamp1 = np.load("test_model_sr/layer3_train_0.16_test0.05_se_list_kan0.npy",allow_pickle=True)
# delay_2_clamp2 = np.load("test_model_sr/llayer3_train_0.16_test0.05_se_list_kan1.npy",allow_pickle=True)
# delay_2_clamp3 = np.load("test_model_sr/layer3_train_0.16_test0.05_se_list_kan2.npy",allow_pickle=True)
# delay_2_clamp4 = np.load("test_model_sr/layer3_train_0.16_test0.05_se_list_kan3.npy",allow_pickle=True)
# delay_2_clamp5 = np.load("test_model_sr/layer3_train_0.16_test0.05_se_list_kan4.npy",allow_pickle=True)
#
#
#
# delay_clamp0_inte = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan0.npy",allow_pickle=True)
# delay_clamp1_inte = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan1.npy",allow_pickle=True)
# delay_clamp2_inte = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan2.npy",allow_pickle=True)
# delay_clamp3_inte = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan3.npy",allow_pickle=True)
# delay_clamp4_inte = np.load("test_model_sr/layer3_train_0.16_test0.16_se_list_kan4.npy",allow_pickle=True)
#
#
# delay_1_clamp1_inte = np.load("test_model_sr/layer1_train_0.05_test0.05_se_list_inte0.npy",allow_pickle=True)
# delay_1_clamp2_inte = np.load("test_model_sr/layer1_train_0.05_test0.05_se_list_inte1.npy",allow_pickle=True)
# delay_1_clamp3_inte = np.load("test_model_sr/layer1_train_0.05_test0.05_se_list_inte2.npy",allow_pickle=True)
# delay_1_clamp4_inte = np.load("test_model_sr/layer1_train_0.05_test0.05_se_list_inte3.npy",allow_pickle=True)
# delay_1_clamp5_inte = np.load("test_model_sr/layer1_train_0.05_test0.05_se_list_inte4.npy",allow_pickle=True)
#
#
#
# delay_inte_clamp1 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# delay_inte_clamp2 = np.load("test_result/train_gain0.4_Integrator/se_list1.npy",allow_pickle=True)
# delay_inte_clamp3 = np.load("test_result/train_gain0.4_Integrator/se_list3.npy",allow_pickle=True)
# delay_inte_clamp4 = np.load("test_result/train_gain0.4_Integrator/se_list3.npy",allow_pickle=True)
# delay_inte_clamp5= np.load("test_result/train_gain0.4_Integrator/se_list4.npy",allow_pickle=True)
#
# # delay_inte_clamp1 = np.load("test_result/train_gain0.4_Integrator_r0_0.1/se_list0.npy",allow_pickle=True)
# # delay_inte_clamp2 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# # delay_inte_clamp3 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# # delay_inte_clamp4 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# # delay_inte_clamp5 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# # # delay_ip = np.load("10*10_test/delay2_po4ao_test_r0_0.05_pi_control.npy",allow_pickle=True)

# delay_ip = np.load("10*10_test/delay2_po4ao_test_r0_0.16_clamp30_inte.npy",allow_pickle=True)

# data_array = np.vstack((delay_clamp0,delay_clamp1,delay_clamp2,delay_clamp3,delay_clamp4))
# data_array_clamp = np.vstack((delay_1_clamp1,delay_1_clamp2,delay_1_clamp3,delay_1_clamp4,delay_1_clamp5))
# # data_array_clamp = np.vstack((delay_2_clamp1,delay_2_clamp2,delay_2_clamp3,delay_2_clamp4,delay_2_clamp5))
#
# data_inte_array = np.vstack((delay_inte_clamp1 ,delay_inte_clamp2,delay_inte_clamp3 ,delay_inte_clamp4 ,delay_inte_clamp5  ))


data_array_mean = np.mean(rl_list,axis=0)
data_array_std = np.std(rl_list,axis =0)

data_array_mean_ = np.mean(inte_list,axis=0)
data_array_std_ = np.std(inte_list,axis =0)

# data_inte_mean = np.mean(data_inte_array,axis=0)
# data_inte_std = np.std(data_inte_array,axis =0)


x = range(len(data_array_mean[:500]))

plt.title(f"train r0=0.16 test r0[0.16,0.1,0.05] RL control and PI control compare sr")
# plt.plot(delay0[20:120], label="integrator")
# plt.plot(delay_clamp1[1:], label="10*10_delay2_r0=0.1_clamp1")
# plt.plot(delay_clamp5[1:], label="10*10_delay2_r0=0.1_clamp5")
plt.plot(data_array_mean[:500], color = 'orange', label="free model RL control")
plt.plot(data_array_mean_[:500], color = 'green', label="PI control ")
# plt.plot(data_inte_mean[:500], color = 'red', label="Integrator")

plt.fill_between(x,data_array_mean[:500]+ data_array_std[:500], data_array_mean[:500]-data_array_std[:500], color = 'orange', alpha=0.5)
plt.fill_between(x,data_array_mean_[:500]+data_array_std_[:500], data_array_mean_[:500]-data_array_std_[:500], color = 'green', alpha=0.5)
# plt.fill_between(x,data_inte_mean[:500]+data_inte_std[:500], data_inte_mean[:500]-data_inte_std[:500], color = 'red', alpha=0.5)

# plt.plot(delay_ip[:100],  color = 'red', label="PI control r0=0.05 sr")
# # plt.plot(delay_clamp30[1:], label="10*10_delay2_r0=0.16_clamp30")
# plt.plot(delay_inte[1:], color = 'red', label="40*40_delay2_r0=0.16_inte")
# plt.plot(po4ao_2[20:], label="step_1000")

# plt.plot(po4ao_3[20:], label="state_3")
# plt.plot(po4ao_mean[20:], label="state_mean")
# plt.plot(po4ao_400 [20:], label="step_400")
plt.legend()
plt.show()

