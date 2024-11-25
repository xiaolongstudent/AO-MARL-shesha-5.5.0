import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

delay_clamp0 = np.load("test_result/compare_sr/kan_se_list0.npy",allow_pickle=True)
delay_clamp1 = np.load("test_result/compare_sr/kan_se_list1.npy",allow_pickle=True)
delay_clamp2 = np.load("test_result/compare_sr/kan_se_list2.npy",allow_pickle=True)
delay_clamp3 = np.load("test_result/compare_sr/kan_se_list3.npy",allow_pickle=True)
delay_clamp4 = np.load("test_result/compare_sr/kan_se_list4.npy",allow_pickle=True)

delay_1_clamp1 = np.load("test_result/gain0.4_worker4/se_list0.npy",allow_pickle=True)
delay_1_clamp2 = np.load("test_result/gain0.4_worker4/se_list1.npy",allow_pickle=True)
delay_1_clamp3 = np.load("test_result/gain0.4_worker4/se_list3.npy",allow_pickle=True)
delay_1_clamp4 = np.load("test_result/gain0.4_worker4/se_list3.npy",allow_pickle=True)
delay_1_clamp5 = np.load("test_result/gain0.4_worker4/se_list4.npy",allow_pickle=True)

delay_inte_clamp1 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
delay_inte_clamp2 = np.load("test_result/train_gain0.4_Integrator/se_list1.npy",allow_pickle=True)
delay_inte_clamp3 = np.load("test_result/train_gain0.4_Integrator/se_list3.npy",allow_pickle=True)
delay_inte_clamp4 = np.load("test_result/train_gain0.4_Integrator/se_list3.npy",allow_pickle=True)
delay_inte_clamp5= np.load("test_result/train_gain0.4_Integrator/se_list4.npy",allow_pickle=True)

# delay_inte_clamp1 = np.load("test_result/train_gain0.4_Integrator_r0_0.1/se_list0.npy",allow_pickle=True)
# delay_inte_clamp2 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# delay_inte_clamp3 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# delay_inte_clamp4 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# delay_inte_clamp5 = np.load("test_result/train_gain0.4_Integrator/se_list0.npy",allow_pickle=True)
# # delay_ip = np.load("10*10_test/delay2_po4ao_test_r0_0.05_pi_control.npy",allow_pickle=True)

# delay_ip = np.load("10*10_test/delay2_po4ao_test_r0_0.16_clamp30_inte.npy",allow_pickle=True)

data_array = np.vstack((delay_clamp0,delay_clamp1,delay_clamp2,delay_clamp3,delay_clamp4))
data_array_clamp = np.vstack((delay_1_clamp1,delay_1_clamp2,delay_1_clamp3,delay_1_clamp4,delay_1_clamp5))
# data_array_clamp = np.vstack((delay_2_clamp1,delay_2_clamp2,delay_2_clamp3,delay_2_clamp4,delay_2_clamp5))

data_inte_array = np.vstack((delay_inte_clamp1 ,delay_inte_clamp2,delay_inte_clamp3 ,delay_inte_clamp4 ,delay_inte_clamp5  ))

data_array_mean = np.mean(data_array,axis=0)
data_array_std = np.std(data_array,axis =0)

data_array_mean_ = np.mean(data_array_clamp,axis=0)
data_array_std_ = np.std(data_array_clamp,axis =0)

data_inte_mean = np.mean(data_inte_array,axis=0)
data_inte_std = np.std(data_inte_array,axis =0)


x = range(len(data_array_mean[:500]))

plt.title(f"10*10 AO_MARL test KAN and linear net compare sr ")
# plt.plot(delay0[20:120], label="integrator")
# plt.plot(delay_clamp1[1:], label="10*10_delay2_r0=0.1_clamp1")
# plt.plot(delay_clamp5[1:], label="10*10_delay2_r0=0.1_clamp5")
plt.plot(data_array_mean[:500], color = 'orange', label="kan net")
plt.plot(data_array_mean_[:500], color = 'green', label="linear net ")
plt.plot(data_inte_mean[:500], color = 'red', label="Integrator")

plt.fill_between(x,data_array_mean[:500]+ data_array_std[:500], data_array_mean[:500]-data_array_std[:500], color = 'orange', alpha=0.5)
plt.fill_between(x,data_array_mean_[:500]+data_array_std_[:500], data_array_mean_[:500]-data_array_std_[:500], color = 'green', alpha=0.5)
plt.fill_between(x,data_inte_mean[:500]+data_inte_std[:500], data_inte_mean[:500]-data_inte_std[:500], color = 'red', alpha=0.5)

# plt.plot(delay_ip[:100],  color = 'red', label="PI control r0=0.05 sr")
# # plt.plot(delay_clamp30[1:], label="10*10_delay2_r0=0.16_clamp30")
# plt.plot(delay_inte[1:], color = 'red', label="40*40_delay2_r0=0.16_inte")
# plt.plot(po4ao_2[20:], label="step_1000")

# plt.plot(po4ao_3[20:], label="state_3")
# plt.plot(po4ao_mean[20:], label="state_mean")
# plt.plot(po4ao_400 [20:], label="step_400")
plt.legend()
plt.show()

