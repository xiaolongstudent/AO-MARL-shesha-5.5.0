import torch.distributed.rpc as rpc
import os
from src.reinforcement_learning.helper_functions.utils.help_initialization import obtain_args, print_and_assertions
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from hcipy import FFMpegWriter
from torch.utils.tensorboard import SummaryWriter

from src.reinforcement_learning.config.GlobalConfig import Config

MASTER_NAME = "Compass"
AGENT_NAME = "Agent{}"


def initialize_master_worker_paradigm(rank,
                                      world_size):
    config = Config()

    args = obtain_args(config)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # TODO changed
    print("rank____",rank)
    if rank == 0:

        # from src.reinforcement_learning.rpc_training.train_rpc_kan import TrainerRPC
        from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC

        # config = Config()

        # b) Modify config file (it is easier to input them on args if you want to do multiple experiments)
        # args = obtain_args(config)

        experiment_name = args.experiment_name
        seed = args.seed
        config.update_conf_with_args(args)

        # e) Set up the initial seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # f) Check the availability of configurations and do some prints
        print_and_assertions(config, seed)

        # g) Create summary writer
        config.savedir = os.path.abspath(config.savedir)
        os.makedirs(config.savedir, exist_ok=True)
        runs_root = os.path.join(config.savedir, "runs")
        os.makedirs(os.path.join(runs_root, "performance"), exist_ok=True)
        os.makedirs(os.path.join(runs_root, "metrics_1"), exist_ok=True)
        writer_performance = SummaryWriter(
            os.path.join(runs_root, "performance", f"performance_{experiment_name}")
        )
        writer_metrics_1 = SummaryWriter(
            os.path.join(runs_root, "metrics_1", f"metrics_{experiment_name}")
        )

        rpc.init_rpc(MASTER_NAME, rank=rank, world_size=world_size)
        trainer = TrainerRPC(config_rl=config,
                             writer_performance=writer_performance,
                             writer_metrics_1=writer_metrics_1,
                             seed=seed,
                                 experiment_name=experiment_name,
                             world_size=world_size,
                             num_gpus=args.num_gpus)

        trainer.train_agent()

        # folder = "test_model_sr"
        # seed = 1234
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # trainer.load_model_dict(config)
        # for i in range(1):
        #     test_result_dict, dm_phase_list = trainer.test_episode('RL')  #RL # Integrator
        #     se_list = test_result_dict['sr_se_test_list']



        #     # se_list = test_result_dict[0]['sr_se_test_list']
        #     sl_list = test_result_dict['sr_sl_test_list']
        #     rms_list = test_result_dict['rms_list']
        #     # np.save(folder+f"layer3_train_0.16_test0.05_se_list_inte{i}.npy",se_list)
        #
        #     np.save(folder + f"0.05_orgial_rms{i}.npy", dm_phase_list)
        #     np.save(folder + f"0.05_orgial_sr{i}.npy",se_list)
        #
        #     # np.save(folder + f"dm_phase_loss_actor4.npy", dm_phase_list)
        #     # np.save(folder+f"layer3_train_0.16_test0.05_sl_list_inte{i}.npy",sl_list)
        #     seed +=20
        #     trainer.set_seed(seed)
        #     # plt.figure()
        #     # anim = FFMpegWriter(os.path.join("output/autoencoder", 'dm_image_loss_actor_full.mp4'), framerate=10)
        #     #
        #     # for i in range(len(dm_phase_list[:100])):
        #     #
        #     #     plt.clf()
        #     #     plt.subplots_adjust(wspace=0.4, hspace=0.4)
        #     #     plt.imshow(dm_phase_list[i])
        #     #     plt.title(f"loss actor_full sr{se_list[i]}")
        #     #
        #     #
            #     anim.add_frame()
            # plt.close()
            # anim.close()

        # print("complish")
        #
        #
        # rpc.shutdown()
    #
    else:
        config = Config()
        args = obtain_args(config)
        config.savedir = os.path.abspath(config.savedir)
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # other ranks are the observer
        rpc.init_rpc(AGENT_NAME.format(rank), rank=rank, world_size=world_size)
        # rpc.shutdown()


    rpc.shutdown()
