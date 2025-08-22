# from src.error_budget.error_budget_multiple_agents import ExperimentManager
from src.autoencoder.obtain_dataset_noise_image import ExperimentManager
import argparse
import numpy as np
import random
import data.par.par4rl.production.production_sh_40x40_8m_3layers_d0_noise

#这是一个测试


if __name__ == "__main__":
    # exp = ExperimentManager()
    # exp.run()

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_file', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_filtered', type=int, default=5)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    exp = ExperimentManager(args)
    exp.run()

