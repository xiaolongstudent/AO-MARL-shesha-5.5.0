# ... existing imports ...
import argparse
import numpy as np
import random
import torch  # 新增torch导入

# ... existing argument parser code ...
args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)  # 新增PyTorch随机种子设置
    # 初始化分布式设置（根据实际情况选择后端）
torch.distributed.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

exp = ExperimentManager(args)
exp.run()