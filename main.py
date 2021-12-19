import os

from network_run_scp import *

if __name__ == "__main__":
    for alpha in [0.1, 0.3, 0.5]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备 优先在GPU上运行
        one_run(run_time="t{}".format(str("t1")), alpha=alpha)
