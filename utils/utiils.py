import os
from torch.utils.tensorboard import SummaryWriter

def get_log_dir(run_name):
    return f'tensorboard_logs/{run_name}'

def initialize_writer(run_name):
    log_dir = get_log_dir(run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return SummaryWriter(log_dir)
