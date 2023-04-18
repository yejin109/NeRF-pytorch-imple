import os
import torch
import datetime

os.environ['EXP_NAME'] = '-'.join(['TEST', 'SYNC', str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))])
os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
os.mkdir(os.environ['LOG_DIR'])
os.environ['VERBOSE'] = "0"
os.environ['DEVICE'] = 'cuda:0'
# os.environ['DEVICE'] = 'cpu'

torch.set_default_tensor_type('torch.cuda.FloatTensor')
