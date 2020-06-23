import os
import numpy as np
from numpy import genfromtxt
from scipy import signal
from scipy.interpolate import make_interp_spline, BSpline
import pickle

import matplotlib
# matplotlib.use('Agg')
matplotlib.use('macosx')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Visualize Training Log')
parser.add_argument('--sample', '-s', action='store_true',
                    help='Whether to perform sample')
parser.add_argument('--avg', '-avg', action='store_true',
                    help='Whether to show average data')
parser.add_argument('--val_eposide', '-val', default=100, type=int,
                    help='Number of eposide used in each training epoch')
parser.add_argument('--max_epoch', '-max', default=None, type=int,
                    help='')
args = parser.parse_args()
print(args)

# -----------------
file_name_list = ['loss', 'train-acc', 'test-acc']
n_row = len(file_name_list)
use_avg = args.avg
# -----------------

method_list = {

    # -------
    # Pruning
    # -------
    # 'CR-8': {
    #     'path_root': "./Results/BERT-GLUE-MRPC/Prune/strbert/runs-CR-8",
    #     'color': 'b',
    #     'marker': '-',
    #     'legend': 'CR-8'
    # },
    #
    # 'CR-50': {
    #     'path_root': "./Results/BERT-GLUE-MRPC/Prune/strbert/runs-CR-50",
    #     'color': 'r',
    #     'marker': '-',
    #     'legend': 'CR-50'
    # },
    #
    # 'CR-25': {
    #     'path_root': "./Results/BERT-GLUE-MRPC/Prune/strbert/runs-CR-25",
    #     'color': 'g',
    #     'marker': '-',
    #     'legend': 'CR-25'
    # },
    #
    # 'CR-100': {
    #     'path_root': "./Results/BERT-GLUE-MRPC/Prune/strbert/runs-CR-100",
    #     'color': 'k',
    #     'marker': '-',
    #     'legend': 'CR-100'
    # },
    #
    'FP': {
        'path_root': "./Results/BERT-GLUE-MRPC/runs-full-precision",
        'color': 'c',
        'marker': '-',
        'legend': 'Full-Precision'
    },

    # --------
    # Quantize
    # --------
    'bitW-8':{
        'path_root': "./Results/BERT-GLUE-MRPC/Quant/qbert/runs-bitW-8",
        'color': 'r',
        'marker': '-',
        'legend': 'bitW-8'
    },

    'bitW-4':{
        'path_root': "./Results/BERT-GLUE-MRPC/Quant/qbert/runs-bitW-4",
        'color': 'b',
        'marker': '-',
        'legend': 'bitW-4'
    }
}

plt.figure()

for method_name, method_info in method_list.items():

    if not os.path.exists(method_info['path_root']):
        print('%s not found' %method_name)
        continue

    for idx, data_name in enumerate(file_name_list):

        ax = plt.subplot(n_row, 1, idx + 1)

        training_log_path = method_info['path_root']

        data = genfromtxt('%s/%s.txt' %(training_log_path, data_name), delimiter=',')

        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        if data_name in ['test-acc']:
            data = data[: args.max_epoch , 1:] # [3000, 100]
        else:
            if args.max_epoch is None:
                end = None
            else:
                end = args.max_epoch * args.val_eposide
            if data.shape[1] == 2:
                data = data[:end, 1]
            else:
                data = data[:end, 2] if use_avg else data[:end, 1]

        if args.sample and 'test' not in data_name:
            sample_idx = np.linspace(0, len(data) - 1, num=100, dtype=int)
            data = data[sample_idx]

        xaxis = range(len(data))
        # print(xaxis)

        ax.plot(xaxis, data, color=method_info['color'], linestyle=method_info['marker'],
                label=method_info['legend'], markersize=1)

        # if data_name not in ['test-acc']:
        #     ax.plot(xaxis, data, color=method_info['color'], linestyle=method_info['marker'],
        #             label=method_info['legend'], markersize=1)
        # else:
        #     mean = np.mean(data, axis=1) # [3000, 1]
        #     std = np.std(data, axis=1) # [3000, 1]
        #     ax.errorbar(xaxis, mean, yerr = std, color=method_info['color'], linestyle=method_info['marker'],
        #             label=method_info['legend'], markersize=1)

        # Get test data
        # if data_name in ['test-acc', 'test-top1-acc', 'test-top5-acc']:
        #     last_ten_acc_mean = np.mean(data[-10:, :])
        #     last_ten_acc_std = np.std(data[-10:, :])
        #     print('[%30s] [%10s] %.3f(%.3f)'
        #           % (method_name, data_name, last_ten_acc_mean, last_ten_acc_std))

        ax.set_ylabel(data_name)
        ax.set_ylabel(data_name, fontsize=15)
        ax.set_xlabel('epoch', fontsize=15)
        if data_name in ['loss', 'constraint']:
            ax.set_yscale('log')
        ax.grid()

        if 'loss' in file_name_list:
            if data_name == 'loss':
                ax.legend(prop={'size': 13})

        else:
            if data_name == 'test-acc':
                ax.legend(prop={'size': 13})
        ax.legend(prop={'size': 13})
plt.show()