import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from utils.utils import NLLSurvLoss, cox_log_rank
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.utils import resample
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import *

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='UCEC/MiCo/MiCo.yaml', type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = make_parse()

    config = args.config
    cfg = read_yaml(config)

    config = f'{args.config}'
    # Cls/BRCA/AMI/TMIL.yaml
    task = config.split('/')[0]  # Cls
    dataset = config.split('/')[1]  # BRCA
    model_name = config.split('/')[2]  # MiCo
    version_identifier = config.split('/')[3].replace('.yaml', '')  # AMIL

    log_path = os.path.join('./logs/', task, dataset, model_name, version_identifier)

    auc_lst = []
    acc_lst = []
    f1_lst =  []
    for i in range(4):
        log_path = Path(log_path)
        _log_path = log_path / f'fold{str(i)}'

        metric = pd.read_csv(_log_path / 'test_metrics.csv')
        print(metric)
        auc_lst.append(metric['auc'].values[0])
        acc_lst.append(metric['acc'].values[0])
        f1_lst.append(metric['f1'].values[0])

    # 得到平均值和标准差
    auc_mean = np.mean(auc_lst)
    acc_mean = np.mean(acc_lst)
    f1_mean = np.mean(f1_lst)
    auc_std = np.std(auc_lst)
    acc_std = np.std(acc_lst)
    f1_std = np.std(f1_lst)

    print(f'acc_mean: {acc_mean:.4f}+{acc_std:.4f}')
    print(f'f1_mean: {f1_mean:.4f}+{f1_std:.4f}')
    print(f'auc_mean: {auc_mean:.4f}+{auc_std:.4f}')

