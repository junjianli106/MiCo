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
import comet_ml
from glob import glob

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='UCEC/MiCo/MiCo.yaml', type=str)
    args = parser.parse_args()
    return args

def get_metrics(log_path, dataset):

    if len(list(glob(str(log_path) + '/*/all_risk_scores.npz'))) != 0:
        all_risk_scores_list = list(glob(str(log_path) + '/*/all_risk_scores.npz'))
        all_censorships_list = list(glob(str(log_path) + '/*/all_censorships.npz'))
        all_event_times_list = list(glob(str(log_path) + '/*/all_event_times.npz'))
        all_csv_list = list(glob(str(log_path) + '/*/result.csv'))
    else:
        all_risk_scores_list = list(glob(str(log_path) + '/*/*/all_risk_scores.npz'))
        all_censorships_list = list(glob(str(log_path) + '/*/*/all_censorships.npz'))
        all_event_times_list = list(glob(str(log_path) + '/*/*/all_event_times.npz'))
        all_csv_list = list(glob(str(log_path) + '/*/*/result.csv'))
    print(len(all_risk_scores_list), len(all_censorships_list), len(all_event_times_list), len(all_csv_list))
    number = len(all_event_times_list)

    # ---->C-index
    c_index_list = []
    c_index_high_list = []
    c_index_low_list = []

    for i in range(number):
        df_fold = pd.read_csv(all_csv_list[i], index_col=0)
        # print(df_fold.columns)
        # df_fold = df_fold.set_index('0')
        df_fold = df_fold.T
        c_index_list.append(df_fold['c_index'])
        c_index_high_list.append(df_fold['c_index_high'])
        c_index_low_list.append(df_fold['c_index_low'])

    c_index = round(np.mean(c_index_list), 4)
    c_index_high = round(np.mean(c_index_high_list), 4)
    c_index_low = round(np.mean(c_index_low_list), 4)
    c_index_std = round(np.std(c_index_list), 4)

    # ---->Save each result_list
    all_censorships_high = []
    all_censorships_low = []
    all_event_times_high = []
    all_event_times_low = []

    for i in range(number):
        hazardsdata = np.load(all_risk_scores_list[i], allow_pickle=True)['arr_0'].tolist()
        median = np.median(hazardsdata)
        hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
        hazards_dichotomize[hazardsdata < median] = 1
        idx = hazards_dichotomize == 0

        all_censorships = np.load(all_censorships_list[i], allow_pickle=True)['arr_0']
        all_event_times = np.load(all_event_times_list[i], allow_pickle=True)['arr_0']

        all_censorships_high = all_censorships_high + all_censorships[idx].tolist()
        all_censorships_low = all_censorships_low + all_censorships[~idx].tolist()
        all_event_times_high = all_event_times_high + all_event_times[idx].tolist()
        all_event_times_low = all_event_times_low + all_event_times[~idx].tolist()

    all_censorships_high = 1 - np.array(all_censorships_high)
    all_censorships_low = 1 - np.array(all_censorships_low)
    all_event_times_high = np.array(all_event_times_high)
    all_event_times_low = np.array(all_event_times_low)

    print(len(all_censorships_high), len(all_censorships_low), len(all_event_times_high), len(all_event_times_low))
    results = logrank_test(all_event_times_high, all_event_times_low, event_observed_A=all_censorships_high,
                           event_observed_B=all_censorships_low)
    pvalue_pred = results.p_value

    print('-' * 29)
    print(f'          {dataset}')
    pvalue_pred = format(pvalue_pred, '.3g')
    print('          P-Value')
    print('-' * 29)
    print(f'p_value  {pvalue_pred}')
    print()

    # ---->Save all metrics as csv
    dict = {'c_index': c_index, 'c_index_high': c_index_high, 'c_index_low': c_index_low, 'c_index_std': c_index_std,
            'p_value': pvalue_pred}
    result = pd.DataFrame(list(dict.items()))
    result.to_csv(log_path + '/result_all.csv')

    # ---->predict output
    print('          Value (± std)')
    print('-' * 29)
    if float(pvalue_pred) < 0.05:
        print(f'c_index: {c_index}(±{c_index_std})*')
        # return rf'{c_index}(±{c_index_std})*'
    else:
        print(f'c_index: {c_index}(±{c_index_std})')
        # return rf'{c_index}(±{c_index_std})'
    return c_index, c_index_std, float(pvalue_pred)


# 分测试单个数据集和多个数据集
# 单个数据集 python metrics.py --config UCEC/AMIL/AMIL.yaml
# 多个数据集 python metrics.py --config AMIL/AMIL.yaml


if __name__ == '__main__':
    args = make_parse()

    task = args.config.split('/')[0]
    dataset_name = args.config.split('/')[1]  # os.path.split()[0]  # BLCA
    model_name = args.config.split('/')[2]  # AMIL
    version_identifier = args.config.split('/')[3].replace('.yaml', '')  # AMIL

    log_path = os.path.join('./logs/', dataset_name, model_name, version_identifier)
    _, _, _ = get_metrics(log_path, dataset_name)






