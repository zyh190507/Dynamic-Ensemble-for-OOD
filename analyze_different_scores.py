"""
分析不同的正常度（lof、maha 等等）打分，他们的结果对比
目的是选出一个或几个最优的打分策略

每个文件（唯一的数据集 + seen labels ratio + seed）样式
strategy | layer | tuning | speedup | .....


这里首先把相同的 “数据集 + seen labels ratio + unique_strategy” 收到一起（即 数据集 + seen labels ratio + unique_strategy 构成一个 dict0 的 key）这个 dict0 的 value 也是一个 dict1
dict1 的 key 就是某个指标，value 就是不同的 seed 求得的对应该指标的结果们（是个 list）

"""
import copy
import os.path
import webbrowser
from pathlib import PurePosixPath
from typing import Dict, Any, List, Optional
from os import listdir
from os.path import isfile, join
import statistics

import pandas as pd


def get_one_result(path: str) -> Dict[str, Dict[int, Any]]:
    """
    这个数据构成是一个 {列名：{index: VALUE}}，相当于左侧额外添加了一列 index 列（0，1，2，。。。）
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    return df.to_dict()


def filter_files(root: str, prefix: str, postfix: str) -> List[str]:
    """

    :param postfix: 不包括 ".csv" 字样
    :param prefix:
    :param root:
    :return:
    """

    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

    answer = []

    if not os.path.isdir(root):
        return answer

    for directory_name in listdir(root):
        file_path = join(root, directory_name)
        if isfile(file_path) and directory_name.startswith(prefix) \
                and PurePosixPath(directory_name).stem.endswith(postfix):  # https://stackoverflow.com/a/47496703
            answer.append(file_path)

    return answer


def valid_all_datasets(paths: List[str]) -> None:
    """
    校验所有的文件，都是行列数目相同，而且格式一样
    :param paths:
    :return:
    """
    ###########################################################################
    # 先选一个参考样本
    for path in paths:
        one_result = get_one_result(path)

        # 校验防呆，保证相同行列
        column_nums = len(one_result.keys())
        columns = list(one_result.keys())
        row_nums = len(one_result['strategy'].values())
        rows = [(a, b) for a, b in zip(one_result['strategy'].values(), one_result['layer'].values())]

        # print(f'列名有 {columns}, 行名有 {rows}')
        break
    else:
        raise ValueError('no file')

    ###########################################################################
    # 其他都要求和这个样本一样
    for path in paths:
        # print('DEBUG', f'CHECKING {path}')
        cur = get_one_result(path)

        assert column_nums == len(cur.keys()), f'行列有问题'
        assert row_nums == len(cur['layer'].values()), f'行列有问题'
        assert columns == list(cur.keys())
        assert rows == [(a, b) for a, b in zip(cur['strategy'].values(), cur['layer'].values())]

    ##########################################################################

def display_in_browser(df: pd.DataFrame, dataset_name: str) -> None:
    # https://stackoverflow.com/questions/59214988/convert-csv-file-to-html-and-display-in-browser-with-pandas
    html = df.to_html()
    path = os.path.abspath(f'../tmp/{dataset_name}.html')
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(html)
    webbrowser.open(url)


def check(root: str, datasets_names: Optional[List[str]] = None, wanted_unique_strategy_name: str = 'esm_-1'):
    # 正常度打分方法
    norm_scores_methods = [
        # 'energy',
        # 'entropy',
        'knn',
        'lof_cosine',
        # 'lof_euclidean',
        # 'maha',
        # 'maxLogit',
        # 'msp',
        # 'odin'
    ]

    # 数据集名字
    if datasets_names is None:
        datasets_names = [
            'banking_0.25',
            'banking_0.75',

            'clinc_0.25',
            'clinc_0.75',

            'clincnooos_0.25',
            'clincnooos_0.75',

            'clinc_full_0',
            'clinc_small_0',

            'mcid_0.25',
            'mcid_0.75',

            'stackoverflow_0.25',
            'stackoverflow_0.75',
        ]

    # unique_strategy
    unique_strategy_names = [
        'esm_-1',
        # 'pabee_-1',
        # 'each_layer_10',
        'each_layer_2',
        'each_layer_1'
    ]

    seeds = [
        42,
        52,
        62
    ]

    metric_names = [
        'speedup',
        # 'tuning',
        'ACC-all',
        'F1',
        'F1-ood',
        'F1-ind'
    ]

    # 校验所有的文件
    all_paths = [path for norm_scores_method in norm_scores_methods for dataset_name in datasets_names for path in filter_files(root, f'{norm_scores_method}_{dataset_name}', '')]

    # 有的数据集没有训练，剔除就好了
    datasets_names = [dataset_name for dataset_name in datasets_names if any(dataset_name in path for path in all_paths)]

    if len(all_paths) == 0:
        return
    #
    # print(f'DEBUG: 一共有 {len(all_paths)} 文件')
    # print(f'DEBUG: 评估指标包含 {metric_names}')

    # 指标 eval: ACC-all,F1,F1-ood,F1-ind
    valid_all_datasets(all_paths)

    # 一次性全读出来
    files = {path: get_one_result(path) for path in all_paths}

    # 处理成 dict['{数据集}+{seen labels ratio}+{unique_strategy_name}+{norm_scores_method}', dict[metric_name, list[float | str | ...]]]
    files_results = {}
    for dataset_name in datasets_names:  # 这里其实就暗含了 seen labels ratio，例如 'banking_0.25' 等
        for unique_strategy_name in unique_strategy_names:  # unique_strategy 例如 'esm_-1' 'each_layer_1' 等
            for norm_scores_method in norm_scores_methods:  # 正常度打分方法: 'energy' 'odin' 等

                metrics = {}

                for seed in seeds:
                    path = os.path.join(root, f'{norm_scores_method}_{dataset_name}_{seed}.csv')  # 自己手动组配文件名，别懒逼
                    file_result = files[path]

                    # 牢记一点，这个数据构成是一个 {列名：{index: VALUE}}，相当于左侧额外添加了一列 index 列（0，1，2，。。。）
                    line_nums = max(file_result['strategy'].keys())

                    # 加一个校验，防止行和行互换之类的
                    for line_num in range(0, line_nums+1):
                        cur_line_unique_strategy_name = f'{file_result["strategy"][line_num]}_{file_result["layer"][line_num]}'
                        if cur_line_unique_strategy_name == unique_strategy_name:
                            wanted_line_num = line_num
                            break
                    else:
                        raise ValueError(f"NO {unique_strategy_name}")

                    for metric_name in metric_names:
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(file_result[metric_name][wanted_line_num])

                files_results[f'{dataset_name}+{unique_strategy_name}+{norm_scores_method}'] = metrics

    # print('cur_dataset_result', files_results)
    #################################################################################################################################

    # 绘制表格
    # wanted_norm_scores_methods = ['knn', 'lof_cosine']
    wanted_norm_scores_methods = ['lof_cosine']

    for dataset_name in datasets_names:  # 这里其实就暗含了 seen labels ratio，例如 'banking_0.25' 等
        df_answer: Dict[str, Dict[str, Any]] = {}
        for norm_scores_method in wanted_norm_scores_methods:  # 'knn', 'lof_cosine' ...
            cur_result = files_results[f'{dataset_name}+{wanted_unique_strategy_name}+{norm_scores_method}']
            for metric_name in metric_names:
                if metric_name not in df_answer:
                    df_answer[metric_name] = {}
                # （针对不同的 seed）
                assert len(cur_result[metric_name]) == 3

                # df_answer[metric_name][f'{norm_scores_method}_42'] = cur_result[metric_name][0]
                # df_answer[metric_name][f'{norm_scores_method}_52'] = cur_result[metric_name][1]
                # df_answer[metric_name][f'{norm_scores_method}_62'] = cur_result[metric_name][2]

                if isinstance(cur_result[metric_name][0], str):
                    assert cur_result[metric_name][0] == cur_result[metric_name][1] and cur_result[metric_name][1] == cur_result[metric_name][2]
                    df_answer[metric_name][f'{norm_scores_method}_avg'] = cur_result[metric_name][0]
                    df_answer[metric_name][f'{norm_scores_method}_var'] = 'NONE'
                else:
                    df_answer[metric_name][f'{norm_scores_method}_avg'] = statistics.mean(cur_result[metric_name])
                    df_answer[metric_name][f'{norm_scores_method}_var'] = statistics.variance(cur_result[metric_name])

                # 加一行区分
                df_answer[metric_name][f'-{norm_scores_method}-'] = '-'

        # 保留两位小数
        df_answer_ = copy.deepcopy(df_answer)
        for key0, values0 in df_answer.items():
            for key1, value1 in values0.items():
                if isinstance(value1, float):
                    df_answer_[key0][f'{key1}:.2f'] = f'{round(value1, 2):.2f}'
        df_answer = copy.deepcopy(df_answer_)

        # 每个 dataset 打印一次（总结一次）
        df = pd.DataFrame(df_answer)

        print(dataset_name, root)
        print(df)

        display_in_browser(df, f'{dataset_name}')

        # import json
        # print(json.dumps(df_answer, indent=4))

        print('-' * 80)


def main():
    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500
    pd.options.display.expand_frame_repr = False  # https://stackoverflow.com/a/45709154

    # BEST!

    bests = {
        'banking_0.25': './model_output/banking_0.25/ad30.10.2_lr2.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize16__lambda0.1__scale1.51.2',
        'banking_0.75': './model_output/banking_0.75/ad30.10.2_lr2.0e-05__epoch50__lossce_and_div__batchsize32__lambda0.1__scale1.51.4',

        'stackoverflow_0.25': './model_output/stackoverflow_0.25/ad20.150.4_lr2.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.51.4',
        'stackoverflow_0.75': './model_output/stackoverflow_0.75/ad30.10.3_lr2.0e-05__epoch50__lossce_and_div__batchsize32__lambda0.1__scale1.41.5',

        'clinc_0.25': './model_output/clinc_0.25/ad30.150.4_lr5.0e-06__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.81.7',
        'clinc_0.75': './model_output/clinc_0.75/ad30.10.2_lr5.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.5'
    }

    for dataset_name, path in bests.items():
        # print(dataset_name)
        check(path, [dataset_name], 'esm_-1')
        # check(path, [dataset_name], 'pabee_-1')
        print('#' * 80)

    # root = './model_output/search_stackoverflow_0.75_3090/ad30.10.3_lr2.0e-05__epoch50__lossce_and_div__batchsize32__lambda0.1__scale1.41.5'
    # check(root, wanted_unique_strategy_name='esm_-1')
    # check(root, wanted_unique_strategy_name='pabee_-1')
    #
    # root = './model_output/search_banking_0.75/'
    #
    # for directory_name in listdir(root):
    #     file_path = join(root, directory_name)
    #     if os.path.isdir(file_path):
    #         print(file_path)
    #         print('esm:')
    #         check(file_path, ['banking_0.75'], wanted_unique_strategy_name='esm_-1')
    #         print('pabee:')
    #         check(file_path, ['banking_0.75'], wanted_unique_strategy_name='pabee_-1')
    #         print('*' *90)


if __name__ == '__main__':
    main()
