import copy
from typing import List, Dict, Any

# import matplotlib
import torch

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score


def harmonic_mean(data: List[float]) -> float:  # 计算调和平均数
    total = 0
    for i in data:
        if i == 0:  # 处理包含0的情况
            return 0
        total += 1/i
    return len(data)/total


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    """
    https://zhuanlan.zhihu.com/p/165219346

    利用 detach 生成一个与 current graph 无关的 tensor（Returned Tensor shares the same storage with the original one），
    然后 cpu 转化设备，最后转为 numpy
    注意：tensor.detach().cpu() 不一定执行 copy 操作（Tensor.cpu 提及 If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned）
    :param x: 可以不限设备等
    :return:
    """
    return x.detach().cpu().numpy()


def au_sklearn(num_labels_ind: int, y_true: np.ndarray, y_prob: np.ndarray, round_control: int = 2) -> Dict[str, Any]:
    """
    仅考虑模型的 id 和 ood 的分界能力

    https://zhuanlan.zhihu.com/p/35583721
    https://blog.csdn.net/leokingszx/article/details/121105381

    :param round_control:
    :param num_labels_ind: ood 对应的 label id
    :param y_true: shape: [N,]; 表示预测的 labels（有 ood label）
    :param y_prob: shape: [N]; 表示第 layer_index 层的那个正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常
    :return:
    """
    # ind 设为正样本
    y_true_ = copy.deepcopy(y_true)
    oos_mask = (y_true_ == num_labels_ind)
    y_true_[oos_mask] = 0
    y_true_[~oos_mask] = 1

    # 正常度越高，越可能 ind
    y_prob_ = y_prob

    aupr_oos = round(roc_auc_score(y_true_, y_prob_) * 100, round_control)

    aupr_in = round(average_precision_score(y_true_, y_prob_) * 100, round_control)

    ########################################################################

    # ood 设为正样本
    y_true_ = copy.deepcopy(y_true)
    oos_mask = (y_true_ == num_labels_ind)
    y_true_[oos_mask] = 1
    y_true_[~oos_mask] = 0

    # 异常度越高（正常度越低；正常度取负号，就是异常度），越可能 ind
    y_prob_ = -y_prob

    aupr_ood = round(average_precision_score(y_true_, y_prob_) * 100, round_control)

    return {'auroc': aupr_oos, 'aupr_in': aupr_in, 'aupr_ood': aupr_ood}


def F_measure(pred: torch.Tensor, labels: torch.Tensor, round_setting: int = 2) -> Dict[str, np.float64]:
    """

    :param round_setting:
    :param pred: shape: [N,]; 表示预测的 labels（有 ood label）
    :param labels: shape: [N]; torch.int64;  ground truth 的 labels
    :return:
    """
    y_pred = tensor2numpy(pred)
    y_true = tensor2numpy(labels)

    # https://blog.csdn.net/m0_38061927/article/details/77198990
    # shape: [nums_labels', nums_labels'] 注意这里 nums_labels' 是 confusion_matrix 统计得到的
    # cm[i][j] 表示真实分类为 i 预测为 j 的样本数目
    # 这里的 label 从小到大排序的，要保证 ood label 最大！
    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    # 预测对了就判断一个正，正的数目 除以 样本总数目
    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    acc = round(accuracy_score(y_true, y_pred) * 100, round_setting)

    recalls, precisions, f1s = [], [], []

    n_class = cm.shape[0]
    for index in range(n_class):
        tp = cm[index][index]
        # 查全率
        recall = tp / cm[index].sum() if cm[index].sum() != 0 else 0
        # 查准率
        precision = tp / cm[:, index].sum() if cm[:, index].sum() != 0 else 0

        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
        
        recalls.append(recall * 100)
        precisions.append(precision * 100)
        f1s.append(f1 * 100)

    f1 = np.mean(f1s).round(round_setting)
    f1_seen = np.mean(f1s[:-1]).round(round_setting)
    f1_unseen = round(f1s[-1], round_setting)  # 认为 OOD 为正样本，其余（即 IND）为负样本，求得 F1

    results = {'ACC-all': acc, 'F1': f1, 'F1-ood': f1_unseen, 'F1-ind': f1_seen}

    return results


def estimate_best_threshold(cur: Dict[str, float], best_results: Dict[str, float], strategy: str) -> bool:
    """
    帮助找到 threshold
    """
    assert strategy in ['ALL', 'SUM', 'HARMONIC']

    if len(best_results) == 0:
        return True

    # 比较 'F1_IND', 'F1_OOD'（主要在这两个点跑到 sota）
    best_metrics_names = ['F1_IND', 'F1_OOD']
    if strategy == 'ALL':
        return all(cur[metric_name] >= best_results[metric_name] for metric_name in best_metrics_names)

    # 原来要求的 ALL 大于等于感觉苛刻了一些，取平均吧
    if strategy == 'SUM':
        return sum(cur[metric_name] for metric_name in best_metrics_names) >= sum(best_results[metric_name] for metric_name in best_metrics_names)
    # 调和平均
    assert strategy == 'HARMONIC'
    return harmonic_mean([cur[metric_name] for metric_name in best_metrics_names]) >= harmonic_mean([best_results[metric_name] for metric_name in best_metrics_names])


# def plot_oos_ind(ind_val, oos_val):
#     x_ind_val = np.array(range(len(ind_val)))
#     x_oos_val = np.array(range(len(oos_val)))
#
#     plt.scatter(x_ind_val, np.array(ind_val), c='green')
#     plt.scatter(x_oos_val, np.array(oos_val), c='red')
#
#     plt.show()
