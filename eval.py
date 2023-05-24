import copy
import random
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import os.path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

import utils
from utils import F_measure, tensor2numpy


class Evaluator:
    METHOD = 'NONE'

    def __init__(self,
                 model: torch.nn.Module,
                 root: str,
                 file_postfix: str,
                 dataset_name: str,
                 device: torch.device,
                 num_labels: int,
                 tuning: str,
                 scale_ind: float,
                 scale_ood: float,
                 valid_all_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 model_forward_cache: Optional[Dict[DataLoader, Tuple[torch.Tensor, ...]]] = None
                 ):
        """

        :param model:
        :param dataset_name: 数据名字
        :param device:
        :param num_labels: num_labels 包含了 oos
        :param tuning: valid_all/valid；valid_all 表示使用 val 数据集里面的 ood 数据（ind + ood），而 valid 表示只用 ind 数据
        """
        self.dataset_name = dataset_name
        self.valid_all_dataloader = valid_all_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = model
        self.tuning = tuning
        self.tpr = 95
        self.scale_ind = scale_ind
        self.scale_ood = scale_ood
        self.device = device

        output_file = os.path.join(root, f'{self.METHOD}_{file_postfix}')
        self.output_file = output_file

        self.num_labels = num_labels
        # 不含 oos
        self.num_labels_IND = self.num_labels - 1

        if model_forward_cache is None:
            self.model_forward_cache = {}
        else:
            self.model_forward_cache = model_forward_cache


    def model_forward_with_cache(self, dataloader: DataLoader) -> Tuple[torch.Tensor, ...]:
        """
        在 model_forward 上面加上 cache 功能，原因是 model_forward 仅仅是模型本身的东西（total_feats、total_logits、total_labels）
        与具体的正常度打分函数（knn lof entropy 等）无关，所以连续对同一数据集，进行不同的正常度打分的时候，无需重复 infer 操作
        :return:
        """
        if dataloader in self.model_forward_cache:
            # 防止有打分函数实现中，修改了 item 的数值
            return tuple((item.clone().to(self.device)) for item in self.model_forward_cache[dataloader])
        
        result = self.model_forward(dataloader)
        # 防止有打分函数实现中，修改了 item 的数值
        self.model_forward_cache[dataloader] = tuple((item.clone().cpu()) for item in result)
        return result


    def model_forward(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        跑一次模型

        :param dataloader:
        :return:
            # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能二次映射了）
            # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
            # shape: [N] torch.int64  ground truth 的 labels
        """
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量
        total_feats = None
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        total_logits = None
        # shape: [N] torch.int64  ground truth 的 labels
        total_labels = None
        self.model.eval()
        with torch.no_grad():
            with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    " + ",
                    TimeRemainingColumn(),
            ) as progress:
                len_dataloader = len(dataloader)
                epoch_tqdm = progress.add_task(description="epoch progress", total=len(dataloader))
                for step, batch in enumerate(dataloader, start=1):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    labels = batch['labels']
                    feats, logits = self.model(batch, mode='eval')
                    feats, logits = feats.double(), logits.double()
                    # 适配特殊情况，即只使用最后一层作为 features、logits，此时输出为 [N, hidden_size=768] 和 [N, num_labels-1]
                    # 改成 [layers_nums=1, N, hidden_size=768] 和 [layers_nums=1, N, num_labels-1]
                    if feats.dim() == 2 and logits.dim() == 2:
                        feats = feats.unsqueeze(dim=0)
                        logits = logits.unsqueeze(dim=0)

                    if total_feats is not None:
                        total_feats = torch.cat((total_feats, feats), dim=1)
                        total_logits = torch.cat((total_logits, logits), dim=1)
                        total_labels = torch.cat((total_labels, labels))
                    else:
                        total_feats = feats
                        total_logits = logits
                        total_labels = labels

                    progress.update(epoch_tqdm, advance=1, description=f'test_Evaluator - {step:03d}/{len_dataloader:03d}')

        return total_feats, total_logits, total_labels

    def search_threshold_by_valid_all(self, scores: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        如果可见 ood 的 val 数据，threshold 应该这么找

        对每一层，
        preds[layer_index] 中假设所有数据的 labels 都是 IND，然后针对每一层，去遍历可能的 threshold，把该层 scores[layer_index] 中低于改阈值的设为 OOD（预测的 preds[layer_index] 对应 label 作废，改为 OOD，其余 label 不变）
        看看哪个阈值效果最优

        :param scores: shape: [layers_nums, N,]; scores[layer_index, :] 表示第 layer_index 层的某正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常
        :param labels: shape: [N]; torch.int64;  ground truth 的 labels
        :param preds: shape: [layers_nums, N,];  表示利用第 layer_index 层的模型输出，预测的 ind 的 labels
        :return: shape: [layers_nums]; 每层的最佳 threshold（仅用该层的 preds[layer_index] 和该层的正常度打分 scores[layer_index]，找该层的最佳 threshold，使得 ind 和 ood 指标都比较好）
        """
        thresholds = []
        for layer_index, cur_layer_scores in enumerate(scores):  # cur_layer_scores shape: [N,]; 表示第 layer_index 层的某正常度打分
            # 仅提取那些分类在 IND 的数据，对应的打分
            # shape: [N',]
            scores_ind = cur_layer_scores[labels != self.num_labels_IND]

            # 设置 threshold 的筛选区间
            # left, right = min(scores_ind).cpu(), max(scores_ind).cpu()
            left: torch.Tensor = sorted(scores_ind)[round(len(scores_ind)*(1-0.96))].cpu()  # scalar (zero-dimensional tensor)
            right: torch.Tensor = sorted(scores_ind)[round(len(scores_ind)*(1-0.7))].cpu()  # scalar (zero-dimensional tensor)

            # shape: [N,]; 表示利用第 layer_index 层的模型输出，预测的 ind 的 labels
            cur_layer_preds = preds[layer_index]

            best_f1 = -1
            best_acc = -1
            best_threshold = -1
            for threshold in np.linspace(left, right, 400):
                # 对于过低的正常程度打分的句子，把预测的 label 作废，改为 oos
                new_pred = copy.deepcopy(cur_layer_preds)
                new_pred[cur_layer_scores < threshold] = self.num_labels_IND

                res = F_measure(new_pred, labels)
                if res['F1'] > best_f1 and res['ACC-all'] > best_acc:
                    best_threshold = threshold
                    best_f1 = res['F1']
                    best_acc = res['ACC-all']
            thresholds.append(best_threshold)

        return torch.tensor(thresholds).to(self.device)

    def search_threshold_by_valid(self, scores: torch.Tensor) -> torch.Tensor:
        """
        如果不可见 ood 的 val 数据，threshold 应该这么找

        对每一层，
        排序该层的正常度打分 cur_layer_scores，取较低的分数，作为 threshold

        :param scores: shape: [layers_nums, N,]; scores[layer_index, :] 表示第 layer_index 层的某正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常
        :return: shape: [layers_nums]; 每层的最佳 threshold（仅用该层的 preds[layer_index] 和该层的正常度打分 scores[layer_index]，找该层的最佳 threshold，使得 ind 和 ood 指标都比较好）
        """
        return torch.stack([sorted(cur_layer_scores)[round(len(cur_layer_scores) * (1 - self.tpr*0.01))] for cur_layer_scores in scores])

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param dataloader:
        :return:

            shape: [layers_nums, N,]
            score[layer_index, :] 表示第 layer_index 层的那个正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常

            利用 argmax probability/logits 得到 predict_ind_labels，这里先假设都是 IND 数据
            shape: [layers_nums, N]
            preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 ind 的 labels

            shape: [N] torch.int64  ground truth 的 labels（包含 ood）
        """
        pass

    def get_ensemble_pred(self, total_preds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        其实就是投票计数
        :param total_preds:
            # shape: [layers_nums, N]
            # preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 labels（包含 ood）
        :return:
            # shape: [N] torch.int64  预测的 labels（包含 ood）
        """
        t_ind = self.scale_ind
        t_ood = self.scale_ood
        k = 0.75
        n_layers, n = total_preds.shape

        base = [t_ind] * self.num_labels_IND
        base.append(t_ood)
        esm_pred = [-1] * n
        exit_layer_index = [-1] * n
        for data_index in range(n):
            # 该模型每一层，针对某个数据的预测情况
            # shape: [layers_nums, ]
            preds = total_preds[:, data_index]
            vote = [0] * (self.num_labels_IND + 1)
            for layer_index, label in enumerate(preds):
                vote[label] += 1
                if vote[label] >= base[label] * pow(layer_index+1, k):
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1
                    break
                elif layer_index == n_layers - 1:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        speed_up = n_layers / np.mean(exit_layer_index)
        return esm_pred, speed_up

    def get_pabee_pred(self, total_preds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        pabee 方法，如果连续的一系列“判定器”，他们的输出结果都相同，且判定器数目超过了一定阈值(patience)，则提前 exit
        :param total_preds:
            # shape: [layers_nums, N]
            # preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 labels（包含 ood）
        :return:
            # shape: [N] torch.int64  预测的 labels（包含 ood）
        """
        patience = 4
        layers_nums, n = total_preds.shape

        esm_pred = [-1] * n
        exit_layer_index = [-1] * n  # 从第一层开始计数（第一层、第二层。。。。）
        for data_index in range(n):
            # 该模型每一层，针对某个数据的预测情况
            # shape: [layers_nums, ]
            preds = total_preds[:, data_index]
            last_label = preds[0]
            count = 0
            for layer_index, label in enumerate(preds):
                count = count + 1 if label == last_label else 0
                if count >= patience:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1
                    break
                elif layer_index == layers_nums - 1:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        pabee_speedup = layers_nums / np.mean(exit_layer_index)
        return esm_pred, pabee_speedup

    def get_random_pred(self, total_preds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        其实就是每个数据，随机选一层的分类器结果，作为预测结果
        :param total_preds:
            # shape: [layers_nums, N]
            # preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 labels（包含 ood）
        :return:
            # shape: [N] torch.int64  预测的 labels（包含 ood）
        """
        n_layers, n = total_preds.shape

        esm_pred = [-1] * n
        exit_layer_index = [-1] * n
        for data_index in range(n):
            # 该模型每一层，针对某个数据的预测情况
            # shape: [layers_nums, ]
            preds = total_preds[:, data_index]

            # 随机选一层的分类器结果，作为预测结果
            choosen_layer_index = random.randrange(0, n_layers)
            esm_pred[data_index] = preds[choosen_layer_index]
            exit_layer_index[data_index] = choosen_layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        speed_up = n_layers / np.mean(exit_layer_index)
        return esm_pred, speed_up

    def get_threshold_score(self, mode: str) -> torch.Tensor:
        """

        :param mode: valid_all/valid；valid_all 表示使用 val 数据集里面的 ood 数据（ind + ood），而 valid 表示只用 ind 数据
        :return: shape: [layers_nums]; 每层的最佳 threshold（仅用该层的相关输出，找该层的最佳 threshold，使得 ind 和 ood 指标都比较好）
        """
        dataloader: DataLoader = self.valid_all_dataloader if mode is 'valid_all' else self.valid_dataloader

        # shape: [layers_nums, N,]
        # score[layer_index, :] 表示第 layer_index 层的那个正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常

        # 利用 argmax probability/logits 得到 predict_ind_labels，这里先假设都是 IND 数据
        # shape: [layers_nums, N]
        # preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 ind 的 labels

        # shape: [N] torch.int64  ground truth 的 labels（包含 ood）
        valid_score, valid_pred, valid_labels = self.predict_ind_labels(dataloader)

        if mode is 'valid_all':
            # 如果可见 val 的 ood 数据，暴力搜索最优 threshold
            return self.search_threshold_by_valid_all(valid_score, valid_labels, valid_pred)
        else:
            # 对每一层的 threshold，排序该层的正常度打分 cur_layer_scores，取较低的分数，作为 threshold
            return self.search_threshold_by_valid(valid_score)

    def _write(self, res: Dict[str, Any]):
        """
        写入数据
        :param res:
        :return:
        """
        if os.path.exists(self.output_file):
            df = pd.read_csv(self.output_file)
            new = pd.DataFrame(res, index=[1])
            df = df.append(new, ignore_index=True)
            df.to_csv(self.output_file, index=False)
        else:
            new = [res]
            df = pd.DataFrame(new)
            df.to_csv(self.output_file, index=False)
        print(res)

    def eval(self):
        #######################################################################################################
        # 把 test 数据集放入 model，
        #             运行 self.predict_ind_labels 完后，可以得到 test_scores（x_i 正常度打分） 和 test_preds（预测的 ind 的 labels） 和 test_labels（ground truth labels，包含 ood）
        #             设法找到 test_scores 这个正常度打分中的 ind 和 ood 分割的阈值，
        #             test_preds 中，若某个 x_i 正常度过低，覆写为 OOD 这个 label

        # shape: [layers_nums, N,]
        # score[layer_index, :] 表示第 layer_index 层的那个正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常

        # 利用 argmax probability/logits 得到 predict_ind_labels，这里先假设都是 IND 数据
        # shape: [layers_nums, N]
        # preds[layer_index, :] 表示利用第 layer_index 层的模型输出，预测的 ind 的 labels

        # shape: [N] torch.int64  ground truth 的 labels（包含 ood）
        test_scores, test_preds, test_labels = self.predict_ind_labels(self.test_dataloader)

        # shape: [layers_nums]; 每层的最佳 threshold（仅用该层的相关输出，找该层的最佳 threshold，使得 ind 和 ood 相关指标都比较好）
        threshold_score = self.get_threshold_score(self.tuning)
        test_preds[test_scores < threshold_score.view(-1, 1)] = self.num_labels_IND

        # test_labels: shape: [N]; torch.int64  ground truth 的 labels
        # test_pred: shape: [layers_nums, N]; 根据每层的输出结果，结合正常程度之类的打分，预测的 ind 和 ood 的 labels
        labels, preds = test_labels, test_preds

        # 每层的 threshold 处理没有干扰，所以 early exit 时候，我们可以当作后面的层，对应的模型没有继续跑过，同时 threshold 这些也没跑过
        esm_pred, esm_speedup = self.get_ensemble_pred(preds)
        pabee_pred, pabee_speedup = self.get_pabee_pred(preds)
        random_pred, random_speedup = self.get_random_pred(preds)

        #######################################################################################################
        # 写入同一个文件
        print(f'{self.METHOD}: ')
        for layer_index in range(len(preds)):
            results = {'strategy': 'each_layer', 'layer': layer_index+1, 'speedup': 1.0, 'tuning': self.tuning, 'scale': f'{self.scale_ind}+{self.scale_ood}', **F_measure(preds[layer_index], labels)}
            self._write(results)
        results = {'strategy': 'esm', 'layer': -1, 'speedup': esm_speedup, 'tuning': self.tuning, 'scale': f'{self.scale_ind}+{self.scale_ood}', **F_measure(esm_pred, labels)}
        self._write(results)
        results = {'strategy': 'pabee', 'layer': -1, 'speedup': pabee_speedup, 'tuning': self.tuning, 'scale': f'{self.scale_ind}+{self.scale_ood}', **F_measure(pabee_pred, labels)}
        self._write(results)
        results = {'strategy': 'random', 'layer': -1, 'speedup': random_speedup, 'tuning': self.tuning, 'scale': f'{self.scale_ind}+{self.scale_ood}', **F_measure(random_pred, labels)}
        self._write(results)

        print('#' * 80)

    def auc(self):
        """

        注意: 有的还是需要 logit 来得到正常度评分（例如 MspEvaluator）
        :return:
        """
        # shape: [layers_nums, N,]
        # score[layer_index, :] 表示第 layer_index 层的那个正常度打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常

        # shape: [N] torch.int64  ground truth 的 labels（包含 ood）
        test_scores, _, test_labels = self.predict_ind_labels(self.test_dataloader)

        print(f'{self.METHOD}: ')
        for layer_index in range(len(test_scores)):
            results = {'METRIC': self.METHOD, 'tuning': self.tuning, 'layer': layer_index + 1, **utils.au_sklearn(
                self.num_labels_IND, y_true=utils.tensor2numpy(test_labels), y_prob=utils.tensor2numpy(test_scores[layer_index])
            )}
            self._write(results)

        print('#' * 80)


# 正常度打分用句向量
class KnnEvaluator(Evaluator):
    METHOD = 'knn'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn_neighs = self._cal_knn_basic()

    def _cal_knn_basic(self) -> List[NearestNeighbors]:
        train_feats, _, _ = self.model_forward_with_cache(self.train_dataloader)

        def cal_by_layer(feats) -> NearestNeighbors:
            feats = utils.tensor2numpy(feats)
            neigh = NearestNeighbors(n_neighbors=5, n_jobs=4)
            neigh.fit(feats)
            return neigh

        return [cal_by_layer(feats) for feats in train_feats]

    def _get_score(self, total_feats: torch.Tensor) -> torch.Tensor:
        """
        正常度打分
        https://www.cnblogs.com/pinard/p/6065607.html
        对任一 feat，求其最近 k 邻向量，取较远那个距离的负数，作为正常度打分（距离越远，正常度变小，越可能异常）

        :param total_feats: shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        :return:
        """
        total_score = []
        for layer_index in range(len(total_feats)):
            dist, _ = self.knn_neighs[layer_index].kneighbors(tensor2numpy(total_feats[layer_index]), n_neighbors=5)
            total_score.append(dist[:, -1])
        # https://zhuanlan.zhihu.com/p/429901066
        return -torch.tensor(np.array(total_score)).to(self.device)

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        feats, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)
        scores = self._get_score(feats)
        _, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


# 正常度打分用 logits（或者归一化后概率）
class MspEvaluator(Evaluator):
    """
    基于预测分数（概率），如果针对某数据，在其所有 labels 打分中，+，仍然很小（极端情况下，1/num_labels_IND，即均匀分布），则可能是异常点
    """
    METHOD = 'msp'

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return scores, pred, ground_truth_labels


# 正常度打分用 logits（或者归一化后概率）
class MaxLogitEvaluator(Evaluator):
    """
    https://github.com/hendrycks/anomaly-seg/blob/8f78ffd6d7560fc6dfe0f5aed76bfcd239092a9b/eval_ood.py#L110
    MspEvaluator 基于预测分数（概率），如果针对某数据，在其所有 labels 打分中，+，仍然很小（极端情况下，1/num_labels_IND，即均匀分布），则可能是异常点
    这里使用的是，logits 打分（未归一化的概率），作为正常度打分
    """
    METHOD = 'maxLogit'

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


# 正常度打分用 logits（或者归一化后概率）
class EnergyEvaluator(Evaluator):
    """
    https://zhuanlan.zhihu.com/p/343678039
    https://github.com/wetliu/energy_ood/blob/77f3c09b788bb5a7bfde6fd3671228320ea0949c/CIFAR/test.py#L134

    MspEvaluator 基于预测分数（概率），如果针对某数据，在其所有 labels 打分中，+，仍然很小（极端情况下，1/num_labels_IND，即均匀分布），则可能是异常点
    这里使用的是，xxx，作为正常度打分
    """
    METHOD = 'energy'

    def __init__(self, temperature: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        _, pred = torch.max(logits, dim=-1)

        # T == 1 时候，等价于 torch.log(torch.sum(torch.exp(logits), dim=-1))
        scores = torch.mul(torch.logsumexp(torch.div(logits, self.T), dim=-1), self.T)
        return scores, pred, ground_truth_labels


# 正常度打分用 logits（或者归一化后概率）
class EntropyEvaluator(Evaluator):
    """
    熵越大，越混乱（最大时候，概率平均分布），则可能是异常点
    （-熵）表示正常度
    """
    METHOD = 'entropy'

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        _, pred = torch.max(logits, dim=-1)

        probs = F.softmax(logits, dim=-1)
        scores = -torch.sum((-probs*torch.log(probs)), dim=-1)

        return scores, pred, ground_truth_labels


# 正常度打分用 logits（或者归一化后概率）
class OdinEvaluator(Evaluator):
    """
    https://openreview.net/pdf?id=H1VGkIxRZ
    不知道原作者写了个jb，看论文觉得应该是这样（Out-of-distribution Detector 出来个 p 是啥啊，mdzz）
    logits 上面先加一个 temperature，然后看 probability
    """
    METHOD = 'odin'

    def __init__(self, temperature: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(F.softmax(torch.div(logits, self.T), dim=-1), dim=-1)
        return scores, pred, ground_truth_labels


# 正常度打分用句向量
class MahaEvaluator(Evaluator):
    """
    马氏距离
    https://zhuanlan.zhihu.com/p/46626607
    https://zhuanlan.zhihu.com/p/109100222

    """
    METHOD = 'maha'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids, self.linv = self._cal_maha_basic()

    def _cal_maha_basic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :return:
            result_centroids shape: [layers_nums, num_labels-1, hidden_size=768] 把相同的 labels 的数据，聚集到一起，求他们的中心（features 相加取平均）
            result_linv: shape: [layers_nums, hidden_size=768, hidden_size=768] linv.T @ linv = sigma^(-1)
        """

        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [N] torch.int64  ground truth 的 labels
        train_feats, _, train_labels = self.model_forward_with_cache(self.train_dataloader)

        def cal_per_layer(feats: torch.Tensor, labels: torch.Tensor):
            """

            https://zhuanlan.zhihu.com/p/46626607
            https://zhuanlan.zhihu.com/p/109100222

            :param feats: [N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
            :param labels: shape: [N] torch.int64  ground truth 的 labels
            :return:
                centroids shape: [num_labels-1, hidden_size=768] 把相同的 labels 的数据，聚集到一起，求他们的中心（features 相加取平均）
                linv: shape: [hidden_size=768, hidden_size=768] linv.T @ linv = sigma^(-1)

            """

            # 把相同的 labels 的数据，聚集到一起，求他们的中心（features 相加取平均）
            centroids = torch.zeros(self.num_labels_IND, 768).to(self.device)
            class_count = [0] * self.num_labels_IND
            for i, feature in enumerate(feats):
                label = labels[i]
                centroids[label] += feature
                class_count[label] += 1
            centroids /= torch.tensor(class_count).float().unsqueeze(1).to(self.device)

            # shape: [N, hidden_size=768] mu[i, :] 表示 x_i 对应 label，其“聚类”中心向量
            mu = centroids[labels]
            # X - μ_X
            x_mu = feats - mu
            # shape: [hidden_size=768, hidden_size=768]
            sigma = torch.matmul(x_mu.T, x_mu)

            # shape: [hidden_size=768, hidden_size=768]
            # 数学性质：linv.T @ linv = sigma^(-1)
            linv = torch.linalg.inv(torch.linalg.cholesky(sigma))
            return centroids, linv

        result_centroids, result_linv = [], []
        for layer_index in range(len(train_feats)):
            cur_layer_centroids, cur_layer_linv = cal_per_layer(train_feats[layer_index], train_labels)
            result_centroids.append(cur_layer_centroids)
            result_linv.append(cur_layer_linv)
        return torch.stack(result_centroids), torch.stack(result_linv)

    def _get_score_pred(self, total_feats) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param total_feats: shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        :return:
            shape: [layers_nums, N] total_pred[layer_index, i] 表示 x_i 和 哪个 label 最近
            shape: [layers_nums, N] total_score[layer_index, i] 表示 x_i 和 哪个 label 最近，对应的距离（取负），则马氏距离越大（即 total_score 值越小），越可能是 OOD；即这也是个正常度打分

        """
        def get_score_by_layer(layer_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """

            :param layer_index:
            :return:
                values, indices
            """

            # cur_layer_centroids shape: [num_labels-1, hidden_size=768] 把相同的 labels 的数据，聚集到一起，求他们的中心（features 相加取平均）
            # cur_layer_linv: shape: [hidden_size=768, hidden_size=768] linv.T @ linv = sigma^(-1)
            cur_layer_centroids = self.centroids[layer_index]
            cur_layer_linv = self.linv[layer_index]

            # [N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
            cur_layer_feats = total_feats[layer_index]

            n = cur_layer_feats.shape[0]
            num_labels_ind = cur_layer_centroids.shape[0]

            # 某一层的马氏距离（看 cur_layer_feats_i 距离哪个 label 对应的中心点最近，那就是该 label）:
            #   D^2 = (cur_layer_feats_i - self.centroids[layer_index, ind_label_index]).T @ sigma^(-1) @ (cur_layer_feats_i - self.centroids[layer_index, ind_label_index])
            #       = (cur_layer_feats_i - self.centroids[layer_index, ind_label_index]).T @ {self.linv[layer_index].T @ self.linv[layer_index]}  @
            #               (cur_layer_feats_i - self.centroids[layer_index, ind_label_index])

            # shape: [N, num_labels_ind, hidden_size=768]
            # x[i, ind_label_index] = cur_layer_feats[i] - cur_layer_centroids[ind_label_index]
            x = cur_layer_feats.unsqueeze(1).expand(n, num_labels_ind, -1) - cur_layer_centroids.unsqueeze(0).expand(n, num_labels_ind, -1)
            # shape: [N, num_labels_ind, hidden_size=768]
            # x[i, ind_label_index] = (cur_layer_feats[i] - self.centroids[layer_index, ind_label_index]) @ self.linv[layer_index]
            x = torch.matmul(x, cur_layer_linv)
            # shape: [N, num_labels_ind]
            # d[i, ind_label_index] 表示 x_i 和每个 label 中心的距离（取负了）
            d = -torch.sum(x ** 2, dim=-1)

            values, indices = torch.max(d, dim=-1)
            # shape: [N,] indices[i] 表示 x_i 和 哪个 label 最近
            # shape: [N,] values[i] 表示 x_i 和 哪个 label 最近，对应的距离（取负）
            return values, indices

        print('ST')
        total_score, total_pred = [], []
        for layer_index_ in range(len(total_feats)):
            cur_layer_score, cur_layer_pred = get_score_by_layer(layer_index_)
            total_score.append(cur_layer_score)
            total_pred.append(cur_layer_pred)
        return torch.stack(total_score), torch.stack(total_pred)

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [N] torch.int64  ground truth 的 labels
        feats, _, ground_truth_labels = self.model_forward_with_cache(dataloader)
        scores, pred = self._get_score_pred(feats)
        return scores, pred, ground_truth_labels


# 正常度打分用句向量
class _LofEvaluator(Evaluator):
    METHOD = 'lof'

    def __init__(self, distance_metric: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.lof_adapters[i] 一共有 layers_nums 层，每层有各自的 LOF 打分模型
        self.lof_adapters = self._cal_lof_basic(distance_metric)

    def _cal_lof_basic(self, metric: str) -> List[LocalOutlierFactor]:
        train_feats, _, _ = self.model_forward_with_cache(self.train_dataloader)

        def cal_by_layer(feats: torch.Tensor) -> LocalOutlierFactor:
            """

            :param feats: [N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
            :return:
            """
            lof = LocalOutlierFactor(n_neighbors=20, metric=metric, novelty=True, n_jobs=4)
            lof.fit(tensor2numpy(feats))
            return lof

        return [cal_by_layer(feats) for feats in train_feats]

    def _get_score(self, total_feats: torch.Tensor) -> torch.Tensor:
        """
        正常度打分

        :param total_feats: shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量
        :return: shape: [layers_nums, N,]
            result[layer_index, :] 表示第 layer_index 层的那个 LOF 打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常
            It is the opposite as bigger is better, i.e. large values correspond to inliers.
        """
        # List[torch.Tensor] 每个元素都是 shape [N, hidden_size=768]，一共 layers_nums 个元素
        total_score = [torch.tensor(self.lof_adapters[layer_index].score_samples(tensor2numpy(total_feats[layer_index])))
                       for layer_index in range(len(total_feats))]
        return torch.stack(total_score).to(self.device)

    def predict_ind_labels(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [layers_nums, N, hidden_size=768]  torch.float32 bert 模型某一层得到 hidden states 中的对应句向量（可能有二次重新映射）
        # shape: [layers_nums, N, num_labels-1]  torch.float32 表示每个 class 的“概率”（未归一化）；num_labels 包含了 oos
        # shape: [N] torch.int64  ground truth 的 labels
        feats, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        # shape: [layers_nums, N,]
        # score[layer_index, :] 表示第 layer_index 层的那个 LOF 打分模型，给每个 x_i 的对应生成特征打的分数，越小表示越可能异常
        scores = self._get_score(feats)

        # 利用 argmax probability/logits 得到 predict_ind_labels，这里先假设都是 IND 数据
        # shape: [layers_nums, N]
        _, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


# 正常度打分用句向量
class LofCosineEvaluator(_LofEvaluator):
    METHOD = 'lof_cosine'

    def __init__(self, *args, **kwargs):
        super().__init__(distance_metric='cosine', *args, **kwargs)


# 正常度打分用句向量
class LofEuclideanEvaluator(_LofEvaluator):
    METHOD = 'lof_euclidean'

    def __init__(self, *args, **kwargs):
        super().__init__(distance_metric='euclidean', *args, **kwargs)
