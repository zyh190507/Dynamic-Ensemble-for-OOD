from typing import Tuple, List

from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from my_args import OtherArguments


class Classifier(nn.Module):
    """
    最早一版（类 pabee）区别是，pooler 部分（把 [cls] 对应的 hidden states 重新映射，映射矩阵是 shape [hidden_size, hidden_size]）是每层分类器共享的
    现在不共享了
    且参考 https://github.com/HanleiZhang/Adaptive-Decision-Boundary/blob/main/model.py ，sentence embedding 也修改
    """
    def __init__(self, config: BertConfig, num_ind_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_ind_labels)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:
        """

        :param hidden_states: shape: [N, sequence_length, hidden_size] bert 某一层的输出的结果
        :return:
            sentences_embeddings: shape: [N, hidden_size]；句子向量表示，用于 ood 打分
            logits：shape: [N, num_ind_labels]；未归一化的 labels 预测概率分布
        """
        # [cls] 对应的向量；shape: [N, hidden_size]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        # shape: [N, hidden_size]
        sentences_embeddings = self.dropout(pooled_output)

        # shape: [N, num_ind_labels]
        logits = self.classifier(sentences_embeddings)
        return sentences_embeddings, logits


class BertForSequenceClassificationWithPabee(nn.Module):
    def __init__(self, pertained_config: BertConfig, other_args: OtherArguments, num_ind_labels: int):
        """

        :param pertained_config:
        :param other_args:
        :param num_ind_labels:
        """
        super().__init__()
        self.num_ind_labels = num_ind_labels

        self.bert: BertModel = BertModel.from_pretrained(other_args.model_name_or_path, config=pertained_config)

        # 每层后面一个分类器
        self.classifiers = nn.ModuleList(
            [Classifier(pertained_config, num_ind_labels) for _ in range(pertained_config.num_hidden_layers)]
        )

        self.loss_type = other_args.loss_type
        self.diversity_loss_weight = other_args.diversity_loss_weight
        assert 1 >= self.diversity_loss_weight >= 0  # 防呆

    def forward_each_layer(self, query) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """

        :param query:
        :return:
            feats: bert 有几层，就有多少个元素；每个元素都是 shape: [N, hidden_size]，表示每层 bert 后，算得的句向量表示
            logits: bert 有几层，就有多少个元素；每个元素都是 shape: [N, num_ind_labels]，表示每层 bert 后，利用算得的句向量表示，训练的分类器，他的每一类的情况
        """
        # 丢掉一部分多余的数据
        query = {key: value for key, value in query.items() if key not in ['labels', 'original_text', 'sent_id']}

        bert_output = self.bert(output_hidden_states=True, return_dict=True, **query)
        # 长度是 1+pertained_config.num_hidden_layers，即包括 embedding 层输出和每个 transformer 层的输出
        hidden_states = bert_output.hidden_states

        feats, logits = [], []

        # cur_layer_hidden_states: shape: [N, sequence_length, hidden_size]
        for index, cur_layer_hidden_states in enumerate(hidden_states[1:]):
            # cur_layer_pooled_output大体就是把 [CLS] token 对应的“句向量”过个全连接；shape: [N, hidden_size]
            # cur_layer_logit shape: [N, num_ind_labels]
            cur_layer_pooled_output, cur_layer_logit = self.classifiers[index](cur_layer_hidden_states)

            feats.append(cur_layer_pooled_output)
            logits.append(cur_layer_logit)

        return feats, logits

    def forward(self,
                query,
                mode
                ):
        labels = query["labels"]
        labels = labels.view(-1)

        # feats: bert 有几层，就有多少个元素；每个元素都是 shape: [N, hidden_size]，表示每层 bert 后，算得的句向量表示
        # logits: bert 有几层，就有多少个元素；每个元素都是 shape: [N, num_ind_labels]，表示每层 bert 后，利用算得的句向量表示，训练的分类器，他的每一类的情况（未归一化）
        feats, logits = self.forward_each_layer(query)

        if mode == 'eval':
            # shape: [layers_nums, N, hidden_size]
            # shape: [layers_nums, N, num_ind_labels]
            return torch.stack(feats, dim=0), torch.stack(logits, dim=0)

        if mode == 'train':
            if self.loss_type == 'original':
                losses = 0
                # cur_layer_logits shape: [N, num_ind_labels]
                for layer_index, cur_layer_logits in enumerate(logits):
                    # https://blog.csdn.net/goodxin_ie/article/details/89645358
                    # loss shape: [N,]
                    # loss_fct = CrossEntropyLoss(reduction='none')
                    # loss = loss_fct(cur_layer_logits.view(-1, self.num_ind_labels), labels)

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(cur_layer_logits.view(-1, self.num_ind_labels), labels)

                    losses += loss
                return losses
            if self.loss_type == 'increase':
                total_loss = 0
                total_weights = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(cur_layer_logits.view(-1, self.num_ind_labels), labels)

                    total_loss += loss * (layer_index + 1)
                    total_weights += layer_index + 1
                loss = total_loss / total_weights
                return loss

            # 只看最后一层的结果
            if self.loss_type == 'plain':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[-1].view(-1, self.num_ind_labels), labels)
                return loss

            if self.loss_type == 'ce_and_div':
                # 参考 https://arxiv.org/pdf/2105.13792.pdf 公式 14
                total_ce_loss = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(cur_layer_logits.view(-1, self.num_ind_labels), labels)
                    total_ce_loss += loss

                total_diversity_loss = 0
                scores = [torch.softmax(cur_layer_logits, dim=1) for cur_layer_logits in logits]  # 归一化（CrossEntropyLoss 要求的）
                for layer_index_i, cur_layer_logits in enumerate(logits[1:], start=1):  # 最下面那一层分类器不用算 diversity_loss
                    min_ce_loss = None
                    for layer_index_j in range(layer_index_i):
                        loss_fct = CrossEntropyLoss()
                        ce_loss_i_j = loss_fct(cur_layer_logits, scores[layer_index_j])
                        if min_ce_loss is None:
                            min_ce_loss = ce_loss_i_j
                        else:
                            min_ce_loss = min(min_ce_loss, ce_loss_i_j)
                    total_diversity_loss += min_ce_loss

                return total_ce_loss - self.diversity_loss_weight * total_diversity_loss

            if self.loss_type == 'ce_and_div_drop-last-layer':
                # 参考 https://arxiv.org/pdf/2105.13792.pdf 公式 14
                total_ce_loss = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(cur_layer_logits.view(-1, self.num_ind_labels), labels)
                    total_ce_loss += loss

                total_diversity_loss = 0
                scores = [torch.softmax(cur_layer_logits, dim=1) for cur_layer_logits in logits]  # 归一化（CrossEntropyLoss 要求的）
                # "Further, we find that better results can sometimes be obtained when removing the diversity term of the last internal classifier"
                for layer_index_i, cur_layer_logits in enumerate(logits[1: -1], start=1):  # 最下面那一层分类器不用算 diversity_loss
                    min_ce_loss = None
                    for layer_index_j in range(layer_index_i):
                        loss_fct = CrossEntropyLoss()
                        ce_loss_i_j = loss_fct(cur_layer_logits, scores[layer_index_j])
                        if min_ce_loss is None:
                            min_ce_loss = ce_loss_i_j
                        else:
                            min_ce_loss = min(min_ce_loss, ce_loss_i_j)
                    total_diversity_loss += min_ce_loss

                return total_ce_loss - self.diversity_loss_weight * total_diversity_loss


        raise ImportError(f"unknown mode {mode} or loss_type {self.loss_type}")
