"""
FreeLB 实现
参考 https://blog.csdn.net/weixin_42001089/article/details/115458615
https://github.com/zhuchen03/FreeLB/blob/d8c169519208acebe3791febbb5b2724f04dae83/fairseq-RoBERTa/examples/roberta/wsc/wsc_task.py#L83
"""
from typing import Dict, Any

import torch
from torch import nn


class FreeLB:
    def __init__(self, adv_k: int, adv_lr: float, adv_init_mag: float, adv_max_norm: float, base_model: str = 'bert', adv_norm_type: str = 'l2'):
        self.adv_k = adv_k  # 论文 K；好像 m 也是?
        self.adv_lr = adv_lr  # 论文 α
        self.adv_max_norm = adv_max_norm  # 论文 ε
        self.adv_init_mag = adv_init_mag  # 论文 ε
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
        assert self.adv_norm_type == "l2"

        self.random_states = {}

    @staticmethod
    def _model_forward(model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        返回 loss
        :param inputs:
        :return:
        """
        loss = model(inputs, mode='train')
        return loss

    def _save_random_state(self) -> None:
        self.random_states = {
            'cuda_cur_state': torch.cuda.get_rng_state(),
            'cur_state': torch.get_rng_state()
        }

    def _load_random_state(self) -> None:
        torch.cuda.set_rng_state(self.random_states['cuda_cur_state'])
        torch.set_rng_state(self.random_states['cur_state'])

    def step(self, model: nn.Module, inputs: Dict[str, Any]) -> float:
        """
        执行完一个 batch 的梯度操作，注意没有 optimizer step
        要求模型和 inputs 全部在同一个 device
        :param model:
        :param inputs:
        :return:
        """
        ################################################################################################################################################
        # 原来的学习方法
        self._save_random_state()

        loss = self._model_forward(model, inputs)
        # https://github.com/zhuchen03/FreeLB/blob/d8c169519208acebe3791febbb5b2724f04dae83/fairseq-RoBERTa/examples/roberta/wsc/wsc_task.py#L150
        # 直观理解就是这次实际是 K+1 次梯度叠加了，权重就除以 (K+1)
        loss: torch.Tensor = torch.div(loss, (1 + self.adv_k))
        loss.backward()

        total_loss = loss.item()

        ################################################################################################################################################
        # adversarial train

        input_ids = inputs['input_ids']  # shape: [N, sequence_length]

        tmp_embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids).clone().detach()  # shape: [N, sequence_length, hidden_size]

        # δ_0 初始化
        if self.adv_init_mag > 0:
            input_mask: torch.Tensor = inputs['attention_mask']  # shape: [N, sequence_length]  表明每个句子的实际长度
            input_lengths = torch.sum(input_mask, 1)  # shape: [N]

            # shape: [N, sequence_length, hidden_size]; 对应论文 algorithm 1 里面 第 4 行 U(−ε, ε) （后面 adv_init_mag === ε）
            delta = torch.zeros_like(tmp_embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)

            # mag shape: [N, sequence_length]；
            # mag[i,:] 对应了 sentences[i]，其中“真实” token 地方对应的都是 sequence_length，padding 地方都是 0
            # 对应论文 algorithm 1 里面 第 4 行  1/sqrt(N)
            dims = input_lengths * tmp_embeds_init.size(-1)
            mag = self.adv_init_mag / torch.sqrt(dims)

            delta = (delta * mag.view(-1, 1, 1))
        else:
            delta = torch.zeros_like(tmp_embeds_init)

        for i in range(self.adv_k):
            ###############################################################################
            # 隐式执行论文 algorithm 1 里面 第 8 行

            delta.requires_grad_()  # delta 扰动参与 loss 计算，其梯度方向，表示了 loss 增长最快（攻击最切中要害）的方向

            # word_embeddings 自始至终没有改变过（没有执行 optimizer.step），就是正常累加梯度（论文 algorithm 1 里面 第 8 行）
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
            # inputs_embeds 不是 None，则不走 word_embeddings(input_ids) ，而是直接使用 inputs_embeds 作为后续 transformer 层的输入
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None

            self._load_random_state()

            loss = self._model_forward(model, inputs)
            # https://github.com/zhuchen03/FreeLB/blob/d8c169519208acebe3791febbb5b2724f04dae83/fairseq-RoBERTa/examples/roberta/wsc/wsc_task.py#L203
            # 直观理解就是这次实际是 K+1 次梯度叠加了，权重就除以 (K+1)
            loss: torch.Tensor = torch.div(loss, (1 + self.adv_k))
            loss.backward()  # 隐式执行论文 algorithm 1 里面 第 8 行，这里论文自己官方实现也没有 1/K 操作

            total_loss += loss.item()

            #################################################################################
            # if i == self.adv_k - 1:
            #     break

            # 对应论文 algorithm 1 里面 第 11 行
            # detach: The result will never require gradient.
            # 即这里处理后 delta_grad 不走反向梯度传播
            delta_grad = delta.grad.clone().detach()
            delta = delta.clone().detach()

            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)

            # 这里处理中 delta 不走反向梯度传播
            delta = delta + self.adv_lr * delta_grad / denorm

            # 对应论文 algorithm 1 里面 第 11 行  Π 是个约束函数（不是累乘）
            # 避免 delta_t 过大
            if self.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1)
                exceed_mask = torch.gt(delta_norm, self.adv_max_norm).to(embeds_init)  # (delta_norm > self.adv_max_norm).to(embeds_init)
                reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights)

        return total_loss