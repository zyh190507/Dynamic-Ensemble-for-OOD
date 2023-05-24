from typing import Dict, Any

import torch
from torch import nn


class TrainStep:
    @staticmethod
    def _model_forward(model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        返回 loss
        :param inputs:
        :return:
        """
        loss = model(inputs, mode='train')
        return loss

    def step(self, model: nn.Module, inputs: Dict[str, Any]) -> float:
        """
        执行完一个 batch 的梯度操作，注意没有 optimizer step
        要求模型和 inputs 全部在同一个 device
        :param model:
        :param inputs:
        :return:
        """

        loss = self._model_forward(model, inputs)
        loss.backward()

        total_loss = loss.item()
        return total_loss