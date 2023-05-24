from typing import Optional

import math
import copy

import fitlog
import numpy as np
import torch
from sklearn import metrics

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from transformers.trainer import Trainer

import utils


class SimpleTrainer(Trainer):
    def __init__(self, supcont_pre_epoches: int, clip: float, train_step, model_path_: Optional[str] = None, **kwargs):
        self.supcont_pre_epoches = supcont_pre_epoches
        self.clip = clip
        self.model_path_ = model_path_
        self.train_step = train_step
        super().__init__(**kwargs)

    def evaluation_cal(self, model, val_dataloader) -> float:
        # shape: [N,] ground truth（此时不可见 ood 数据）
        total_labels = None
        # shape: [layers_nums, N] 预测的 ind 的 labels
        total_pred = None

        model.eval()

        with torch.no_grad():
            for step, inputs in enumerate(val_dataloader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                # shape: [N,]
                labels = inputs['labels']
                # shape: [layers_nums, N, num_ind_labels]
                _, logits = model(inputs, mode='eval')

                # shape: [layers_nums, N] 预测的 ind 的 labels
                _, pred = torch.max(logits, dim=-1)

                if total_pred is None:
                    total_labels = labels
                    total_pred = pred
                else:
                    total_labels = torch.cat((total_labels, labels))
                    total_pred = torch.cat((total_pred, pred), dim=1)

        y_pred = utils.tensor2numpy(total_pred)
        y_true = utils.tensor2numpy(total_labels)
        # f1 = [metrics.f1_score(y_true, y_pred[i], average='macro') for i in range(12)]
        # return f1[-1]

        accs = [metrics.accuracy_score(y_true, y_pred[i]) for i in range(12)]
        cur_acc = np.mean(accs)
        return cur_acc.item()

    def train_ce_loss(self):
        model = self.model
        model.to(self.args.device)

        ################################################################################################
        train_dataloader = self.get_train_dataloader()
        valid_loader = self.get_eval_dataloader()
        # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        max_steps = math.ceil(self.supcont_pre_epoches * num_update_steps_per_epoch)
        self.args.warmup_steps = max_steps * 0.1
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        best_f1 = 0
        model_best_param_dict = None

        sup_con_num_train_epochs = self.supcont_pre_epoches
        wait = 5

        for epoch in range(sup_con_num_train_epochs):

            with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    " + ",
                    TimeRemainingColumn(),
            ) as progress:
                cur_epoch_sum_loss = 0
                cur_epoch_step_nums = 0

                epoch_tqdm = progress.add_task(description="epoch progress", total=len(train_dataloader))
                model.train()
                for step, inputs in enumerate(train_dataloader):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.args.device)
                    self.optimizer.zero_grad()

                    # 求梯度
                    loss = self.train_step.step(model, inputs)

                    # log
                    progress.update(epoch_tqdm, advance=1, description='Iter (ce_loss=%5.3f)' % loss)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    cur_epoch_sum_loss += loss
                    cur_epoch_step_nums += 1

                cl_loss_name = 'ce_loss'
                cur_epoch_loss = cur_epoch_sum_loss / cur_epoch_step_nums
                fitlog.add_loss(cur_epoch_loss, name=cl_loss_name, step=epoch)
                # Get model performing in valid IND
                f1 = self.evaluation_cal(model, valid_loader)

                progress.update(epoch_tqdm, description=f'Epoch: [{epoch+1:03d}]: Loss {cur_epoch_loss:.4f} | F1 {f1:.4f}')

                if f1 > best_f1:
                    # torch.save(model, model_file_path)
                    model_best_param_dict = copy.deepcopy(model.state_dict())
                    best_f1 = f1
                    wait = 0
                else:
                    wait += 1
                    if wait >= 5:
                        break

        # https://pytorch123.com/ThirdSection/SaveModel/
        torch.save(model_best_param_dict, self.model_path_)

        model.load_state_dict(model_best_param_dict)
