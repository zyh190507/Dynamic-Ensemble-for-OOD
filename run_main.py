import os
import logging
import shutil
import sys

import torch
import fitlog

import transformers
import transformers.utils.logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed, BertConfig,
)
from transformers.trainer_utils import is_main_process

import eval as my_eval
from handle_data import load_datasets, convert_to_nums, data_collator
from my_args import DataTrainingArguments, FitLogArguments, OtherArguments
from my_trainer import SimpleTrainer
from transformers.training_args import TrainingArguments
from models import (
    BertForSequenceClassificationWithPabee
)
from my_hf_argparser import HfArgumentParser
import train_step_freelb, train_step_plain

logger = logging.getLogger(__name__)
torch.set_num_threads(6)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script_v0.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # https://zhuanlan.zhihu.com/p/296535876
    # HfArgumentParser可以将类对象中的实例属性转换成转换为解析参数。
    parser = HfArgumentParser((OtherArguments, DataTrainingArguments, TrainingArguments, FitLogArguments))
    other_args: OtherArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    fitlog_args: FitLogArguments

    assert len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    other_args, data_args, training_args, fitlog_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    # # TODO: DEBUG
    # training_args.do_train = False
    # other_args.scale = 1.5
    # other_args.scale_ood = 1.3

    # 校验防呆
    assert other_args.loss_type in ["original", "increase", "ce_and_div_drop-last-layer", "ce_and_div"]

    # trainer 执行 DataLoader 转化，有一步 _remove_unused_columns
    # 什么 sb 默认啊
    training_args.remove_unused_columns = False

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 设置文件、模型、metrics 保存（能追溯）
    if training_args.do_train:
        adv_args = f'{other_args.adv_k}{other_args.adv_lr}{other_args.adv_init_mag}'

        if other_args.scale == other_args.scale_ood:
            model_output_root = f'./model_output/{data_args.data}_{data_args.known_ratio}/ad{adv_args}_lr{training_args.learning_rate:.1e}__epoch{other_args.supcont_pre_epoches}__loss{other_args.loss_type}' \
                                f'__batchsize{training_args.per_device_train_batch_size}__lambda{other_args.diversity_loss_weight}__scale{other_args.scale}/'
        else:
            model_output_root = f'./model_output/{data_args.data}_{data_args.known_ratio}/ad{adv_args}_lr{training_args.learning_rate:.1e}__epoch{other_args.supcont_pre_epoches}__loss{other_args.loss_type}' \
                                f'__batchsize{training_args.per_device_train_batch_size}__lambda{other_args.diversity_loss_weight}__scale{other_args.scale}{other_args.scale_ood}/'
        if not os.path.exists(model_output_root):
            os.makedirs(model_output_root)
    else:
        # 所以这里要求的是 pt 文件，和 json 配置文件必须同一个文件夹里面
        # dirname 去掉文件名，返回目录
        model_output_root = os.path.dirname(os.path.abspath(sys.argv[1]))

    if training_args.do_train:
        json_file_name = '_' + '_'.join([data_args.data, str(data_args.known_ratio), str(training_args.seed)]) + '.json'
        shutil.copy2(os.path.abspath(sys.argv[1]), os.path.join(model_output_root, json_file_name))

    ################################################################################
    # 设置 fitlog
    fitlog.set_log_dir(other_args.fitlog_dir)
    fitlog_args_dict = {
        "seed": training_args.seed,
        "warmup_steps": training_args.warmup_steps,
        "task_name": f'{data_args.data}-{data_args.known_ratio}-{training_args.seed}'

    }
    fitlog_args_name = [i for i in dir(fitlog_args) if i[0] != "_"]
    for args_name in fitlog_args_name:
        args_value = getattr(fitlog_args, args_name)
        if args_value is not None:
            fitlog_args_dict[args_name] = args_value
    fitlog.add_hyper(fitlog_args_dict)

    ################################################################################

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    ################################################################################################
    datasets = load_datasets(data_args)

    # 有多少 label

    # 获取 num_all_labels（其中加上了 oos）
    # 这里是本项目自定义的
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    assert not is_regression
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"].unique("label")
    label_list += ['oos']
    num_all_labels = len(label_list)

    tokenizer: transformers.PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        other_args.model_name_or_path,
        cache_dir=other_args.cache_dir,
        # local_files_only=True,
        use_fast=True,
    )

    datasets = convert_to_nums(data_args, datasets, label_list, tokenizer)

    ################################################################################################
    # 得到 model（这里暂不支持加载已经训练好的）

    pertained_config: BertConfig = AutoConfig.from_pretrained(
        other_args.model_name_or_path,
        finetuning_task=None,
        # local_files_only=True,
        cache_dir=other_args.cache_dir,
    )

    # model = pjmodel.BertForSequenceClassificationWithPabee.from_pretrained(other_args.model_name_or_path, num_ind_labels=num_all_labels-1)
    model = BertForSequenceClassificationWithPabee(pertained_config=pertained_config, other_args=other_args, num_ind_labels=num_all_labels-1)

    #####################################################################################
    # 训练

    model_file_name = '_' + '_'.join([data_args.data, str(data_args.known_ratio), str(training_args.seed)]) + '.pt'

    if other_args.adv_k > 0:
        train_step = train_step_freelb.FreeLB(
            adv_k=other_args.adv_k,
            adv_lr=other_args.adv_lr,
            adv_init_mag=other_args.adv_init_mag,
            adv_max_norm=other_args.adv_max_norm
        )
    else:
        train_step = train_step_plain.TrainStep()
    trainer = SimpleTrainer(
        supcont_pre_epoches=other_args.supcont_pre_epoches,
        clip=other_args.clip,
        model_path_=os.path.join(model_output_root, model_file_name),  # 保存模型
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid_seen"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_step=train_step
    )
    if training_args.do_train:
        trainer.train_ce_loss()
    else:
        model.load_state_dict(torch.load(os.path.join(model_output_root, model_file_name)))
        model.to(training_args.device)

    ################################################################################################
    # Evaluation
    if training_args.do_predict:
        file_postfix = '_'.join([data_args.data, str(data_args.known_ratio), str(training_args.seed)]) + '.csv'

        # # TODO: remove
        # model_output_root = f'./model_output/tmp/{other_args.scale}_{other_args.scale_ood}'
        # if not os.path.exists(model_output_root):
        #     os.makedirs(model_output_root)

        valid_all_dataloader = trainer.get_eval_dataloader(datasets['valid_all'])
        valid_dataloader = trainer.get_eval_dataloader(datasets['valid_seen'])
        train_dataloader = trainer.get_train_dataloader()
        test_dataloader = trainer.get_test_dataloader(datasets["test"])

        model_forward_cache = {}

        kwargs = dict(
            model=model,
            root=model_output_root,
            file_postfix=file_postfix,
            dataset_name=data_args.data,
            device=training_args.device,
            num_labels=num_all_labels,
            tuning='valid_all',
            scale_ind=other_args.scale,
            scale_ood=other_args.scale_ood,
            valid_all_dataloader=valid_all_dataloader,
            valid_dataloader=valid_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model_forward_cache=model_forward_cache
        )

        evaluator = my_eval.KnnEvaluator(**kwargs)
        evaluator.eval()
        # # evaluator = my_eval.MspEvaluator(**kwargs)
        # # evaluator.eval()
        # # evaluator = my_eval.MaxLogitEvaluator(**kwargs)
        # # evaluator.eval()
        # # evaluator = my_eval.EnergyEvaluator(temperature=1, **kwargs)
        # # evaluator.eval()
        # # evaluator = my_eval.EntropyEvaluator(**kwargs)
        # # evaluator.eval()
        # # evaluator = my_eval.OdinEvaluator(temperature=100, **kwargs)
        # # evaluator.eval()
        # # evaluator = my_eval.MahaEvaluator(**kwargs)
        # # evaluator.eval()
        evaluator = my_eval.LofCosineEvaluator(**kwargs)
        evaluator.eval()
        # evaluator = my_eval.LofEuclideanEvaluator(**kwargs)
        # evaluator.eval()

    return None


if __name__ == "__main__":
    main()
