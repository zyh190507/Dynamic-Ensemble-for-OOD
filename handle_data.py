from typing import List

import torch
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict
from datasets.arrow_dataset import Batch

from my_args import DataTrainingArguments


def data_collator(features):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    first = features[0]
    batch = {}
    if "original_text" in first:
        batch["original_text"] = [f["original_text"] for f in features]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def convert_to_nums(data_args: DataTrainingArguments, datasets: DatasetDict, label_list: List[int], tokenizer: transformers.PreTrainedTokenizerBase) -> DatasetDict:
    # 查看 input 有什么特征（这里只有一个 text，作为输入特征；bert 是输入两句话的，另一句就是 None 了）
    # Preprocessing the datasets
    # 这里是本项目自定义的

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # 校验并获得 label_to_id
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    label_to_id = {v: i for i, v in enumerate(label_list)}

    #####################################################################################
    # 数据编码（把 label text 转为 int 这些，方便后续转为 Tensor）
    def preprocess_function(examples: Batch):
        """
        字段有 input_ids；token_type_ids；attention_mask;label; sent_id; original_text

        input_ids；token_type_ids 就是 bert 里面是第几句话；attention_mask 标记每句话真实长度

        labels: List[int]
        sent_id: List[int]
        original_text: List[str]
        """
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        result["sent_id"] = [index for index, i in enumerate(examples["label"])]
        result["original_text"] = examples[sentence1_key]
        return result

    # # map: Apply a function to all the elements in the table
    datasets = datasets.map(preprocess_function, batched=True, batch_size=None, load_from_cache_file=not data_args.overwrite_cache)
    return datasets


def load_datasets(data_args: DataTrainingArguments) -> DatasetDict:
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    df_train = pd.read_csv(data_args.train_file, sep='\t', dtype=str)
    df_valid = pd.read_csv(data_args.valid_file, sep='\t', dtype=str)
    df_test = pd.read_csv(data_args.test_file, sep='\t', dtype=str)

    if data_args.data not in ['clinc_full', 'clinc_small']:
        # 从 test 集合里面的所有 label 中，挑选 known_ratio 比例的 labels 作为 ID，其余的 OOD
        unique_labels = df_train.label.unique()
        seen_labels = np.random.choice(unique_labels, int(len(unique_labels) * data_args.known_ratio), replace=False)

        # df_train[..] 是在 pandas.DataFrame 做了一个 copy 操作，无论对 df_valid_oos、df_train_seen、df_train_seen 这些进行什么修改，都不会影响到原来的 DataFrame
        df_train_seen: pd.DataFrame = df_train[df_train.label.isin(seen_labels)]
        df_valid_seen: pd.DataFrame = df_valid[df_valid.label.isin(seen_labels)]
        df_valid_oos: pd.DataFrame = df_valid[~df_valid.label.isin(seen_labels)]

        df_valid_oos.loc[:, "label"] = 'oos'

        # df_test 可能本身也有 oos 数据，不过 oos label 肯定不在 unique_labels（训练集不可见 ood 数据）
        # 所以这里的处理实际是等价于处理后（程序中运行的）的 ood 包含了 原来数据集里面的 oos label + unseen_labels

        # 这里即使 df_test 含有 oos 也没啥问题，因为 seen_labels 和 unique_labels 一定没有 oos
        df_test.loc[~df_test.label.isin(seen_labels), "label"] = 'oos'
    else:
        # data_args.data == 'clinc_full' 此时自带 ood label，不用额外根据 data_args.known_ratio 参数 mask label 了
        df_train_seen = df_train
        df_valid_seen = df_valid
        df_valid_oos = pd.read_csv(f'./data/{data_args.data}/valid_oos.tsv', sep='\t', dtype=str)

    df_valid_all = pd.concat([df_valid_seen, df_valid_oos])

    data = {
        "train": Dataset.from_pandas(df_train_seen, preserve_index=False),
        "valid_seen": Dataset.from_pandas(df_valid_seen, preserve_index=False),
        "valid_oos": Dataset.from_pandas(df_valid_oos, preserve_index=False),
        "valid_all": Dataset.from_pandas(df_valid_all, preserve_index=False),
        "test": Dataset.from_pandas(df_test, preserve_index=False),
    }

    # Dict[str, Dataset]
    datasets = DatasetDict(data)
    return datasets
