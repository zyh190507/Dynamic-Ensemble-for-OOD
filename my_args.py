from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    """
    known_ratio data 可以从 json 里面设置，其余的均为自动生成或者写死
    """
    known_ratio: float = field(metadata={"help": "MASK (1-known_ratio) labels as OOD labels"})
    data: str = field(metadata={"help": "dataset name"})

    ##################################################################################
    # 自动生成
    # train_file valid_file test_file 依赖 data
    train_file: str = field(
        init=False, metadata={"help": "A csv or a json file containing the training data."}
    )
    valid_file: str = field(
        init=False, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: str = field(
        init=False, metadata={"help": "A csv or a json file containing the test data."}
    )

    ##################################################################################
    # 写死

    # default=xxx && init=False 禁止初始化时候修改（类似于 const）
    # 这样强制在 tokenize 预处理时候，所有的句子都 padding 成了 max_seq_length
    max_seq_length: int = field(
        default=128,
        init=False,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        init=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    overwrite_cache: bool = field(
        default=True,
        init=False,
        metadata={"help": "FORCE Overwrite the cached preprocessed datasets or not."}
    )

    def __post_init__(self):
        self.train_file = './data/' + self.data + '/train.tsv'
        self.valid_file = './data/' + self.data + '/valid.tsv'
        self.test_file = './data/' + self.data + '/test.tsv'


@dataclass
class OtherArguments:
    """
    模型的训练参数、其他超参等 可以从 json 里面设置，其余的均为自动生成或者写死
    """
    # 基础模型
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    supcont_pre_epoches: int = field(
        metadata={"help": "训练几个 epoch"}
    )

    loss_type: str = field(
        metadata={"help": "损失函数形式: original loss 表示每层分类器的权重一样，increase 则表示随着层数变高，每层分类器权重升高, plain 表示只使用最后一层做分类"}
    )

    diversity_loss_weight: float = field(
        metadata={"help": "diversity_loss 权重，根据论文，应该在 0-1 之间"}
    )

    scale: float = field(
        metadata={"help": "ensemble scale_ind 参数"}
    )

    # adv_k 小于 0，则使用普通的 train step；否则使用 attack
    adv_k: int = field()
    adv_lr: float = field()
    adv_init_mag: float = field()
    adv_max_norm: float = field()

    scale_ood: float = field(
        default=-1,
        metadata={"help": "ensemble scale_ood 参数"}
    )

    ########################################################################
    # 写死

    cache_dir: Optional[str] = field(
        default=None,
        init=False,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    fitlog_dir: str = field(
        default="./logs",
        init=False
    )

    clip: float = field(
        default=0.25,
        init=False
    )

    # TODO: 可能未来支持?
    # load_trained_model: bool = field(
    #     default=False
    # )

    def __post_init__(self):
        # 向下兼容（早期版本，scale_ind 和 scale_ood 不分家的）
        if self.scale_ood == -1:
            self.scale_ood = self.scale


@dataclass
class FitLogArguments:
    task: str = field(default='AUC', init=False)
