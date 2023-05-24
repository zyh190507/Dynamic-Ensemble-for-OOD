"""
banking 数据集一致

clinc 数据集中，
经过对比发现，adb 的 dev、train 数据集和 knn_con 的对应文件保持一致
但是 adb 的 test 数据集 = knn_con 的 test.tsv+train_oos.tsv+valid_oos.tsv

stackoverflow 数据集一致

因此本文要对 clinc 数据加以处理
"""
import os

import pandas as pd


def read_tsv_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep='\t', dtype=str)
    return df


def save_tsv_file(filename: str, df: pd.DataFrame) -> None:
    df.to_csv(filename, sep="\t", index=False)


def main():
    original_root = './data/clinc_full'

    new_root = './data/clinc'
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    data = read_tsv_file(os.path.join(original_root, 'train.tsv'))
    # 防呆：clinc train 数据里面没有 oos 数据
    assert data.label.isin(['oos', 'ood']).values.sum() == 0
    save_tsv_file(os.path.join(new_root, 'train.tsv'), data)

    data = read_tsv_file(os.path.join(original_root, 'valid.tsv'))
    # 防呆：clinc valid 数据里面没有 oos 数据
    assert data.label.isin(['oos', 'ood']).values.sum() == 0
    save_tsv_file(os.path.join(new_root, 'valid.tsv'), data)

    # 处理 test
    knn_con_train_oos_data = read_tsv_file(os.path.join(original_root, 'train_oos.tsv'))
    knn_con_valid_oos_data = read_tsv_file(os.path.join(original_root, 'valid_oos.tsv'))
    knn_con_test_data = read_tsv_file(os.path.join(original_root, 'test.tsv'))

    knn_con_test_oos_indices = knn_con_test_data.label.isin(['oos'])
    new_test_data = pd.concat([
        knn_con_test_data[~knn_con_test_oos_indices],
        knn_con_train_oos_data,
        knn_con_valid_oos_data,
        knn_con_test_data[knn_con_test_oos_indices]
        ], ignore_index=True)

    save_tsv_file(os.path.join(new_root, 'test.tsv'), new_test_data)


if __name__ == '__main__':
    main()
