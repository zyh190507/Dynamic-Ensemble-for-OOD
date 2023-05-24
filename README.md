## Dependencies
### Use anaconda to create python environemnt:
`conda create --name python=3.7`

### Install all required libraries:
`pip install -r requirements.txt` 注意 torch 需要手动安装


### Run:
1. 数据在 data 文件夹下，注意 clinc 数据集是由原始数据集处理得到的，处理方法详见 data/clinc/readme.md
2. 训练使用示例，注意 `do_train` 要置为 `True`
    ```
   python run_main.py json/plain/banking_25_52.json
   ```
3. 结果（包括训练完的 pt 模型）都放在了 model_output 里面。处理结果（平均数、方差等）可以使用 analyze_different_scores.py
