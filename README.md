## Dependencies
### Use anaconda to create python environemnt:
`conda create --name python=3.7`

### Install all required libraries:
`pip install -r requirements.txt`


## Run:
### The data is located in the "data" folder. Please note that the Clinc dataset is derived from the original dataset and the processing method is described in detail in the "data/clinc/readme.md" file.

### Run the experiments (for example banking_25) Tip: set `do_train` as `True`
   ```
   python run_main.py json/plain/banking_25_52.json
   ```

### The results, including the trained PT model, are stored in the "model_output" directory. To process the results, such as calculating averages, variances, etc., you can use the "analyze_different_scores.py" script.

