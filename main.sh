# shellcheck disable=SC2034

###############################################################################################################################

#for v_dataset in banking_25 banking_75
#for v_dataset in clinc_25 clinc_75
for v_dataset in stackoverflow_25 stackoverflow_75
do
  for v_seed in 42 52 62
  do
      sleep 5
      python run_main.py jsons/plain/${v_dataset}_${v_seed}.json
  done
done
