# 5) Finetune
ipython --pdb neuralhydrology/nh_run_scheduler.py finetune -- --directory configs/ensemble_lstm_finetune/ --runs-per-gpu 2 --gpu-ids 0

# 6) Finetune evaluate
ipython --pdb neuralhydrology/nh_run_scheduler.py evaluate -- --directory /cats/datastore/data/runs/ensemble_finetune/FINE --runs-per-gpu 2 --gpu-ids 0

# 7) FINETUNE merge results
ipython --pdb neuralhydrology/utils/nh_results_ensemble.py -- --run-dirs /cats/datastore/data/runs/ensemble_finetune/FINE/* --save-file /cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p --metrics NSE MSE KGE FHV FMS FLV

# 8) Finetune results
cd /home/tommy/tommy_multiple_forcing;
ipython --pdb analysis/read_nh_results.py -- --run_dir /cats/datastore/data/runs/ensemble_finetune/FINE/ --ensemble True --ensemble_filename /cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p;
cd /home/tommy/neuralhydrology;
