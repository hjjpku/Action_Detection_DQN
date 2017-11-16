th Hjj_Training.lua -gpu [] -data_path ActivityNet -class [] -alpha [] -log_err ./log/training_error_[].log -log_log ./log/log[] -batch_size [] -replay_buffer [] -lr [] -epochs []

th Hjj_Validate2.lua -data_path Thumos -class 4 -model_name ./New/g_1_23_fine -gpu 0 -alpha 0.2


CUDA_VISIBLE_DEVICES=0 th ./Hjj_Training12.lua -data_path Thumos -name a -alpha 0.2 -log_err log/traning_error_a -batch_size 200 -replay_buffer 2000 -lr 1e-4 -epochs 50 &> ./local.txt 

CUDA_VISIBLE_DEVICES=2 th Hjj_Training9.lua -gpu 0 -data_path Thumos -name h -class 9 -alpha 0.2 -log_err ./log/training_error_h -batch_size 200 -replay_buffer 2000 -lr 1e-2 -epochs 50 > ./log/logh
