CUDA_VISIBLE_DEVICES="0" python path/to/your/project/DICArt/runners/DICArt_eval.py \
--data_path path/to/your/project//data/ArtImage-High-level/ArtImage \
--sampling_steps 1000 \
--batch_size 1 \
--seed 42 \
--cate_id 1 \
--saved_model_name experiment_01 \
--pts_encoder pointnet2 \
--pretrained_model_path_test path/to/your/project//DICArt/D3PM_6D/epoch_model_epoch_179_angle_diff_28.9099_trans_diff_0.1237.pt \
--num_bins 360 \
--diffusion_steps 1000 \
--eval 

