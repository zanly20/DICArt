CUDA_VISIBLE_DEVICES="0" python path/to/your/project/DICArt/runners/DICArt_trainer.py \
--data_path path/to/your/project/data/ArtImage-High-level/ArtImage \
--sampling_steps 1000 \
--batch_size 96 \
--eval_freq 10 \
--n_epochs 200 \
--lr 3e-4 \
--seed 42 \
--cate_id 1 \
--num_bins 360 \
--diffusion_steps 1000 \
--saved_model_name experiment_01 \
--pts_encoder pointnet2 \
--pretrained_model_path None \
--is_train 
# --eval 
