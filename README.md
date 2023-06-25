# Hourglass Attention Network for Image Inpainting ECCV 2022
# visualization during training
python - m visdom.server
# train:
python train.py --no_flip --no_rotation --no_augment --img_file your_data --lr 1e-4
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --img_file your_data --lr 1e-5 --continue_train
# test:
python test.py --batchSize 1 --mask_type 3 --img_file your_data --mask_file your_mask
