# Hourglass Attention Network for Image Inpainting ECCV 2022
# visualization during training
python - m visdom.server
# train:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-4
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-5 --continue_train
# test:
python test.py --mask_type 3 --image_file your_data --mask_file your_mask --continue_train
