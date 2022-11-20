# Hourglass Attention Network for Image Inpainting ECCV 2022
Training:
Train a model (defaullt: random irregular hole)
python train.py --name places2 --img_file your_image_path  --no_flip --no_rotation --no_augment

Set --mask_type in options/base_options.py for different training masks. --mask_file path is needed for external irregular mask, such as the irregular mask dataset provided by Partial Convolutions (the test mask)
To view training results and loss plots, run python -m visdom.server and copy the URL http://localhost:8097.
Training models will be saved under the checkpoints folder.
The more training options can be found in options folder.

Testing:
test the model
python test.py  --name places2 --img_file your_image_path
