# pretrain model
python train.py -e 3 -b 1 -f 'checkpoints_pretraining/checkpoint_cars.pth'

# train with needle imgs
python needle_train.py -e 3 -b 1 -f 'checkpoints_pretraining/checkpoint_cars.pth'

# predict 1 img
python needle_predict.py -m 'checkpoints/checkpoint_epoch5.pth' -i 'data/imgs_needle3/2022-11-21_21-14-25_75.png_left.png' -v -n

# predict batch imgs
python neelde_predict_batch.py -m 'checkpoints/checkpoint_epoch5.pth' -i 'data/imgs_needle3' -o 'output/'