python needle_train.py -e 3 -b 1 -f 'checkpoints/checkpoint_cars.pth'

python predict.py -m 'checkpoints/checkpoint_epoch5.pth' -i 'xxxxx' -v -n

python predict_batch.py -m 'checkpoints/checkpoint_epoch5.pth' -i 'data/imgs_needle3' -o 'output/'