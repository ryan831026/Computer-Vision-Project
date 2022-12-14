# pretrain model
python train.py -e 3 -b 1

# train with needle imgs
python needle_train.py -e 1 -b 2 -f 'checkpoints_pretraining/checkpoint_brain.pth' -d 2
python needle_train.py -e 3 -b 2 -f 'checkpoints_pretraining/checkpoint_brain.pth' -d 4 -r

# predict 1 img
python needle_predict.py -m 'checkpoints/n2_brain_dice95_epoch3_flipcolor.pth' -i 'data/imgs_needle2/2022-11-18_17-29-11_left_77.png' -v -n

# predict batch imgs
python needle_predict_batch.py -m 'checkpoints/n2_brain_dice90_epoch2_norandom.pth' -i 'data/imgs_needle2' -o 'output/output2/'
python needle_predict_batch.py -m 'checkpoints/n4_brain_dice86_epoch3_flipcolor.pth' -i 'data/imgs_needle4' -o 'output/output4/'