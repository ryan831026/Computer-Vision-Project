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

# testing with batch imgs
python needle_predict_batch.py -m 'checkpoints/n4_brain_dice84_epoch3_flipcolor.pth' -i 'data/imgs_testing2' -o 'output/testing/'

# How training data was obtained:
## Training data preparataion
1) Download the example insertion ros2 bag file from here and place it to `get_segments` folder: 
https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/yzhetpi1_jh_edu/EgFb4PZsU9hJtzmr1aYiHuIB4nesFivvtPTX07DqWVyLnw?e=bemnKr
2) Sequentially run the following commands from get_segments folder:

```
#Parsing and saving to file needle insertion depths positions:
python3 process_bag.py 2022-11-23_18-42-03 -r
```

```
#Parsing, randomly sampling and saving to file timestamps:
python3 process_bag.py 2022-11-23_18-42-03 -c -left -isTimeStamp
python3 process_bag.py 2022-11-23_18-42-03 -c -right -isTimeStamp
```

```
#Saving left and right images
python3 process_bag.py 2022-11-23_18-42-03 -c -right
python3 process_bag.py 2022-11-23_18-42-03 -c -left
```

```
#Get masks:
python3 needle_segment/main.py
```