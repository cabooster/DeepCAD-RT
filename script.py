import os
import sys

flag = sys.argv[1]

###################################################################################################################################################################
# Only train, using 1 GPU and batch_size=1
if flag == 'train':
    os.system('python train.py --datasets_folder DataForPytorch \
                               --n_epochs 30 --GPU 0 --batch_size 1 \
                               --img_h 150 --img_w 150 --img_s 150 \
                               --train_datasets_size 3500')  

if flag == 'test':
    os.system('python test.py --denoise_model ModelForPytorch --datasets_folder DataForPytorch \
                              --GPU 0 --batch_size 1 \
                              --test_datasize 300')
