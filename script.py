import os
import sys

flag = sys.argv[1]

###################################################################################################################################################################
# Only train, using 1 GPU and batch_size=1
if flag == 'train':
    os.system('python train.py --datasets_folder DataForPytorch \
                               --n_epochs 40 --GPU 0,1 --batch_size 2 \
                               --img_h 150 --img_w 150 --img_s 150 \
                               --train_datasets_size 3000')  

if flag == 'test':
    os.system('python test.py --denoise_model ModelForPytorch --datasets_folder DataForPytorch \
                              --GPU 0,1 --batch_size 2 \
                              --test_datasize 300')
