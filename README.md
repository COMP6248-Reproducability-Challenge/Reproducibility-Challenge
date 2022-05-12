# COMP6248-Reproducibility-Challenge
## download mini-imagenet
link: https://paperswithcode.com/dataset/miniimagenet-1

## dataset path structure
```
├── mini-imagenet: total 100 classes, 60000 images
     ├── images: 60000 images
     ├── train.csv: 64 classes, 38400 images
     ├── val.csv: 16 classes, 9600 images
     └── test.csv: 20 classes, 12000 images
```

step:

1. run 'restructure_csv.py' to generate classfication dataset: 'new_train.csv' and 'new_val.csv'
2. run 'yange.py' to reduce dataset size: 'new_train_hk.csv' and 'new_val_hk.csv'
3. copy 'new_train_hk.csv' to 'new_train_hk_ori.csv', run 'rewrite_rep.py' to get 'new_train_hk_3000.csv', copy 'new_train_hk_3000.csv' to 'new_train_hk.csv'
4. run 'train_single_gpu.py' to get representation: update 'new_train_hk.csv', change '.weights/' to '.weights_rep/'
5. run 'train_single_gpu_linear.py' to get classification:  'new_train_hk_ori.csv' and 'new_val_hk.csv', , change '.weights/' to '.weights_cls/'
6. rename 'new_train_hk.csv' to 'new_train_hk_v1_final.csv'

experiments:

fully:  
     cls:100
     imagenet-pretrained model
     adam 0.01 epoch 100
     res: 48.4

v1:  
rep:
     cls:100
     imagenet-pretrained model
     train using gt first, then update 'new_train_hk.csv'
     sgd 0.1 epoch 100
     pseudo label generation: transforms.RandomRotation(15)
     rep learning: nothing
cls: adam 0.01
     res: 0.466

v2:  
rep:
     cls:100
     imagenet-pretrained model
     update 'new_train_hk.csv' using pre-trained model first, train next
     sgd 0.1 epoch 90
     pseudo label generation: transforms.RandomRotation(15)
     rep learning: nothing
cls: adam 0.01
     res: 0.459

v3:  
rep:
     cls:500
     imagenet-pretrained model
     initialize 'new_train_hk.csv' with 500 first, train next, then update 'new_train_hk.csv'
     sgd 0.1 epoch 90
     pseudo label generation: transforms.RandomRotation(15)
     rep learning: nothing
cls: adam 0.01
     res: 0.497

v4:  
rep:
     cls:500
     imagenet-pretrained model
     initialize 'new_train_hk.csv' with 500 first, train next, then update 'new_train_hk.csv'
     every 5 epoch
     sgd 0.1 epoch 90
     pseudo label generation: transforms.RandomRotation(15)
     rep learning: nothing
cls: adam 0.01
     res: 0.473

v5:  
rep:
     cls:500
     imagenet-pretrained model
     initialize 'new_train_hk.csv' with 500 first, train next, then update 'new_train_hk.csv'
     sgd 0.1 epoch 90
     pseudo label generation: color jitter 1.0
     rep learning: color jitter 1.0
cls: adam 0.01
     res: 0.485

v6:  
rep:
     cls:500
     imagenet-pretrained model
     initialize 'new_train_hk.csv' with 500 first, train next, then update 'new_train_hk.csv'
     sgd 0.1 epoch 90
     pseudo label generation: color jitter 0.5
     rep learning: color jitter 0.5
cls: adam 0.01
     res: 0.467
