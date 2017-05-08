#cifar100
python train_cifar.py --gpus 2,3 --data-dir /mnt/hdd/lytton/mx_data/cifar-100 --lr 0.2 --wd 0.0001 --batch-size 128 --data-shape 32 --num-epoches 400 --network inception-bn-small --num-classes 100 --lr-step 200,300 --model-prefix cifar100_inception

#cifar100 fmconv
python train_cifar.py --gpus 2,3 --data-dir /mnt/hdd/lytton/mx_data/cifar-100 --lr 0.2 --wd 0.0001 --batch-size 128 --data-shape 32 --num-epoches 400 --network inception-bn-small-fmconv --num-classes 100 --lr-step 200,300 --model-prefix cifar100_inception_fmconv --fmconv-slowstart 3  --fmconv-drop 0.5 --fmconv-factor 20 



#resnet 
python train_cifar.py --gpus 2,3 --data-dir /mnt/hdd/lytton/mx_data/cifar-100 --lr 0.1 --wd 0.0001 --batch-size 128 --data-shape 32 --num-epoches 200 --network resnet-small --num-classes 100 --lr-step 100,150 --model-prefix cifar100_res18 --warmup --res-module-num 18

#cifar100 fmconv
python train_cifar.py --gpus 2,3 --data-dir /mnt/hdd/lytton/mx_data/cifar-100 --lr 0.1 --wd 0.0001 --batch-size 128 --data-shape 32 --num-epoches 200 --network resnet-small-fmconv --num-classes 100 --lr-step 100,150 --model-prefix cifar100_res18_fmconv  --res-module-num 18  --fmconv-slowstart 3 --fmconv-drop 0.5 --fmconv-factor 20


