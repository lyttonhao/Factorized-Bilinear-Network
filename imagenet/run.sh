#res34-fmconv
python train_imagenet.py --gpus 0,1,2,3,4,5,6,7  --model-prefix resnet34-fmconv --network resnet-fmconv --batch-size 256 --aug-level 2  --num-epoches 120 --frequent 50  --lr 0.1 --wd 0.0001   --lr-step 60,75,90  --fmconv-drop 0.5  --fmconv-slowstart 1 --depth 34 --fmconv-factor 50

#inception-fmconv
python train_imagenet.py --gpus 4,5,6,7  --model-prefix inception-bn-fmconv --network inception-bn-fmconv --batch-size 128 --aug-level 2 --log-file log/inception-bn-fmconv.log --num-epoches 55 --frequent 50  --lr 0.1 --wd 0.0001  --fmconv-slowstart 1 --fmconv-factor 20 --fmconv-drop 0.5 --lr-step 60,75,90
