# single prefix
python train_logit.py --model hidden-logit-residual --data_dir  logits/gsm/dexperts-7B-chat7B/ --lr 1e-5 --epoch 1000 --save_epoch 1000 --gpu 2 --prefix 0_1000
# multi prefix
python train_logit.py --model hidden-logit-residual --data_dir  logits/all/dexperts-7B-chat7B/ --lr 1e-5 --epoch 1000 --save_epoch 100 --gpu 2 --prefix 0_1000 1000_1500 1500_2500 2500_3300 
