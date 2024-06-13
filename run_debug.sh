rm -rf paintmind/stage1/__pycache__
python3 train_net.py 2>&1 | tee 'train.log'
