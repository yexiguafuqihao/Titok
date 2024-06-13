rm -rf paintmind/__pycache__  paintmind/modules/__pycache__ paintmind/modules/mask2former/__pycache__ paintmind/engine/__pycache__
rm -rf paintmind/stage1/__pycache__ paintmind/stage1/__pycache__
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
NCCL_DEBUG=INFO accelerate launch --config_file=$1 train_net.py  2>&1 | tee 'train.log'