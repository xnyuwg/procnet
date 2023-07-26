GPUS="0"

CUDA_VISIBLE_DEVICES=${GPUS} python run.py \
--run_save_name=exp0 \
--batch_size=32 \
--epoch=100
