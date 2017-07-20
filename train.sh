# Inception v3 checkpoint file.
ROOT_PATH="${HOME}/youngjung/cycleGAN_multimodal"
INCEPTION_CHECKPOINT="${ROOT_PATH}/model/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${ROOT_PATH}/model/train/"

export CUDA_VISIBLE_DEVICES=1
# Run the training script.
python im2txt/train.py \
  --input_file_pattern="${MSCOCO_DIR}/forShowAndTell/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}" \
  --train_inception=False \
  --number_of_steps=1000000 \
  --log_every_n_steps=10 \
  --vocab_file="${MSCOCO_DIR}/forShowAndTell/word_counts.txt"
