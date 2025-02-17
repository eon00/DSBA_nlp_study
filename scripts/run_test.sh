export CUDA_VISIBLE_DEVICES=0

# Python 학습 스크립트 실행
# python ../train.py --config "$CONFIG_FILE"

CONFIG_FILE="../configs/config.yaml"

# Python Inference 실행
python ../src/nlp_main.py --config "$CONFIG_FILE"