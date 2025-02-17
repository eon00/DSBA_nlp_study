export CUDA_VISIBLE_DEVICES=1

CONFIG_FILE="../configs/config.yaml"
Mode="train"

# Python Inference 실행
python ../src/nlp_main.py --config "$CONFIG_FILE" --mode "$Mode"