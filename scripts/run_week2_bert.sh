export CUDA_VISIBLE_DEVICES=0

CONFIG_FILE="../configs/config_bert.yaml"

# Python Inference 실행
# python ../src/main_final.py --config "$CONFIG_FILE" ## accumulation 넣은 것

python ../src/main_accumulation_huggingface.py --config "$CONFIG_FILE" ## accumulation 안 넣은 것