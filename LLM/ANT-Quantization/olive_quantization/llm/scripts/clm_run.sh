transformer_model=${1:-"gpt2"}              # 模型名称
dataset=${2:-"wikitext"}                    # 数据集名称
dataset_config=${3:-"wikitext-103-raw-v1"}  # 数据集配置
q_mode=${4:-"ant-int-flint"}                # 量化模式
q_bit=${5:-"4"}                             # 量化位数
batch_size=${6:-"8"}                        # 批量大小
port=${7:-46666}                            # 端口号
desc=${8:-""}                               # 自定义描述字符串, 用于日志文件名
n8=${9:-"0"}                                #

mkdir -p ./log
mkdir -p ./log/bigscience
mkdir -p ./log/facebook

log_name=""
if [ "$dataset" = "wikitext" ] ; then
  log_name=$transformer_model"_"$dataset_config"_"$q_bit"bit_batch"$batch_size"_"$desc
else
  log_name=$transformer_model"_"$dataset"_"$q_bit"bit_batch"$batch_size"_"$desc
fi

# python -u -m torch.distributed.launch --nproc_per_node=1 --master_port $port run_clm.py \
python -u ./llm/run_clm.py \
  --model_name_or_path $transformer_model \
  --dataset_name $dataset \
  --dataset_config_name $dataset_config \
  --output_dir checkpoints/$transformer_model \
  --do_eval \
  --mode=$q_mode --wbit=$q_bit --abit=$q_bit --a_low=75 --a_up=250 --w_low=75 --w_up=250 --layer_8bit_n=$n8 \
  --eval_batch_size=$batch_size --train_batch_size=$batch_size --quantize_batch_size=$batch_size \
  2>&1 | tee ./log/${log_name}.log \
