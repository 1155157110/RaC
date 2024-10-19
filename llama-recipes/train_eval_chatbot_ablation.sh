CUDA_VISIBLE_DEVICES=0,1,2,3

ABLATION=$1

if [ ! "$ABLATION" ]; then
    echo "Input an ablation experiment name in ['qa', 'qa_rephrase', 'qar_correct_expl', 'qar_correct_wrong1_expl', 'qar_correct_wrong2_expl']!"
    exit 1
fi

DIR=your_output_dir_$ABLATION
MODEL_SIZE=7
# DATASET=augmented_ablation
DATASET=ablation
LOG_PATH=logs/${MODEL_SIZE}b/$DIR
cd llama-recipes

[ -d $LOG_PATH ] || mkdir -p $LOG_PATH
torchrun > ${LOG_PATH}/epoch1.txt \
    --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
    --enable_fsdp --use_peft --peft_method lora \
    --model_name ../llama-2-${MODEL_SIZE}b/${MODEL_SIZE}B \
    --fsdp_config.pure_bf16 \
    --batch_size_training 12 \
    --dataset custom_dataset \
    --custom_dataset.file "preprocess_data/$DATASET/${ABLATION}.py:get_preprocessed_custom" \
    --output_dir ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/epoch1 \
    --num_epochs 1 \
    --save_model

# continue training from last epoch
for i in `seq 1 19`
do
    torchrun > ${LOG_PATH}/epoch$(($i+1)).txt \
        --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
        --enable_fsdp --use_peft --peft_method lora \
        --model_name ../llama-2-${MODEL_SIZE}b/${MODEL_SIZE}B \
        --fsdp_config.pure_bf16 \
        --batch_size_training 12 \
        --dataset custom_dataset \
        --custom_dataset.file "preprocess_data/$DATASET/${ABLATION}.py:get_preprocessed_custom" \
        --output_dir ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/epoch$(($i+1)) \
        --from_peft_checkpoint ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/epoch$i \
        --num_epochs 1 \
        --save_model
done
