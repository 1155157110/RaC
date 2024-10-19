CUDA_VISIBLE_DEVICES=0,1,2,3

DIR=your_output_dir
MODEL_SIZE=7
FOLD_NUM=20
DATASET=augmented_kfold
# DATASET=kfold
LOG_PATH=logs/${MODEL_SIZE}b/$DIR
cd llama-recipes

for f in `seq 1 $FOLD_NUM`
do
    [ -d ${LOG_PATH}/fold$f ] || mkdir -p ${LOG_PATH}/fold$f
    torchrun > ${LOG_PATH}/fold$f/epoch1.txt \
        --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
        --enable_fsdp --low_cpu_fsdp \
        --use_peft --peft_method lora \
        --model_name ../llama-2-${MODEL_SIZE}b/${MODEL_SIZE}B \
        --fsdp_config.pure_bf16 \
        --batch_size_training 12 \
        --dataset custom_dataset \
        --custom_dataset.file "preprocess_data/$DATASET/$f.py:get_preprocessed_custom" \
        --output_dir ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/fold$f/epoch1 \
        --num_epochs 1 \
        --save_model
done

# continue training from last epoch
for i in `seq 1 39`
do
    for f in `seq 1 $FOLD_NUM`
    do
        [ -d ${LOG_PATH}/fold$f ] || mkdir -p ${LOG_PATH}/fold$f
        torchrun > ${LOG_PATH}/fold$f/epoch$(($i+1)).txt \
            --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
            --enable_fsdp --low_cpu_fsdp \
            --use_peft --peft_method lora \
            --model_name ../llama-2-${MODEL_SIZE}b/${MODEL_SIZE}B \
            --fsdp_config.pure_bf16 \
            --batch_size_training 12 \
            --dataset custom_dataset \
            --custom_dataset.file "preprocess_data/$DATASET/$f.py:get_preprocessed_custom" \
            --output_dir ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/fold$f/epoch$(($i+1)) \
            --from_peft_checkpoint ../llama-2-${MODEL_SIZE}b/chatbot\(finetuned_15k_$DIR\)/fold$f/epoch$i \
            --num_epochs 1 \
            --save_model
    done
done
