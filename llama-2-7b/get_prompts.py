import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
import tqdm
import pandas as pd

model_id = "7B"
# model_id = "Llama-2-7b-hf"

def get_response(eval_prompt, model, tokenizer):
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    # print(f"model inputs: {model_input}")
    model.eval()
    with torch.no_grad():
        # print("-------------------------------------output: -------------------------------------")
        model_output_tokens = model.generate(**model_input, max_new_tokens=800, pad_token_id=tokenizer.eos_token_id)[0]
        model_output = tokenizer.decode(model_output_tokens, skip_special_tokens=True)
        # print(f"model output: {model_output}")
    
    return model_output[ len(eval_prompt)+1: ]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, help='path to the lora weights')
    parser.add_argument('--dataset', type=str, choices=["EasyDataset", "HardDataset", "ComprehensiveDataset"])
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.lora_path)
    
    test_dataset = pd.read_excel('chatbot_datasets/Data_OpenSource.xlsx', sheet_name=args.dataset)

    output_filename = f"{args.dataset}.txt"
    output_file = open(output_filename, "w")
    for prompt_idx in tqdm.tqdm(range(len(test_dataset['Question']))):
        prompt = test_dataset['Question'][prompt_idx]
        response = get_response(prompt, model=model, tokenizer=tokenizer)
        output_file.write(f"prompt: {prompt_idx}\n{response}\n")