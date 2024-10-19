# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/rxavier/economicus

import copy
import datasets


def get_preprocessed_custom(dataset_config, tokenizer, split):
    dataset_file = f"../llama-2-7b/chatbot_datasets/Datasets_15k_augment_20fold/train/augmented_train_split/Datasets_15k_augmented_train_{19}fold_20.csv"
    dataset = datasets.load_dataset("csv", data_files=dataset_file, split='train', encoding = "utf-8")

    if split == 'train':
        # dataset = datasets.load_dataset("csv", data_files="/home/openwifi-lab2/Desktop/llama/llama-2-7b/chatbot_datasets/Datasets_15k.csv", split='train', encoding = "ISO-8859-1")
        dataset = dataset.filter(lambda x, idx: idx < int(dataset.num_rows*0.95), with_indices=True)
    elif split == 'validation':
        dataset = dataset.filter(lambda x, idx: int(dataset.num_rows*0.95) < idx, with_indices=True)
    else:
        raise NotImplementedError
    # dataset = dataset.filter(lambda x: x['Question'])

    def tokenize_add_label(sample):
        question = sample['Question']+'\n'
        answer_rephrase = "Correct answer:\n"+sample['Correct Answer']+"\n"
        answer_rephrase += "Explanation:\n"+sample['Rephrase']+"\n"
        answer_explanation = answer_rephrase
        answer_explanation += "The explanation to the correct answer is: "+sample['Correct Answer Expl']+"\n"
        answer_explanation += "The explanation to some wrong answers are:\nWrong answer: "+sample['Wrong Answer_1']
        answer_explanation += "\nWrong answer: "+sample['Wrong Answer_2']
        answer_explanation += "\nWrong answer: "+sample['Wrong Answer_3']

        # print(f"question:\n{question}\n")
        # print(f"answer_rephrase:\n{answer_rephrase}\n")
        # print(f"answer_explanation:\n{answer_explanation}\n")
        # exit(0)
        prompt = tokenizer.encode(tokenizer.bos_token + question, add_special_tokens=False)
        # response = tokenizer.encode(answer_rephrase +  tokenizer.eos_token, add_special_tokens=False)
        response = tokenizer.encode(answer_explanation +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + response,
            "attention_mask" : [1] * (len(prompt) + len(response)),
            "labels": [-100] * len(prompt) + response
        }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset
