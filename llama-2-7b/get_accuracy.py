import datasets
import pandas as pd

def evaluate_prompt(prompt, expected_answer):
    import re
    pattern = re.compile(r'Correct answer:\n[ABCDabcd]\)')
    results = pattern.findall(prompt)

    # print(f"{results = }")
    if len(results) > 1:
        for response in results:
            if response[-2].upper() != results[0][-2].upper():
                # print(f"Inconsistent results: {prompt = }")
                return 0
        pass
    if len(results) == 0:
        # print(f"No Answer found: {prompt = }")
        return 0
    if results[0][-2].upper() == expected_answer.upper():
        return 1
    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["EasyDataset", "HardDataset", "ComprehensiveDataset"])
    args = parser.parse_args()

    valid_cnt = 0
    # dataset = f"EasyDataset"
    # dataset = f"HardDataset"
    dataset = args.dataset

    test_set = pd.read_excel('chatbot_datasets/Data_OpenSource.xlsx', sheet_name=dataset)
    correct_answer = [response[0] for response in test_set['Correct Answer']]

    input_file = open(f"{dataset}.txt", 'r')
    prompt_idx = 0
    prompt = ""

    for line in input_file:
        if line == f'prompt: {prompt_idx}\n':
            # print(f"evaluate prompt: {prompt}")
            if prompt_idx != 0:
                valid_cnt += evaluate_prompt(prompt, correct_answer[prompt_idx-1])
            prompt_idx += 1
            prompt = ""
        else:
            prompt += line
    # evaluate last prompt
    valid_cnt += evaluate_prompt(prompt, correct_answer[prompt_idx-1])
    
    # print(f"{dataset:15} Valid: {valid_cnt[dataset]}")

    print(f"{'# of correct responses ('+str(len(correct_answer))+' in total)':23}")
    total_question_num = len(correct_answer)
    print(f"{dataset:20} {valid_cnt} correct in {total_question_num}, accuracy: {valid_cnt/total_question_num:3f}")
