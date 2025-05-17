import time, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
import re
import ipdb
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
import tiktoken
import transformers
import matplotlib.pyplot as plt
import random
import accelerate
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0')

def generate_causal_mask(question_end_index, answer_blocks, ending_start_index, total_tokens):
    min_dtype = torch.finfo(torch.bfloat16).min
    causal_mask = torch.full((total_tokens, total_tokens), fill_value=0, dtype=torch.bfloat16)
    current_row = question_end_index
    for idx in range(question_end_index):
        causal_mask[idx, 0:idx + 1] = 1
    causal_mask[question_end_index:, :question_end_index] = 1
    for block in answer_blocks:
        start_col = block[0]
        assert start_col == current_row, "Mismatching Triangle"
        end_col = block[1]
        block_len = end_col - start_col
        for i in range(block_len):
            causal_mask[current_row + i, start_col : start_col + i + 1] = 1
        causal_mask[end_col - 1, start_col : end_col] = 1
        current_row += block_len
    causal_mask = causal_mask[None, None, :, :]
    return causal_mask

def visualize_attention_mask(causal_mask):
    causal_mask_float32 = causal_mask.to(torch.float32)
    mask_2d = causal_mask_float32[0, 0].cpu().numpy()
    print(mask_2d)
    cmap = np.zeros_like(mask_2d)
    cmap[mask_2d == 0] = 1 
    cmap[mask_2d == 1] = 0
    plt.figure(figsize=(10, 10))
    plt.imshow(cmap, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()
    
def generate_custom_positional_ids(question_end_index, answer_slices, ending_index, total_tokens):
    pos_ids = list(range(total_tokens))
    for start, end in answer_slices:
        for i in range(start, end):
            pos_ids[i] = (i - start) + question_end_index
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).unsqueeze(0) # Batch size = 1
    return pos_ids_tensor

class SamplePrompt:
    def __init__(self, question):
        self.question = question
        self.question_text = None

    def parse_choices(self):
        choices = []
        start_idx = self.question.find("(A)")
        if start_idx != -1:
            parts = self.question[start_idx:].split(" (")
            for part in parts:
                if ")" in part:
                    letter, choice = part.split(")", 1)
                    letter = letter.strip().replace("(", "").replace(")", "")
                    choices.append((letter, choice.strip()))
        return choices

    def sample_choices_and_prompts(self, sample_size, trials):
        random.seed(0)
        parsed_choices = self.parse_choices()

        sampled_prompts = []
        sampled_choices = []
        self.question_text = self.question.split(" (A)")[0] + " \n"
        for _ in range(trials):
            current_sample = random.sample(parsed_choices, sample_size)
            reassigned_sample = [
                (original_label, chr(65 + i), text)
                for i, (original_label, text) in enumerate(current_sample)
            ]
            current_prompt = " ".join(
                [f"({new_label}) {text}" for _, new_label, text in reassigned_sample]
            )
            current_prompt += " (E) None of the above"
            current_prompt += " Answer:"
            current_prompt += " \n"      
            current_choices = [original_label for original_label, _, _ in reassigned_sample]
            sampled_prompts.append(current_prompt)
            sampled_choices.append(current_choices)
#         sampled_prompts += "<|reserved_special_token_0|>" 
        return sampled_prompts, sampled_choices

    def sample(self, sample_size, trials):
        sampled_prompts, sampled_choices = self.sample_choices_and_prompts(sample_size, trials)
        sampled_question = [self.question_text]
        for prompt in sampled_prompts:
            sampled_question.append(prompt)
        sample_result = "".join(sampled_question)
        return sample_result, sampled_choices
    
def parse_llama_prompt(prompt: str, tokenizer):
    inputs = tokenizer(prompt, padding=False, return_tensors="pt")
    full_tokens = inputs.input_ids
    assistant_marker = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    marker_char_index = prompt.find(assistant_marker)
    ending_index = len(tokenizer.encode(prompt[:marker_char_index]))
    answer_block_marker = "(A)"
    answer_block_char_index = prompt.find(answer_block_marker)
    question_end_index = len(tokenizer.encode(prompt[:answer_block_char_index]))
    answer_blocks_text = prompt[answer_block_char_index:marker_char_index].strip()
    answer_block_lines = answer_blocks_text.splitlines()

    answer_blocks = []
    current_search_index = answer_block_char_index
    for line in answer_block_lines:
        if not line.strip():
            continue
        line_start = prompt.find(line, current_search_index)
        if line_start == -1:
            raise ValueError("Could not locate an answer block line in the prompt.")
        line_end = line_start + len(line)
        start_token_index = len(tokenizer.encode(prompt[:line_start]))
        end_token_index = len(tokenizer.encode(prompt[:line_end]))
        answer_blocks.append([start_token_index, end_token_index])
        current_search_index = line_end
    answer_blocks[-1][1] += 1

    return question_end_index, answer_blocks, ending_index


def sanity_check(prompt: str, tokenizer, question_end_index, answer_blocks, ending_index):
    inputs = tokenizer(prompt, padding=False, return_tensors="pt")
    tokens = inputs.input_ids
    
    question_text = tokenizer.decode(tokens[0][:question_end_index].tolist()).strip()
    print("Question:")
    print(question_text)
    print("-" * 40)
    
    print("Answer Blocks:")
    for idx, (start, end) in enumerate(answer_blocks, start=1):
        block_text = tokenizer.decode(tokens[0][start:end].tolist()).strip()
        print(f"Block {idx}:")
        print(block_text)
        print("-" * 40)
    
    assistant_text = tokenizer.decode(tokens[0][ending_index:].tolist()).strip()
    print("Ending Section:")
    print(assistant_text)
    print("-" * 40)
    
class LLM_Agent:

    def __init__(self, llm, tokenizer, batch_size=1):
        self.tokenizer = tokenizer
        self.llm = llm
        self.A_ids = self.tokenizer("A").input_ids[-1]  #
        self.B_ids = self.tokenizer("B").input_ids[-1]  #
        self.C_ids = self.tokenizer("C").input_ids[-1]  #
        self.D_ids = self.tokenizer("D").input_ids[-1]  #
        
    @torch.no_grad()
    def output_sample_logits(self, input_text):
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        logits_list = []
        total_tokens = input_ids.shape[1]
        q_end, answer_blocks, ending_start_index = parse_llama_prompt(input_text, self.tokenizer)
        attention_mask = generate_causal_mask(q_end, answer_blocks, ending_start_index, total_tokens).to(device)
#         visualize_attention_mask(attention_mask)
        amask = attention_mask.to(dtype=torch.bfloat16)
        amask = (1.0 - amask) * torch.finfo(amask.dtype).min
        p_ids = generate_custom_positional_ids(q_end, answer_blocks, ending_start_index, total_tokens).to(device)
        outputs = self.llm.forward(input_ids=input_ids, attention_mask=amask, position_ids=p_ids)
        all_logits = outputs.logits
        for block in answer_blocks:
            row_num = block[1] - 1
            logits = all_logits[0, row_num, :].flatten()
            probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[self.A_ids],
                        logits[self.B_ids],
                        logits[self.C_ids],
                        logits[self.D_ids],
                    ]
                ),
                dim=0,
            )
            )
            logits_list.append(probs)
        return logits_list
    
def main():
    access_token = "YOUR ACCESS TOKEN HERE"

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_id = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir="./scratch", token=access_token, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
        token=access_token,
        cache_dir="./scratch",
        trust_remote_code=True
    )
    
    ds = load_dataset("allenai/qasc")
    eval_dataset = ds['validation']
    dataset_size = len(eval_dataset)
    llm_ans_buf = []
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    assistance = LLM_Agent(model, tokenizer, batch_size=1)
    save_prob = []
    torch.cuda.synchronize()
    t0 = time.time()
    for ques_idx in tqdm(range(dataset_size)):
        question_doc = eval_dataset[ques_idx]
        prob_map = {choice: [] for choice in choices}
        proba = []
        formatted_question =  question_doc["formatted_question"]
        sampler = SamplePrompt(formatted_question)
        sampled_prompts, sampled_choices = sampler.sample(4, 20)
        logits = assistance.output_sample_logits(sampled_prompts)
        for i in range(len(logits)):
            logit = logits[i]
            cur_choice = sampled_choices[i]
            for i in range(len(cur_choice)):
                prob_map[cur_choice[i]].append(logit[i])   
        for choice in choices:
            temp_prob = prob_map[choice]
            avg = sum(temp_prob) / len(temp_prob)
            proba.append(avg)
        tensor_prob = torch.tensor(proba)
        save_prob.append(tensor_prob)
        llm_ans_buf.append(int(tensor_prob.argmax(-1)))
    torch.cuda.synchronize()
    t1 = time.time()
    answer_key_mapping = {chr(i + ord('A')): i for i in range(8)} 
    mapped_answer_keys = [answer_key_mapping[ans] for ans in eval_dataset["answerKey"]]

    acc_score = accuracy_score(mapped_answer_keys, llm_ans_buf)
    dataset_name = "allenai/qasc"
    results = {
            "model": model_name,
            "dataset": dataset_name,
            "corpus": None,
            "retriever": None,
            "accuracy": acc_score,
            "pred_ans": llm_ans_buf,
            "golden_ans": mapped_answer_keys,
            "time": t1-t0
        }
    model_name = model_name.replace("/", "-")
    dataset_name_new = dataset_name.replace("/", "-")
    results_fname = f"./json/model_{model_name}_dataset_{dataset_name_new}_with_sample4_20trials_seed0.json"
    with open(f"./{results_fname}", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Result saved to {results_fname}")
    
    
if __name__ == "__main__":
    main()
    
