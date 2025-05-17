import time, json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import re
import ipdb
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import default_data_collator
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tiktoken
import accelerate

class LLM_Agent:

    def __init__(self, llm, tokenizer, batch_size=1):
        self.tokenizer = tokenizer
        self.llm = llm
        self.tokenizer.padding_side = "left"
        self.A_ids = self.tokenizer("A").input_ids[-1]  #
        self.B_ids = self.tokenizer("B").input_ids[-1]  #
        self.C_ids = self.tokenizer("C").input_ids[-1]  #
        self.D_ids = self.tokenizer("D").input_ids[-1]  #
        self.E_ids = self.tokenizer("E").input_ids[-1]  #
        self.F_ids = self.tokenizer("F").input_ids[-1]  #
        self.G_ids = self.tokenizer("G").input_ids[-1]  #
        self.H_ids = self.tokenizer("H").input_ids[-1]  

    @torch.no_grad()
    def output_logit(self, input_text, **kwargs):
        inputs = self.tokenizer(input_text, padding=False, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        logits = self.llm(
            input_ids=input_ids,
        ).logits[:, -1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[self.A_ids],
                        logits[self.B_ids],
                        logits[self.C_ids],
                        logits[self.D_ids],
                        logits[self.E_ids],
                        logits[self.F_ids],
                        logits[self.G_ids],
                        logits[self.H_ids],
                    ]
                ),
                dim=0,
            )
        )

        return probs

def main():
    access_token = "YOUR ACCESS TOKEN HERE"
    dataset_name = "allenai/qasc"
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir="./scratch", token=access_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=access_token,
        cache_dir="./scratch",
        trust_remote_code=True
    )
    assistance = LLM_Agent(model, tokenizer, batch_size=1)
    ds = load_dataset("allenai/qasc")
    eval_dataset = ds['validation']
    dataset_size = len(eval_dataset)
    llm_ans_buf = []
    save_prob = []
    torch.cuda.synchronize()
    t0 = time.time()
    for ques_idx in tqdm(range(dataset_size)):
        question_doc = eval_dataset[ques_idx]
        prompt = question_doc["formatted_question"] + " Answer:"
        prob = assistance.output_logit(prompt)
        save_prob.append(prob)
        llm_ans_buf.append(int(prob.argmax(-1)))
    torch.cuda.synchronize()
    t1 = time.time()
    answer_key_mapping = {chr(i + ord('A')): i for i in range(8)}
    mapped_answer_keys = [answer_key_mapping[ans] for ans in eval_dataset["answerKey"]]
    acc_score = accuracy_score(mapped_answer_keys, llm_ans_buf)
    results = {
        "model": model_id,
        "dataset": dataset_name,
        "corpus": None,
        "retriever": None,
        "accuracy": acc_score,
        "pred_ans": llm_ans_buf,
        "golden_ans": mapped_answer_keys,
        "time": t1-t0
    }
    model_name = model_id.replace("/", "-")
    dataset_name_new = dataset_name.replace("/", "-")
    results_fname = f"./json/model_{model_name}_dataset_{dataset_name_new}_standard_inference.json"
    with open(f"./{results_fname}", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Result saved to {results_fname}")
        
if __name__ == "__main__":
    main()