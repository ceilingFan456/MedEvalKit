import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice,judger,judge_judgement,judge_open_end_vqa,get_compare_messages,judge_close_end_vqa
from mathruler.grader import extract_boxed_content
from ..base_dataset import BaseDataset
from ..question_formats import get_judgement_prompt,get_close_ended_prompt

class Radrestruct(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    
    def load_data(self):
        dataset_path = self.dataset_path
        json_path = os.path.join(dataset_path,"testset.json")
        with open(json_path, "r") as f:
            dataset = json.load(f)

        # ['index', 'Figure_path', 'Caption', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Answer', 'split']
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        image_name = sample["image_name"]
        question = sample["question"]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        question_type = sample["question_type"]
        if question_type == "CLOSED":
            prompt = get_judgement_prompt(question,is_reasoning)
        else:
            prompt = get_close_ended_prompt(question,is_reasoning)
        image_path = os.path.join(self.dataset_path,"imgs",image_name)
        img = Image.open(image_path).convert("RGB")
        messages = {"prompt":prompt,"image":img}
        sample["messages"] = messages
        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        metrics = {
            "total metrics" : {
                "total":0,
                "right":0
            },
            "close" : {
                "total" : 0,
                "right" : 0
            }
        }

        open_id = []
        for i,out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            if extract_boxed_content(response)!= "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response,"answer")

            answer = out_sample["answer"]
            question = out_sample["question"]
            answer_type = out_sample["question_type"]
            answer = answer.lower().strip()
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if answer_type == "CLOSED":
                metrics["close"]["total"] += 1
                correct = judge_judgement(answer,response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            else:
                metrics["open"]["total"] += 1

                c_metrics = judge_close_end_vqa(answer,response)
                out_samples[i]["correct"] = c_metrics["em"]
                out_samples[i]["metrics"] = c_metrics
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1 
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric] 

                if os.environ.get("use_llm_judge","False") == "True":
                    messages = get_compare_messages(question,response,answer)
                    messages_list.append(messages)
                    open_id.append(i)

        if os.environ.get("use_llm_judge","False") == "True":
            metrics["total metrics"]["right"] = 0
            metrics["open"]["right"] = 0
            metrics["close"]["right"] = 0
            llm = judger
            results = llm.generate_outputs(messages_list)
            for i,result in zip(open_id,results):
                result = extract(result,"judge")
                result = True if result == "0" else False
                out_samples[i]["correct"] = result
                if result:
                    metrics["open"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

        
        metrics["total metrics"]["acc"] = metrics["total metrics"]["right"]/metrics["total metrics"]["total"]
        metrics["open"]["acc"] = metrics["open"]["right"]/metrics["open"]["total"]
        metrics["close"]["acc"] = metrics["close"]["right"]/metrics["close"]["total"]

        for metric in metrics["open"]:
            if metric not in ["right","total"]:
                metrics["open"][metric] = metrics["open"][metric]/metrics["open"]["total"]
        return metrics,out_samples



                