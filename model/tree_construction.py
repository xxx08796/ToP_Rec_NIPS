from collections import UserList

from regex import F
from ToP.llm import LLMMessage, LLMRegistry
import yaml
import pandas as pd
import numpy as np
import os
import json
import asyncio
from tqdm.asyncio import tqdm
from ToP.utils import safe_parser, find_tree_by_category
from typing import List, Dict, Any
import random
from logging import getLogger
from ToP.recommender import TweetRecommender, TreeConstructor
import argparse
import time



dataset = "twitter"
parser = argparse.ArgumentParser()


parser.add_argument("--dataset_root_path",
                    default=f'./../dataset/{dataset}',
                    type=str,
                    help="root path of dataset")

parser.add_argument("--model_name",
                    default='Qwen2.5-72B-Instruct-AWQ',
                    choices=['Qwen2.5-32B-Instruct-AWQ','QwQ-32B-AWQ'],
                    type=str,
                    required=False,
                    help="llm model name")


parser.add_argument("--user_size",
                    default=None,
                    type=int)

parser.add_argument("--method",
                    default='kmeans',
                    type=str,
                    choices=['kmeans', 'random'],
                    help='the method for tree construction')

parser.add_argument("--workflow",
                    default="inter_augmentation",
                    choices=["data_construct", "inter_augmentation", "all", 'rebalance'])

parser.add_argument("--prob",
                    default=True,
                    type=bool,
                    help="whether to use the prob tweeet selection instead of llm selection")
parser.add_argument("--prob_method",
                    default = "emb",
                    choices = ["random", "emb"],
                    help="the method for prob tweet selection")
parser.add_argument("--emb_model", 
                    default = "distiluse-base-multilingual-cased-v2",
                    help="model for the uer/tweet embedding")
parser.add_argument("--num_cats", 
                    default = 3)
parser.add_argument("--num_leaves",
                    default = 3)
parser.add_argument("--num_augs", type=int, default=3,)

parser.add_argument("--aug_ratio",
                    type = float,
                    default = None)
parser.add_argument("--sample_weight", type=float, default=5.0,
                    help="Weight for sampling new categories")


args = parser.parse_args()
args.root_content_path = f'{args.dataset_root_path}/{dataset}.item'
args.user_info_path = f'{args.dataset_root_path}/{dataset}.user'

args.interaction_history_path = f'{args.dataset_root_path}/{dataset}.train.inter'

args.emb_path = f'{args.dataset_root_path}/embeddings'
args.prompts_path = f'{args.dataset_root_path}/prompts.yaml'
args.rebalance_prompts_path = f'{args.dataset_root_path}/rebalance_prompts.yaml'
args.output_dir = args.dataset_root_path
args.tree_path = f'{args.output_dir}/{args.method}_category_trees.json'
args.classified_root_content_path = f'{args.output_dir}/{args.method}_tree_classified_root_content.csv'
args.inter_aug_path = f'{args.output_dir}/inter_augment.txt'

args.storage_dir = f'{args.dataset_root_path}/storage'
args.rebalance_tree_path = f'{args.output_dir}/{args.method}_category_trees_rebalanced.json'
args.rebalance_classified_root_content_path = f'{args.output_dir}/{args.method}_tree_classified_root_content_rebalanced.csv'


async def main():
    start_time = time.time()
    if args.workflow in ["data_construct", "all"]:
        constructor = TreeConstructor(args)
        await constructor.construct_tree(method=args.method)
        await constructor.classify_and_save_tweets()
        await constructor.tree_rebalance()
    if args.workflow in ["inter_augmentation", "all"]:
        recommender = TweetRecommender(args)
        await recommender.recommend()
        
    end_time = time.time()
    time_consumption = end_time - start_time
    print(f"Tims consumption {time_consumption:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())

