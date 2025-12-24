from typing import List, Dict, Any, Optional
from ToP.llm import LLMMessage, LLMRegistry
import pandas as pd
from ToP.utils import safe_parser, validate_tree_structure, find_tree_by_category, async_task_runner
import yaml
import asyncio
from tqdm.asyncio import tqdm
import time
import json
import os
import numpy as np
from .tree_manager import TreeManager, dict_to_tree




class TreeConstructor:
    '''
    construct the category tree
    assign the tweet to the leaf nodes
    check the balance and branch pruning
    params:
        root_content: the root content of the tweets, # contain columns: mid,content,category
        prompts_path: the path of the prompts
        tree_path: the path to save the dict tree
        classified_tweet_path: save the classified root content
    '''
    def __init__(self, args, ):
        self.root_content = pd.read_csv(args.root_content_path, sep='\t', usecols=['tweet_id:token', 'content:token', 'keyword:token']).rename(columns={'tweet_id:token':'mid', 'content:token':'content', 'keyword:token':'category'})
        self.categories = self.root_content["category"].unique()
        #self.root_content = root_content
        self.model_name = args.model_name
        self.model = LLMRegistry().get(model_name=args.model_name)
        with open(args.prompts_path, 'r', encoding='utf-8') as file:
            self.prompts = yaml.load(file, Loader=yaml.FullLoader)
        with open(args.rebalance_prompts_path, 'r', encoding='utf-8') as file:
            self.rebalance_prompts = yaml.load(file, Loader=yaml.FullLoader)
        self.tree_path = args.tree_path
        self.emb_path = args.emb_path
        self.trees:List[Dict] = []
        self.classified_df:pd.DataFrame = None
        self.classified_tweet_path = args.classified_root_content_path
        self.rebalance_tree_path = args.rebalance_tree_path
        self.rebalance_classified_tweet_path = args.rebalance_classified_root_content_path


    async def construct_tree(self, method= "random"):
        if method == "kmeans":
            emb_dict = np.load(f'{self.emb_path}/item_emb.npy', allow_pickle=True).item()
            tasks = [
                self.generate_category_tree_kmeans(category=cat, emb_dict=emb_dict)
                for cat in self.categories
            ]
        #self.trees = await asyncio.gather(*tasks)
        self.trees = await async_task_runner(tasks, describe="Constructing category trees", max_concurrent=6)
        os.makedirs(os.path.dirname(self.tree_path), exist_ok=True)

        with open(self.tree_path, 'w', encoding='utf-8') as file:
            json.dump(self.trees, file, ensure_ascii=False, indent=2)



    async def generate_category_tree_kmeans(self, category: str, emb_dict: dict, sample_size: int=140, c: int=10):
        mid = np.array(emb_dict['mid'], dtype=int); emb = emb_dict['emb']
        category_tweets = self.root_content[self.root_content["category"] == category]["content"]
        if len(category_tweets) > sample_size:
            category_tweets_id = self.root_content[self.root_content["category"] == category]["mid"]
            cate_idx = np.where(np.isin(mid, category_tweets_id))[0]; cate_emb = emb[cate_idx, :]
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=c, random_state=42); kmeans.fit(cate_emb); cluster_labels = kmeans.labels_
            samples_per_cluster = sample_size // c; remainder = sample_size % c
            selected_samples = []
            for cluster_id in range(c):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if cluster_id < remainder: num_samples = samples_per_cluster + 1
                else: num_samples = samples_per_cluster
                if len(cluster_indices) > num_samples: sampled_indices = np.random.choice(cluster_indices, num_samples, replace=False)
                else: sampled_indices = cluster_indices
                selected_samples.append(cate_idx[sampled_indices])  # to global idx
            selected_samples = np.concatenate(selected_samples)
            selected_weibo_id = mid[selected_samples]
            sample_tweets = self.root_content[self.root_content["mid"].isin(selected_weibo_id)]["content"].tolist()
            # sample_tweets = category_tweets.sample(n=sample_size, replace=True).tolist()
        else:
            sample_tweets = category_tweets.tolist()

        # sample_tweets_str = "\n".join(sample_tweets)
        sample_tweets_str = "\n".join([f"{i + 1}: {content}" for i, content in enumerate(sample_tweets)])
        template = self.prompts["category_tree_prompt"]
        tree_split_prompt = LLMMessage(role="user", content=template.format(category=category, sample_tweets=sample_tweets_str))
        for i in range(5):
            try:
                response = await self.model.generate_response_async(messages=[tree_split_prompt], show_tokens=True)
                response = safe_parser(response)
                assert isinstance(response, dict), f"Expected response to be a dictionary, but got {type(response)}"
                if response["root"]:
                    response["root"] = category
                    validate_tree_structure(node = response, is_root=True, depth=0)
                    return response
                else:
                    raise ValueError("false tree structure with no root")
                    #print("false tree structure!")
            except Exception as e:
                print(e)
                print("retrying")

    async def classify_tweet_with_llm(self, tweet_content:str, max_idx:int, id_map:dict, leaf_path:list):
        #template = self.prompts["tweet_classify_prompt"]
        template = self.prompts["tweet_classify_paths"]
        prompt = template.format(tweet_content=tweet_content, leaf_path = leaf_path, id_map = id_map)
        msg = LLMMessage(role="user", content=prompt)

        for i in range(5):
            try:
                response = await self.model.generate_response_async(messages=[msg])
                response = safe_parser(response)
                leaf_id = int(response['leaf_id']) if isinstance(response['leaf_id'], str) else response['leaf_id']
                leaf_name = response['name']
                if 0< leaf_id <= max_idx:
                    if id_map[leaf_id] == leaf_name:
                        return leaf_id
                    else:
                        #print(msg)
                        raise ValueError(f"Invalid response {response},leaf_id {leaf_id}: with unmatched leaf_name {leaf_name}, actual leaf name {id_map[leaf_id]}")
                else:
                    raise ValueError(f"Invalid response {response},leaf_id {leaf_id}: exceeds max_idx {max_idx}")

            except Exception as e:
                print(e)
                print("Retrying...")

    async def classify_tweet_for_sub_df(self, df, category):
        '''
        classify the tweets share the same category
        '''
        try:
            with open(self.tree_path, 'r', encoding='utf-8') as file:
                self.trees = json.load(file)
        except Exception as e:
            print("Construct the tree first")
        #assert self.trees != [], "Construct the tree first"
        idx,category_tree, id2name, leaf_path = find_tree_by_category(category, self.trees)
        if category_tree is None:
            return None
        tasks = []
        for _, row in df.iterrows():
            content = row["content"]
            task = self.classify_tweet_with_llm(content, max_idx=idx, id_map=id2name, leaf_path=leaf_path)
            tasks.append(task)

        # leaf_ids = await asyncio.gather(*tasks)
        leaf_ids = await async_task_runner(tasks,max_concurrent=6, describe=f"Classifying category {category}")

        df['leaf_id'] = leaf_ids
        return df

    async def classify_and_save_tweets(self):
        grouped = self.root_content.groupby('category')

        tasks = []
        for category, group in grouped:
            task = self.classify_tweet_for_sub_df(df = group, category = category)
            tasks.append(task)

        #classified_dfs = await asyncio.gather(*tasks)
        classified_dfs = []
        for task in tqdm(tasks, desc="Processing categories", total=len(tasks)):
            classified_df = await task
            classified_dfs.append(classified_df)

        self.classified_df = pd.concat(classified_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(self.classified_tweet_path), exist_ok=True)

        self.classified_df.to_csv(self.classified_tweet_path, index=False)
        print(f"Classified tweets saved to {self.classified_tweet_path}")

    async def tree_rebalance(self):
        
        # assert self.trees != [], "generarte trees first"
        # assert self.classified_df is not None, "classify tweets first"
        if self.trees == []:
            with open(self.tree_path, 'r', encoding='utf-8') as file:
                self.trees = json.load(file)
            self.classified_df = pd.read_csv(self.classified_tweet_path)
        rebalance_trees = []
        rebalance_dfs = []
        for tree in self.trees:
            category  = tree["root"]
            cat_df = self.classified_df[self.classified_df["category"] == category]
            root_node = dict_to_tree(tree_dict =tree, df = cat_df)
            manager = TreeManager(tree = root_node, tweet_df = cat_df, 
                                  prompts = self.rebalance_prompts,
                                  model_name = self.model_name)
            await manager.rebalance_tree()
            tree_dict, tweet_df = manager.tree_to_dict()
            rebalance_trees.append(tree_dict)
            rebalance_dfs.append(tweet_df)

        os.makedirs(os.path.dirname(self.rebalance_tree_path), exist_ok=True)

        with open(self.rebalance_tree_path, 'w', encoding='utf-8') as file:
            json.dump(rebalance_trees, file, ensure_ascii=False, indent=2)
        
        rebalance_tweet_df = pd.concat(rebalance_dfs, ignore_index=True)
        rebalance_tweet_df.to_csv(self.rebalance_classified_tweet_path, index=False)
        print(f"Rebalance classified tweets saved to {self.rebalance_classified_tweet_path}")
        


