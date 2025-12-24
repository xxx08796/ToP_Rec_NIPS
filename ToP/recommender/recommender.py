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
from ToP.utils import safe_parser, find_tree_by_category,async_task_runner
from typing import List, Dict, Any,Optional
import random 
from logging import getLogger
from pydantic import BaseModel, ConfigDict
from .dialogue_tree import DialogueTree, User
from .embedding_selector import EmbeddingSelector
import math
# import logging
# logger = logging.getLogger(__name__)


class TweetRecommender():
    def __init__(self, args):
        self.user_info = pd.read_csv(args.user_info_path, sep='\t',usecols=['user_id:token', 'description:token']).rename(columns={'user_id:token': 'user_idx', 'description:token':'profile_description'})
        self.interaction_history = pd.read_csv(args.interaction_history_path,sep="\t", header=0, names=["user_idx", "mid"])
        self.root_content = pd.read_csv(args.classified_root_content_path) # [mid, content, category, leaf_id]
        self.interaction_history['category'] = self.interaction_history['mid'].map(dict(zip(self.root_content['mid'], self.root_content['category'])))
        self.model_name = args.model_name
        self.storage_dir = args.storage_dir
        self.num_cats = args.num_cats
        self.num_leaves = args.num_leaves
        self.num_augs = args.num_augs
        self.aug_ratio = args.aug_ratio 
        self.sample_weight = args.sample_weight
        

        with open(args.prompts_path, 'r', encoding='utf-8') as file:
            self.prompts = yaml.load(file, Loader=yaml.FullLoader)["interaction_augmentation_prompt"]
        with open(args.tree_path, 'r', encoding='utf-8') as file:
            self.category_trees = json.load(file)
        
        self.total_categories = self.root_content['category'].unique().tolist()
        if "others" in self.total_categories:
            self.total_categories.remove("others")
        self.model = LLMRegistry().get(self.model_name)
        self.inter_aug_path = args.inter_aug_path
        self.max_try:int = 3
        self.augment_users:List = []
        self.augment_user_size = args.user_size if args.user_size else None
        self.prob = args.prob
        self.prob_method = args.prob_method

        
    def _select_users(self, interaction_sample_size = 10, user_idxs = None):
        N = 10^3
        """
        select the users to be augmented based on its history interactions

        """
        if user_idxs:
            selected_user_idxs = user_idxs
        else:
            if not self.augment_user_size:
                selected_user_idxs = list(self.interaction_history['user_idx'].unique())
            else:
                selected_user_idxs = random.sample(list(self.interaction_history['user_idx'].unique()), self.augment_user_size) # selection should base on the history
        for user_idx in selected_user_idxs:
            profile_description = self.user_info[self.user_info['user_idx'] == user_idx]['profile_description'].values[0]
            user_interactions = self.interaction_history[self.interaction_history['user_idx'] == user_idx]
            #user_interactions = user_interactions.sample(n=10, random_state=42)
            # retrieve the mid of history interactions
            tweet_ids = user_interactions['mid'].unique().tolist()
            interaction_tweet_df = self.root_content[self.root_content['mid'].isin(tweet_ids)]
            categories = interaction_tweet_df['category'].unique().tolist()# store the  categories that had interactions
            if len(interaction_tweet_df) < interaction_sample_size:
                sampled_interaction_tweet_df = interaction_tweet_df
            else:
                sampled_interaction_tweet_df = interaction_tweet_df.sample(n=interaction_sample_size, random_state=42)
            sampled_interaction_history = "\n".join([f"Category: {category} - Content: {content}" for content, category in zip(sampled_interaction_tweet_df['content'], sampled_interaction_tweet_df['category'])])
            # tweet_data.extend(interaction_tweets)
            self.augment_users.append(User(idx = user_idx, profile_description=profile_description, 
                                           interaction_history=sampled_interaction_history, category_history=categories,
                                           inter_nums = len(interaction_tweet_df)
                                           ))

            # generate the user's overall desrciption based on the profile and interaction history
        #print(len(self.augment_users))
        self.embedding_selector = EmbeddingSelector(storage_dir=self.storage_dir, root_content=self.root_content, augmented_users=self.augment_users)
        

    async def _select_categories(self, user: User) -> None:
        """ 
        N \leq 3
        select the possibily interested categories for the user 
        """
        for _ in range(self.max_try):
            try:
                #candidate_categories = [cat for cat in self.total_categories if cat not in user.category_history]
                candidate_categories = self.total_categories
                cat_prompt = self.prompts["category_selection"].format(
                    user_profile=user.profile_description, 
                    #interaction_history=user.interaction_history,
                    interaction_history = "",
                    candidate_categories=candidate_categories,
                    num_cats = self.num_cats
                )

                msgs = [LLMMessage(role='user', content=cat_prompt)]
                response = await self.model.generate_response_async(messages=msgs)
                msgs.append(LLMMessage(role='assistant', content=response))

                response_data = safe_parser(response)
                selected_categories = response_data.get("categories", [])

                user.dialogue_tree = DialogueTree(name="root", type="cat_selection", dialogue=msgs)

                if selected_categories:
                    for cat in selected_categories:
                        if cat in candidate_categories:
                            user.dialogue_tree.add_branch(DialogueTree(name=cat, type="leaf_selection"))
                            user.selected_leafs[cat] = []
                    if not user.selected_leafs:
                        self.augment_users.remove(user)
                        print(f"no proper categories selected for user {user.idx}, user removed")
                else:
                    self.augment_users.remove(user)
                return
            except Exception as e:
                if response:
                    print(f"{response}")
                print(f"Error in _select_categories: {e}")
                print("Retrying")

    async def _select_leafs(self, user:User):
        '''
        combine the cat and leaf selection for acceleration
        No restriction for the categories, balance the interacted cats(factual) and non-factual
        sample by probability
        '''
        # 
        selected_categories = list(user.selected_leafs.keys())
        dialogue = user.dialogue_tree.dialogue # the cat selection dialogue
        cat_trees = {}
        for cat in selected_categories:
            max_id, tree, id2name, leaf_paths = find_tree_by_category(cat, self.category_trees)
            cat_trees[cat]= {
                "max_id": max_id,
                "tree": tree,
                "id2name": id2name,
                "leaf_paths": leaf_paths
            }
        leaf_paths = "\n".join([f"{cat}: {info['leaf_paths']}" for cat, info in cat_trees.items()])
        id_maps = "\n".join([f"{cat}: {info['id2name']}" for cat, info in cat_trees.items()])
        leaf_combine_prompt = self.prompts["leaf_selection"].format(
            user_profile=user.profile_description, 
            candidate_categories=selected_categories,
            leaf_paths = leaf_paths,
            id_maps = id_maps,
            num_leaves = self.num_leaves
        )
        for _ in range(self.max_try):
            try:
                msgs = [LLMMessage(role='user', content=leaf_combine_prompt)]
                # print(f"length of the leaf combine prompt: {len(leaf_combine_prompt)}")
                
                response = await self.model.generate_response_async(messages=msgs)
                #dialogue.append(LLMMessage(role='assistant', content=response))
                response_data = safe_parser(response)
                assert isinstance(response_data, list), "response_data should be a list"
                if response_data:
                    for selection in response_data:
                        cat = selection["category"]
                        leaves = selection["leaves"]
                        if leaves is None: 
                            del user.selected_leafs[cat]
                            continue
                        if cat in selected_categories:
                            max_id = cat_trees[cat]["max_id"]
                            id2name = cat_trees[cat]["id2name"]

                            for leaf in leaves:
                                leaf_id = leaf["id"]
                                leaf_name = leaf["name"]
                                if 0< leaf_id <= max_id:
                                    if id2name[leaf_id] == leaf_name:
                                        user.selected_leafs[cat].append(leaf_id)
                                    else:
                                        #print(msgs)
                                        raise ValueError(f"Invalid response {response},leaf_id {leaf_id}: with unmatched leaf_name {leaf_name}, actual leaf name {id2name[leaf_id]}")
                                else:
                                    raise ValueError(f"Invalid response {response},leaf_id {leaf_id}: exceeds max_idx {max_id}")
                            if not user.selected_leafs[cat]: 
                                del user.selected_leafs[cat]
                                print(f"User{user.idx}, No valid leafs for category {cat}")
                    # print(f"user{user.idx},{user.selected_leafs} ")
                    if not user.selected_leafs: 
                        print(f"User {user.idx}, no valid leaves for all categories, user removed")
                        self.augment_users.remove(user)
                else:
                    self.augment_users.remove(user)
                return
            except Exception as e:
                print(e)
                print("retrying")
        if all(not v for v in user.selected_leafs.values()):
            print(f"User {user.idx}, no valid leaves for all categories, user removed")
            self.augment_users.remove(user)
    
    async def _select_tweets(self, user: User, top_k = 1):
        '''
        select all recommended tweets for the user, return a mid list
        '''
        selected_mids = []

        async def process_leaf_branch(leaf_branch:DialogueTree, content_df):
            leaf_name = leaf_branch.name
            context = user.dialogue_tree.find_dialogue_path(leaf_name).copy()
            tweet_list = "; ".join([f"{i+1}: {content}" for i, content in enumerate(content_df['content'])])
            tweet_selection_prompt = self.prompts["tweet_selection"].format(
                leaf_name=leaf_branch.name,
                tweet_list=tweet_list,
                top_k=top_k
            )
            msg = LLMMessage(role="user", content=tweet_selection_prompt)
            context.append(msg)
            for i in range(self.max_try):
                try:
                    response = await self.model.generate_response_async(messages=context)
                    response = safe_parser(response)
                    selected_indices = response["tweet_indices"]
                    selected_indices = [index - 1 for  index in selected_indices]
                    selected_mids = content_df.iloc[selected_indices, content_df.columns.get_loc('mid')].to_list()
                    leaf_branch.dialogue = [msg, LLMMessage(role="assistant", content=response)]
                    return selected_mids
                except Exception as e:
                    print(e)
                    print("Retrying...")
        tasks = []
        for cat_branch in user.dialogue_tree.sub_branches:
            for leaf_branch in cat_branch.sub_branches:
                cat = cat_branch.name
                leaf_id = int(leaf_branch.name[-1])
                leaf_name = leaf_branch.name
                content_df = self.root_content[(self.root_content['category'] == cat) & (self.root_content['leaf_id'] == leaf_id)]
                if content_df.empty:
                    continue
                task = process_leaf_branch(leaf_branch, content_df)
                tasks.append(task)
        # no need for a restriction?
        results = await asyncio.gather(*tasks)
        for result in results:
            if result:
                selected_mids.extend(result)
        result  = {"user_idx": user.idx, "selected_mids": selected_mids}
        print(result)
        return result
    
    
        
    async def _simplify_select_tweets(self, user:User, top_k:int=5):
        async def process_leaf(cat,content_df):
            tweet_list = "; ".join([f"{i+1}: {content}" for i, content in enumerate(content_df['content'])])
            tweet_selection_prompt = self.prompts["simplified_tweet_selection"].format(
                user_profile=user.profile_description,
                interaction_history=user.interaction_history,
                tweet_list=tweet_list,
                top_k=top_k
            )
            msg = LLMMessage(role="user", content=tweet_selection_prompt)
            
            for i in range(self.max_try):
                try:
                    # print(f"user{user.idx}: prompt length {len(tweet_selection_prompt)}")
                    #print(tweet_selection_prompt)
                    response = await self.model.generate_response_async(messages=[msg])
                    response = safe_parser(response)
                    selected_indices = response["tweet_indices"]
                    selected_indices = [index - 1 for  index in selected_indices]
                    selected_mids = content_df.iloc[selected_indices, content_df.columns.get_loc('mid')].to_list()
                    if not selected_mids: return None
                    return {"category": cat, "selected_mids": selected_mids}
                except Exception as e:
                    print(e)
                    print("Retrying...")
        tasks = []
        for cat in user.selected_leafs:
            leaf_ids = user.selected_leafs[cat]
            content_df = self.root_content[(self.root_content['category'] == cat) & (self.root_content['leaf_id'].isin(leaf_ids))]
            if content_df is None: continue
            content_df = sample_content_df(content_df)
            print(f"user {user.idx} content_df shape {content_df.shape}, cat{cat}, leaf_ids {leaf_ids}")
            task = process_leaf(cat,content_df)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        results = [res for res in results if res]
        if not results: return
        sampled_mids = weighted_index_sample(results, candidate_categories=user.category_history)
        result  = {"user_idx": user.idx, "selected_mids": set(sampled_mids)} # w/o replacement does not support the weighed sampling
        print(result)
        return result

    def _prob_select_tweets(self, user:User, method = "random", sample_ratio = None):
        '''
        after leaf selection, the user will preserve the attributes
        {"cat":[leaf_ids]}
        '''
        selected_leafs = user.selected_leafs
        data = []
        n_inters = user.inter_nums
        if sample_ratio is not None:
            sample_size = math.ceil(n_inters * sample_ratio)
        else:
            sample_size = self.num_augs

        if method == "random":
            for cat in selected_leafs:
                leaf_ids = selected_leafs[cat]
                content_df = self.root_content[(self.root_content['category'] == cat) & (self.root_content['leaf_id'].isin(leaf_ids))]
                mids = content_df['mid'].tolist()
                data.append({"category":cat, "selected_mids":mids})
            sampled_mids = weighted_index_sample(data, sample_size = sample_size,candidate_categories=user.category_history)
            result  = {"user_idx": user.idx, "selected_mids": set(sampled_mids)} # w/o replacement does not support the weighed sampling
            #print(result)
            return result

        elif method == "emb":
            recommendations = self.embedding_selector._user_recommend(user=user)
            for k, v in recommendations.items():
                data.append({"category":k, "selected_mids":v["mids"]})
            if data==[]:
                self.augment_users.remove(user)
                return None
            #sampled_mids = weighted_index_sample(data, candidate_categories=user.category_history)
            try:
                sampled_mids = weighted_index_sample_without_replacement(data, sample_size = sample_size,candidate_categories=user.category_history, weight_out=self.sample_weight)
                result  = {"user_idx": user.idx, "selected_mids": set(sampled_mids)} # w/o replacement does not support the weighed sampling
                #print(result)
                return result
            except Exception as e:
                print(e)
                print(f"Error in sampling, sample data : data{data}")
                return None


    async def recommend(self, user_idxs:list[int] = None, write = True):
        self._select_users(user_idxs = user_idxs)
        if not self.augment_users:
            return
        
        category_tasks = [self._select_categories(user) for user in self.augment_users]
        # await asyncio.gather(*category_tasks)
        await async_task_runner(category_tasks,max_concurrent=64, describe="Selecting Categories")
        
        leaf_tasks = []
        for user in self.augment_users:
            leaf_tasks.append(self._select_leafs(user))
        # await asyncio.gather(*leaf_tasks)
        await async_task_runner(leaf_tasks,max_concurrent=64, describe="Selecting leafs")

        if not self.prob:
            tweet_tasks = []
            for user in self.augment_users:
                tweet_tasks.append(self._simplify_select_tweets(user))
            # results = await asyncio.gather(*tweet_tasks)
            results = await async_task_runner(tweet_tasks, max_concurrent=32,describe="Selecting tweets")
        else: 
            results = []
            print(f"aug ratio {self.aug_ratio}")
            for user in self.augment_users:
                result = self._prob_select_tweets(user, method = self.prob_method, sample_ratio=self.aug_ratio)
                if result:
                    results.append(result)
        if write:
            os.makedirs(os.path.dirname(self.inter_aug_path), exist_ok=True)
            with open(self.inter_aug_path, "w", encoding="utf-8") as f:
                f.write("user_idx\tmid\n")
                for record in results:
                    if record:
                        user_idx = record["user_idx"]
                        for mid in record["selected_mids"]:
                            f.write(f"{user_idx}\t{mid}\n")
            print(f"Data written into {self.inter_aug_path}")
        else:
            inter_list = []
            for record in results:
                if record:
                    user_idx = record["user_idx"]
                    for mid in record["selected_mids"]:
                        inter_list.append([user_idx, mid])
            
            new_inter_ori_idx = np.array(inter_list)
            
            return new_inter_ori_idx



def weighted_index_sample(data, candidate_categories, sample_size=10, weight_in=1.0, weight_out=5.0):
    '''
    data: [{"category":.., "indices": [1, 2, 3]}]
    '''
    index_list = []
    raw_weights = []

    for item in data:
        cat = item["category"]
        weight = weight_in if cat in candidate_categories else weight_out
        for idx in item["selected_mids"]:
            index_list.append(idx)
            raw_weights.append(weight)

    total_weight = sum(raw_weights)
    normalized_probs = [w / total_weight for w in raw_weights]
    sampled_indices = random.choices(index_list, weights=normalized_probs, k=sample_size)

    return sampled_indices

def weighted_index_sample_without_replacement(data, candidate_categories, sample_size=3, weight_in=1.0, weight_out=5.0):
    index_list = []
    raw_weights = []

    for item in data:
        cat = item["category"]
        weight = weight_in if cat in candidate_categories else weight_out
        for idx in item["selected_mids"]: 
            index_list.append(idx)
            raw_weights.append(weight)

    indices_np = np.array(index_list)
    weights_np = np.array(raw_weights)

    sample_size = min(sample_size, len(indices_np))

    sampled_indices = np.random.choice(
        indices_np,
        size=sample_size,
        replace=False,
        p=weights_np/weights_np.sum()
    )
    return sampled_indices.tolist()

def sample_content_df(content_df, sample_size = 20):
    '''
    content_df: mid, content, category, leaf_id
    '''
    if content_df.shape[0] <= sample_size:
        return content_df
    else:
        # compute the number of samples per leaf based on the proportion
        leaf_counts = content_df['leaf_id'].value_counts(normalize=True)

        
        samples_per_leaf = (leaf_counts * sample_size).round().astype(int)


        sampled_df = content_df.groupby('leaf_id', group_keys=False).apply(
            lambda x: x.sample(n=samples_per_leaf[x.name], random_state=42)
        )

        return sampled_df

        
