import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union

from ..ToP.llm.dialogue_tree import User
import requests
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import torch


class EmbeddingSelector:
    def __init__(
            self,
            emb_dir: str = "./embedding_data",
            augmented_users: List[User] = None,
            root_content: pd.DataFrame = None
    ):
        self.root_content = root_content
        self.augmented_users = augmented_users
        self.user_emb = np.load(f"{emb_dir}/user_emb.npy", allow_pickle=True).item()
        self.item_emb = np.load(f"{emb_dir}/item_emb.npy", allow_pickle=True).item()
        self.uid2emb = {int(uid): emb for uid, emb in zip(self.user_emb['uid'], self.user_emb['emb'])}
        self.mid2emb = {int(mid): emb for mid, emb in zip(self.item_emb['mid'], self.item_emb['emb'])}


    def _user_recommend(
            self,
            user:User,
            top_k: int = 5
    ) -> Dict[str, Dict[str, Union[List[int], List[float]]]]:
        if not user.selected_leafs:
            raise ValueError("User has not selected any leafs")
        
        user_emb = self.uid2emb[user.idx]
        recommendations = {}
        for category, leaf_ids in user.selected_leafs.items():
            leaf_mask = (self.root_content['category'] == category) & \
                        (self.root_content['leaf_id'].isin(leaf_ids))
            candidate_df = self.root_content[leaf_mask].copy()
            candidate_mids = candidate_df['mid'].tolist()
            embeddings = [self.mid2emb[mid] for mid in candidate_mids]


            if len(candidate_df) > 0:
                
                similarities = cosine_similarity(
                    [user_emb],
                    np.stack(embeddings, axis=0)
                )[0]
                candidate_df['similarity'] = similarities
                if len(candidate_df) < top_k:
                    recommendations[category] = {
                    "mids": candidate_df['mid'].tolist(),
                    "scores": candidate_df['similarity'].tolist()
                }
                else:
                    top_tweets = candidate_df.sort_values('similarity', ascending=False).head(top_k)

                    recommendations[category] = {
                        "mids": top_tweets['mid'].tolist(),
                        "scores": top_tweets['similarity'].tolist()
                    }
        return recommendations
    
    def recommend(self):
        recommendations = {}
        for user in self.augmented_users:
            recommendation = self._user_recommend(user)
            print(f"user idx{user.idx}, embedding recommendations: {recommendation}")
            recommendations[user.idx] = recommendation
        return recommendations
