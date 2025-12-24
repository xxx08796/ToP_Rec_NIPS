import h5py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union

from .dialogue_tree import User

class EmbeddingStorage:
    
    def __init__(self, storage_dir: str):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.user_h5 = os.path.join(self.storage_dir, "user_embeddings.h5")
        self.tweet_h5 = os.path.join(self.storage_dir, "tweet_embeddings.h5")

        self.model = None
        self.embedding_dim = None

        # 内存索引 {id: h5_index}
        self.user_index = {}
        self.tweet_index = {}

        self._load_existing_index()

    def _init_model(self):
        if self.model is None:
            self.model = SentenceTransformer('HF_models/paraphrase-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _init_h5_file(self, h5_path):
        with h5py.File(h5_path, 'a') as f:
            if 'embeddings' not in f:
                f.create_dataset(
                    'embeddings',
                    shape=(0, self.embedding_dim),
                    maxshape=(None, self.embedding_dim),
                    dtype='float32'
                )

    def _load_existing_index(self):
        print('loading the user embeddings')
        if os.path.exists(self.user_h5):
            with h5py.File(self.user_h5, 'r') as f:
                if 'user_ids' in f and 'user_indices' in f:
                    user_ids = f['user_ids'][:]
                    user_indices = f['user_indices'][:]
                    for user_id, index in zip(user_ids, user_indices):
                        self.user_index[user_id] = index
        print("loading the tweet embeddings")
        if os.path.exists(self.tweet_h5):
            with h5py.File(self.tweet_h5, 'r') as f:
                if 'tweet_ids' in f and 'tweet_indices' in f:
                    tweet_ids = f['tweet_ids'][:]
                    tweet_indices = f['tweet_indices'][:]
                    for tweet_id, index in zip(tweet_ids, tweet_indices):
                        self.tweet_index[tweet_id] = index

    def add_users(self, users: List[User]) -> None:
        self._init_model()
        self._init_h5_file(self.user_h5)

        new_users = [user for user in users if user.idx not in self.user_index]
        if not new_users:
            return
        
        user_indices = [user.idx for user in new_users]
        # try
        descriptions = [user.profile_description for user in new_users]

        embeddings = self.model.encode(descriptions, convert_to_numpy=True)

        with h5py.File(self.user_h5, 'a') as f:
            current_size = f['embeddings'].shape[0]
            
            new_size = current_size + len(embeddings)
            print(f"Adding new users with shape size {len(embeddings)}")
            f['embeddings'].resize((new_size, self.embedding_dim))
            f['embeddings'][current_size:new_size] = embeddings


            for i, user_idx in enumerate(user_indices):
                self.user_index[user_idx] = current_size + i

            if 'user_ids' not in f:
                f.create_dataset('user_ids', shape=(new_size,), maxshape=(None,), dtype='int64')
                f.create_dataset('user_indices', shape=(new_size,), maxshape=(None,), dtype='int64')
            f['user_ids'][current_size:new_size] = user_indices
            f['user_indices'][current_size:new_size] = np.arange(current_size, new_size)

    def add_tweets(self, tweet_data: pd.DataFrame):
        self._init_model()
        self._init_h5_file(self.tweet_h5)

        new_tweet_data = tweet_data[~tweet_data['mid'].isin(self.tweet_index)]
        if new_tweet_data.empty:
            return

        contents = new_tweet_data['content'].tolist()
        embeddings = self.model.encode(contents, convert_to_numpy=True)


        with h5py.File(self.tweet_h5, 'a') as f:
            current_size = f['embeddings'].shape[0]
            new_size = current_size + len(embeddings)
            print(f"Adding new tweets with shape size {len(embeddings)}")
            f['embeddings'].resize((new_size, self.embedding_dim))
            f['embeddings'][current_size:new_size] = embeddings

            for i, mid in enumerate(new_tweet_data['mid']):
                self.tweet_index[mid] = current_size + i

            if 'tweet_ids' not in f:
                f.create_dataset('tweet_ids', shape=(new_size,), maxshape=(None,), dtype='int64')
                f.create_dataset('tweet_indices', shape=(new_size,), maxshape=(None,), dtype='int64')
            f['tweet_ids'][current_size:new_size] = new_tweet_data['mid'].tolist()
            f['tweet_indices'][current_size:new_size] = np.arange(current_size, new_size)

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        with h5py.File(self.user_h5, 'r') as f:
            return f['embeddings'][self.user_index[user_idx]]

    def get_tweet_embedding(self, mid: int) -> np.ndarray:
        with h5py.File(self.tweet_h5, 'r') as f:
            return f['embeddings'][self.tweet_index[mid]]

    def get_tweet_embeddings(self, mids: List[int]) -> np.ndarray:
        indices = [self.tweet_index[mid] for mid in mids]
        with h5py.File(self.tweet_h5, 'r') as f:
            return f['embeddings'][indices]

    def get_all_user_ids(self) -> List[int]:
        return list(self.user_index.keys())

    def get_all_tweet_ids(self) -> List[int]:
        return list(self.tweet_index.keys())


class EmbeddingSelector:
    '''
    initialize after the user selection.
    '''
    def __init__(
            self,
            storage_dir: str = "./embedding_data", 
            augmented_users: List[User] = None,
            root_content: pd.DataFrame = None # the classified root content
    ):
        self.storage = EmbeddingStorage(storage_dir)
        self.root_content = root_content
        self.augmented_users = augmented_users

        # 自动初始化数据
        if augmented_users is not None:
            existing_users = self.storage.get_all_user_ids()
            new_users = [u for u in augmented_users if u.idx not in existing_users]
            if new_users:
                self.storage.add_users(new_users)

        if root_content is not None:
            if not {'mid', 'content'}.issubset(root_content.columns):
                raise ValueError("initial_tweets should contain mid and content ")

            #self.tweet_metadata = root_content.drop(columns=['content']).set_index('mid').to_dict('index')

            existing_tweets = self.storage.get_all_tweet_ids()
            new_tweets = root_content[~root_content['mid'].isin(existing_tweets)]
            if not new_tweets.empty:
                self.storage.add_tweets(new_tweets)


    def _user_recommend(
            self,
            user:User,
            top_k: int = 5
    ) -> Dict[str, Dict[str, Union[List[int], List[float]]]]:
        """
        Generate recommendations based on user ID and preferences
        Args:
            user_id
            user_prefs: {"category": [leaf_id1, leaf_id2]}
            user_emb: (Optional) user embedding
            top_k: k tweets for each category

        Returns:
            recommendations {
                "category1": {
                    "mids": [mid1, mid2...],
                    "scores": [score1, score2...]
                },
                ...
            }
        """
        if not user.selected_leafs:
            raise ValueError("User has not selected any leafs")
        
        user_emb = self.storage.get_user_embedding(user.idx)
        

        recommendations = {}
        for category, leaf_ids in user.selected_leafs.items():
            # process the categorty
            leaf_mask = (self.root_content['category'] == category) & \
                        (self.root_content['leaf_id'].isin(leaf_ids))
            candidate_df = self.root_content[leaf_mask].copy()
            candidate_mids = candidate_df['mid'].tolist()
            tweet_embeddings = self.storage.get_tweet_embeddings(candidate_mids)
            candidate_df['embedding'] = list(tweet_embeddings)


            if len(candidate_df) > 0:
                
                similarities = cosine_similarity(
                    [user_emb],
                    np.stack(candidate_df['embedding'].values)
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
        #print(f"user_idx{user.idx}, embedding recommendations: {recommendations}")
        return recommendations
    
    def recommend(self):
        recommendations = {}
        for user in self.augmented_users:
            recommendation = self._user_recommend(user)
            print(f"user idx{user.idx}, embedding recommendations: {recommendation}")
            recommendations[user.idx] = recommendation
        return recommendations

# no need for the score for the final sample
    