import torch
import numpy as np
from ToP.recommender import TweetRecommender


def group_user(n_user, n_group=20):
    shuffled = torch.randperm(n_user)
    group_size = n_user // n_group
    group_assignments = torch.zeros(n_user, dtype=torch.long)
    for group_id in range(n_group):
        start = group_id * group_size
        end = (group_id + 1) * group_size if group_id < n_group - 1 else n_user
        group_assignments[shuffled[start:end]] = group_id
    return group_assignments


class LLMAugmenter:
    def __init__(self, config):
        self.exist_user = set()
        self.config = config

    async def gen_inter(self, dataset, new_inter_per_user, group, score, topk_group=1):
        group_idx = torch.argsort(score, descending=True)[:topk_group].cpu().numpy()
        group = group.cpu().numpy()
        aug_user = np.where(np.isin(group, group_idx))[0]
        mask = ~np.isin(aug_user, list(self.exist_user))
        aug_user = aug_user[mask]
        aug_user_ori_idx = [dataset.id2user[idx] for idx in aug_user]
        recommender = TweetRecommender(self.config)
        new_inter_ori_idx =await recommender.recommend(aug_user_ori_idx, write=False)
        new_inter = np.array([(dataset.user2id[u], dataset.item2id[i]) for u, i in new_inter_ori_idx])
        self.exist_user.update(aug_user.tolist())
        return new_inter

