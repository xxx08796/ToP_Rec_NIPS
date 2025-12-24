import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

class Evaluator:
    def __init__(self, num_items, config, train_user_pos_items = None):
        self.num_items = num_items
        self.top_k_list = config.top_k if isinstance(config.top_k, list) else [config.top_k]
        self.metrics = config.metrics
        self.config = config
        self.mask_train = True
        self.train_user_pos_items = train_user_pos_items
        

            
    def evaluate(self, eval_loader, model):
        model.eval()
        metric_results = {
            top_k: {m: 0.0 for m in self.metrics}
            for top_k in self.top_k_list
        }
        with torch.no_grad():
            user_emb_all, item_emb_all = model.get_embeddings()
            for batch in tqdm(eval_loader, desc="Evaluating"):
                users = batch["user"].to(self.config.device)
                user_emb = user_emb_all[users]  # [batch_size, dim]
                scores = torch.matmul(user_emb, item_emb_all.T)
                if self.mask_train:
                    train_mask = self._generate_mask(users)
                    if train_mask is not None:
                        scores = scores.masked_fill(train_mask, -1)
                max_top_k = max(self.top_k_list)
                top_k_indices = torch.argsort(scores, dim=1, descending=True)[:, :max_top_k].cpu().numpy()

                pos_items = batch["pos_items"].numpy()  # [batch_size, max_len]
                pos_mask = np.zeros((len(users), self.num_items), dtype=bool)
                for i, items in enumerate(pos_items):
                    valid_pos = items[items != -1]
                    pos_mask[i, valid_pos] = True

                for top_k in self.top_k_list:
                    current_top_k_indices = top_k_indices[:, :top_k]

                    if "recall" in self.metrics:
                        metric_results[top_k]["recall"] += self._recall(current_top_k_indices, pos_mask, top_k)
                    if "entropy" in self.metrics:
                        metric_results[top_k]["entropy"] += self._cate_entropy(current_top_k_indices, eval_loader.dataset.dataset.item_record)

        total_users = len(eval_loader.dataset)
        for top_k in metric_results:
            for metric in metric_results[top_k]:
                metric_results[top_k][metric] /= total_users
        return metric_results

    
    def _generate_mask(self, users):
        if not self.mask_train or self.train_user_pos_items is None:
            return None
        
        batch_size = users.shape[0]
        mask = torch.zeros((batch_size, self.num_items), 
                        dtype=torch.bool,
                        device=self.config.device)
        
        for i, user in enumerate(users.cpu().numpy()):
            if user in self.train_user_pos_items:
                mask[i, list(self.train_user_pos_items[user])] = True
        return mask

    def _recall(self, top_k_indices, pos_mask, top_k):
        batch_size = top_k_indices.shape[0]
        rows = np.arange(batch_size)[:, None].repeat(top_k, axis=1)
        top_k_mask = np.zeros((batch_size, self.num_items), dtype=bool)
        top_k_mask[rows, top_k_indices] = True
        hits = (top_k_mask & pos_mask).sum(axis=1)
        true_pos_counts = pos_mask.sum(axis=1)

        valid_mask = true_pos_counts > 0
        recalls = np.zeros(batch_size)
        recalls[valid_mask] = hits[valid_mask] / true_pos_counts[valid_mask]
        return recalls.sum()


    def _cate_entropy(self, top_k_indices, item_record):
        item_record['item'] = item_record['item'].astype(int)
        max_item_id = item_record['item'].max() + 1
        item_cate_array = np.full(max_item_id, -1, dtype=int)
        item_cate_array[item_record['item'].values] = item_record['category'].values
        all_categories = item_cate_array[top_k_indices]
        batch_size, k = all_categories.shape
        max_category = item_record['category'].max() + 1
        category_counts = np.zeros((batch_size, max_category), dtype=int)
        for i in range(k):
            np.add.at(category_counts, (np.arange(batch_size), all_categories[:, i]), 1)
        total_valid = category_counts.sum(axis=1, keepdims=True)
        prob = category_counts / total_valid
        epsilon = 1e-10
        entropies = -np.sum(prob * np.log(prob + epsilon), axis=1)
        return np.sum(entropies)