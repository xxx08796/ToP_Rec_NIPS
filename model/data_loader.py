import copy

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class InteractionDataset:
    def __init__(self, train_path, valid_path, test_path, item_path):
        self.train_records, self.valid_records, self.test_records, self.item_record = None, None, None, None
        train_df = pd.read_csv(train_path, names=['user', 'item'], sep='\t', skiprows=1)
        valid_df = pd.read_csv(valid_path, names=['user', 'item'], sep='\t', skiprows=1)
        test_df = pd.read_csv(test_path, names=['user', 'item'], sep='\t', skiprows=1)
        item_df = pd.read_csv(item_path, names=['item', 'content', 'keyword', 'category'], sep='\t', skiprows=1)

        self._create_mappings(train_df, valid_df, test_df, item_df)
        self._process_item(item_df)
        self._build_interaction_dict(train_df, valid_df, test_df)
        self._split_datasets(train_df, valid_df, test_df)

    def _create_mappings(self, train_df, valid_df, test_df, item_df):
        all_users = pd.concat([train_df['user'], valid_df['user'], test_df['user']]).unique()
        all_items = item_df["item"].unique()
        self.user2id = {u: idx for idx, u in enumerate(all_users)}
        self.item2id = {i: idx for idx, i in enumerate(all_items)}
        self.id2user = {idx: u for idx, u in enumerate(all_users)}
        self.id2item = {idx: i for idx, i in enumerate(all_items)}
        self.n_users = len(self.user2id)
        self.n_items = len(self.item2id)

    def _build_interaction_dict(self, train_df, valid_df, test_df):
        self.train_user_pos_items = defaultdict(set)
        for u, i in train_df.values:
            self.train_user_pos_items[self.user2id[u]].add(self.item2id[i])
        self.valid_user_pos_items = defaultdict(set)
        for u, i in valid_df.values:
            self.valid_user_pos_items[self.user2id[u]].add(self.item2id[i])
        self.test_user_pos_items = defaultdict(set)
        for u, i in test_df.values:
            self.test_user_pos_items[self.user2id[u]].add(self.item2id[i])

    def _split_datasets(self, train_df, valid_df, test_df):
        self.train_records = np.array([(self.user2id[u], self.item2id[i]) for u, i in train_df.values])
        self.valid_records = np.array([(self.user2id[u], self.item2id[i]) for u, i in valid_df.values])
        self.test_records = np.array([(self.user2id[u], self.item2id[i]) for u, i in test_df.values])

    def add_new_train_data(self, new_records):
        for user_id, item_id in new_records:
            if (user_id, item_id) not in existing_pairs:
                unique_new_records.append([user_id, item_id])

        if not unique_new_records:
            return

        unique_new_records = np.array(unique_new_records)

        self.train_records = np.concatenate([self.train_records, unique_new_records])

        for user_id, item_id in unique_new_records:
            self.train_user_pos_items[user_id].add(item_id)

    def _process_item(self, item_df):
        self.item_record = copy.deepcopy(item_df[['item', 'category']])
        self.item_record['item'] = self.item_record['item'].map(self.item2id)


class TrainDataset(Dataset):
    def __init__(self, dataset, n_negatives):
        self.dataset = dataset
        self.n_neg = n_negatives
        self.all_items = np.arange(dataset.n_items)
        self.user_pos_items = dataset.train_user_pos_items

    def __len__(self):
        return len(self.dataset.train_records)

    def __getitem__(self, idx):
        user, pos_item = self.dataset.train_records[idx]
        neg_items = self._negative_sampling(user)
        return {
            "user": torch.LongTensor([user]),
            "pos_item": torch.LongTensor([pos_item]),
            "neg_items": torch.LongTensor(neg_items)
        }

    def _negative_sampling(self, user):
        pos_set = self.user_pos_items[user]
        negs = []

        while len(negs) < self.n_neg:
            remaining = self.n_neg - len(negs)
            candidates = np.random.choice(
                self.all_items,
                size=min(int(remaining * 1.2), len(self.all_items)),
                replace=False
            )
            valid = [x for x in candidates if x not in pos_set]
            negs.extend(valid[:remaining])

        return negs[:self.n_neg]


class EvaluateDataset(Dataset):
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode
        self.records = getattr(dataset, f"{mode}_records")
        if mode == 'valid':
            self.user_pos_items = dataset.valid_user_pos_items
        elif mode == 'test':
            self.user_pos_items = dataset.test_user_pos_items
        else:
            raise ValueError("Mode must be 'valid' or 'test'")

    def __len__(self):
        return self.dataset.n_users

    def __getitem__(self, idx):
        pos_items = list(self.user_pos_items[idx])
        return {
            "user": torch.LongTensor([idx]),
            "pos_items": torch.LongTensor(pos_items)
        }


def collate_fn(batch):
    users = torch.cat([item["user"] for item in batch])
    pos_items = [item["pos_items"] for item in batch]
    max_len = max(len(items) for items in pos_items)
    padded_pos = torch.full((len(batch), max_len), -1, dtype=torch.long)
    for i, items in enumerate(pos_items):
        padded_pos[i, :len(items)] = items
    return {"user": users, "pos_items": padded_pos}


def create_dataloaders(dataset, batch_size, eval_size):
    train_loader = DataLoader(
        TrainDataset(dataset, 50),
        batch_size=batch_size,
        shuffle=True,
        num_workers=32
    )

    valid_loader = DataLoader(
        EvaluateDataset(dataset, "valid"),
        batch_size=eval_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    test_loader = DataLoader(
        EvaluateDataset(dataset, "test"),
        batch_size=eval_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    return train_loader, valid_loader, test_loader
