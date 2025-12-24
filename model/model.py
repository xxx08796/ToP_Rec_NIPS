import torch.nn as nn
import torch
from torch_geometric.nn import LGConv
from torch_geometric.utils import degree

class RecommenderBase(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

    def forward(self, users, pos_items, neg_items):
        raise NotImplementedError

    def predict(self, users, items):
        u_emb = self.user_emb(users)
        i_emb = self.item_emb(items)
        return (u_emb * i_emb).sum(dim=-1)


class LightGCN(RecommenderBase):
    def __init__(self, n_users, n_items, emb_dim, train_record, num_layers=1):
        super().__init__(n_users, n_items, emb_dim)
        self.num_layers = num_layers
        self.n_users = n_users
        self.n_items = n_items
        self._process_train_record(train_record)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self._reset_parameters()

    def _process_train_record(self, train_record):
        if not isinstance(train_record, torch.Tensor):
            train_record = torch.from_numpy(train_record)
        users = train_record[:, 0]
        items = train_record[:, 1] + self.n_users
        src = torch.cat([users, items])
        dst = torch.cat([items, users])
        edge_index = torch.stack([src, dst], dim=0)
        edge_index = torch.unique(edge_index, dim=1)
        self._update_edges(edge_index)

    def _update_edges(self, edge_index):
        row, col = edge_index
        deg = degree(row, num_nodes=self.n_users + self.n_items, dtype=torch.float)
        norm = torch.pow(deg[row], -0.5) * torch.pow(deg[col], -0.5)
        device = self.user_emb.weight.device
        edge_index = edge_index.to(device)
        norm = norm.to(device)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', norm)

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.user_emb.weight)
        # nn.init.xavier_uniform_(self.item_emb.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embeddings(self):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embeddings = [all_emb]
        current_emb = all_emb
        for conv in self.convs:
            current_emb = conv(current_emb, self.edge_index, self.edge_weight)
            embeddings.append(current_emb)
        final_emb = torch.mean(torch.stack(embeddings), dim=0)
        user_emb, item_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    def forward(self, users, pos_items, neg_items=None):
        user_emb, item_emb = self.get_embeddings()
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        if neg_items is None:
            return (u_emb * pos_emb).sum(dim=-1)
        neg_emb = item_emb[neg_items]
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)
        return pos_scores, neg_scores

    def update_edge_index(self, new_train_record):
        if not isinstance(new_train_record, torch.Tensor):
            new_train_record = torch.from_numpy(new_train_record)
        users = new_train_record[:, 0]
        items = new_train_record[:, 1] + self.n_users
        new_src = torch.cat([users, items])
        new_dst = torch.cat([items, users])
        new_edges = torch.stack([new_src, new_dst], dim=0)
        device = self.edge_index.device
        new_edges = new_edges.to(device)
        combined_edges = torch.cat([self.edge_index, new_edges], dim=1)
        combined_edges = torch.unique(combined_edges, dim=1)
        self._update_edges(combined_edges)



