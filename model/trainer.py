import torch
from tqdm import tqdm

from gradient import get_user_grad

CUDA_LAUNCH_BLOCKING = 1


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader, user_to_group=None):
        self.model.train()
        last_check_point = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        influ = None
        total_loss = 0.0
        user_to_group = user_to_group.to(self.device)
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            users = batch["user"].squeeze().to(self.device)
            pos_items = batch["pos_item"].squeeze().to(self.device)
            neg_items = batch["neg_items"].to(self.device)
            pos_scores, neg_scores = self.model(users, pos_items, neg_items)

            differences = pos_scores.unsqueeze(1) - neg_scores
            loss = -torch.log(torch.sigmoid(differences)).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            delta_raw = {n: (p.detach() - last_check_point[n]) for n, p in self.model.named_parameters()}
            delta_abs = {n: (p.detach() - last_check_point[n]).abs_() for n, p in self.model.named_parameters()}
            mask_dict = {}
            for name, delta in delta_abs.items():
                k = min(512, delta.numel())
                topk_values, topk_indices = torch.topk(delta.flatten(), k)
                mask = torch.zeros_like(delta.flatten(), dtype=bool)
                mask[topk_indices] = True
                mask = mask.reshape(delta.shape)
                mask_dict[name] = mask
            idx_dict = precompute_indices(mask_dict)
            grad_feature = reduce_delta(delta_raw, idx_dict).detach()
            user_feature = get_user_grad(self.model, batch, idx_dict, user_to_group, last_check_point)
            sim = grad_feature @ user_feature.T
            influ = influ + sim if influ is not None else sim
            last_check_point = {n: p.detach().clone() for n, p in self.model.named_parameters()}

        influ = (influ - influ.min()) / (influ.max() - influ.min())
        return total_loss / len(train_loader), influ


def precompute_indices(mask_dict: dict) -> dict:
    indices_dict = {}
    for param_name, mask in mask_dict.items():
        nonzero_coords = torch.nonzero(mask, as_tuple=False)
        indices_dict[param_name] = nonzero_coords
    return indices_dict


def reduce_delta(delta: dict, indices_dict: dict) -> torch.Tensor:
    selected_values = []
    for param_name in delta:
        param_delta = delta[param_name]
        param_indices = indices_dict.get(param_name, None)
        values = param_delta[tuple(param_indices.t())]
        selected_values.append(values.flatten())

    return torch.cat(selected_values)
