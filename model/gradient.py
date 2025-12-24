import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.func import functional_call, vmap, jacrev
from torch_scatter import scatter_mean, scatter


def get_user_grad(model, batch, mask_dict, user_to_group, params):
    device = next(model.parameters()).device
    num_group = user_to_group.max() + 1
    params = {n: p.detach().requires_grad_(True) for n, p in params.items()}
    user_ids = batch['user'].squeeze(1).to(device)  # [batch_size]
    pos_items = batch['pos_item'].squeeze(1).to(device)  # [batch_size]
    neg_items = batch['neg_items'].to(device)  # [batch_size, neg_num]

    def per_user_loss(params, all_users, all_pos, all_neg):
        batch_size, neg_num = all_neg.shape
        flat_users = all_users.unsqueeze(1).expand(-1, neg_num)  # [batch_size*neg_num]
        flat_pos = all_pos.unsqueeze(1).expand(-1, neg_num)  # [batch_size*neg_num]
        flat_neg = all_neg  # [batch_size*neg_num]

        def get_score(u, i):
            return functional_call(model, (params, {}), (u, i, None))

        pos_score = get_score(flat_users, flat_pos)  # [batch*neg]
        neg_score = get_score(flat_users, flat_neg)  # [batch*neg]
        raw_loss = -torch.log(torch.sigmoid(pos_score - neg_score))  # [total_pairs]
        raw_loss = raw_loss.sum(dim=1)  # [total_pairs]
        group_per_sample = user_to_group[all_users]  # [batch_size]
        cnt = scatter(torch.ones_like(raw_loss), group_per_sample, ) * neg_num + 1e-8
        return scatter(raw_loss, group_per_sample, ) / cnt

    batch_jacobian = jacrev(per_user_loss)(params, user_ids, pos_items, neg_items)
    features = torch.zeros(num_group, 0, device=device)
    for param_name in batch_jacobian:
        grad = batch_jacobian[param_name].detach()  # (num_users, *param_shape)
        indices = mask_dict.get(param_name, None)
        ndim = indices.shape[1]
        num_selected = indices.shape[0]
        group_idx = torch.arange(num_group, device=device)[:, None].expand(-1, num_selected).reshape(-1)
        param_idx = indices[None, :, :].expand(num_group, -1, -1).reshape(-1, ndim).unbind(dim=1)
        full_indices = (group_idx,) + tuple(param_idx)
        selected = grad[full_indices]
        selected = selected.view(num_group, num_selected)
        features = torch.cat([features, selected], dim=1)
    return features.detach()
