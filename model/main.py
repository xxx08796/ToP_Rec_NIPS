import torch
from torch.utils.data import DataLoader
from data_loader import InteractionDataset, TrainDataset, create_dataloaders
from model import LightGCN
from util import group_user, LLMAugmenter
from trainer import Trainer
from evaluator import Evaluator
import time
import argparse
import os
import asyncio
import random
import numpy as np
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(description="arguments for the recommendation system")
parser.add_argument("--dataset", type=str, default="./../dataset/twitter")

parser.add_argument("--model_name", default='Qwen2.5-32B-Instruct', type=str)
parser.add_argument("--user_size",default=None,type=int)

# Recommender
parser.add_argument("--prob",
                    default=True,
                    type=bool,
                    help="whether to use the prob tweeet selection instead of llm selection")
parser.add_argument("--prob_method",
                    default = "emb",
                    choices = ["random", "emb"],
                    help="the method for prob tweet selection")
parser.add_argument("--emb_model", 
                    default = "distiluse-base-multilingual-cased-v2",
                    help="model for the uer/tweet embedding")

parser.add_argument("--method",
                    default='kmeans',
                    type=str,
                    choices=['kmeans', 'random'],
                    help='the method for tree construction')
parser.add_argument("--num_cats", 
                    default = 3)
parser.add_argument("--num_leaves",
                    default = 3)
parser.add_argument("--num_augs", type=int, default=3,)

parser.add_argument("--aug_ratio",
                    type = float,
                    default = None)
parser.add_argument("--aug_times",
                        type=int,
                        default = 3)

# Backbone Model
parser.add_argument("--model",type=str, default='lightgcn', choices=['bpr', 'lightgcn'], help='model type')

parser.add_argument("--emb_dim", type=int, default=32,
                    help="Dimension of embedding vectors")
parser.add_argument("--n_negatives", type=int, default=50,
                    help="Number of negative samples per positive sample")

# Training
parser.add_argument("--batch_size", type=int, default=20480,
                    help="Training batch size")
parser.add_argument("--eval_size", type=int, default=4086,
                    help="Evaluation batch size")
parser.add_argument("--learning_rate", type=float, default=5e-3,
                    help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-5,
                    help="Weight decay for regularization")
parser.add_argument("--n_epochs", type=int, default=150,
                    help="Number of training epochs")
parser.add_argument("--aug_gap", type=int, default=200,
                    help="Augmentation interval for data enrichment")
parser.add_argument("--sample_weight", type=float, default=5.0,
                    help="Weight for sampling new categories")
parser.add_argument("--warm", type=int, default=10,)

parser.add_argument("--weight_path", default = "weights/best_model.pth", )

parser.add_argument("--seed", default = 42)

parser.add_argument("--n_layer", default=1,
                    type=int,
                    help="n_layers of gcn" )    
parser.add_argument("--grad_dim", type=int, default=512,help="gradient projectiondimension")

parser.add_argument("--device", type=str, 
                    default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Device for tensor computations (cuda/cpu)")

# Evaluation
parser.add_argument("--top_k", nargs='+', type=int, default=[10, 50, 100],
                    help="Top-k items for recommendation evaluation")
parser.add_argument("--metrics", type=str, default="recall,entropy",
                    help="Evaluation metrics (comma-separated)")
parser.add_argument("--mask_train", default=False,)
args = parser.parse_args()

args.train_path = f"{args.dataset}.train.inter"
args.valid_path = f"{args.dataset}.valid.inter"
args.test_path = f"{args.dataset}.test.inter"
args.item_path = f"{args.dataset}.item"
args.user_info_path = f"{args.dataset}.user"
args.interaction_history_path = args.train_path
args.metrics = args.metrics.split(",")
args.device = torch.device(args.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


async def main(weight_path):
    dataset = InteractionDataset(args.train_path, args.valid_path, args.test_path, args.item_path)
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, args.batch_size, args.eval_size)
    train_user_pos_items = copy.deepcopy(dataset.train_user_pos_items)
    model = LightGCN(dataset.n_users, dataset.n_items, args.emb_dim, dataset.train_records)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, optimizer, args.device)
    evaluator = Evaluator(dataset.n_items, args, train_user_pos_items=train_user_pos_items)
    group = group_user(dataset.n_users, n_group=args.n_group)
    best_avg_recall = 0.0
    k_step_influ = None
    augmenter = LLMAugmenter(args)
    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss, group_influ = trainer.train_epoch(train_loader, user_to_group=group)
        k_step_influ = group_influ + k_step_influ if k_step_influ is not None else group_influ
        if (args.aug_gap - (epoch + 1) % args.aug_gap) < args.k_step:
            k_step_influ = group_influ + k_step_influ if k_step_influ is not None else group_influ
        valid_metrics = evaluator.evaluate(valid_loader, model)

        if (epoch + 1) >= args.warm:
            if (epoch + 1) % args.aug_gap == 0:
                new_train_data = await augmenter.gen_inter(dataset, 5, group, k_step_influ)
                dataset.add_new_train_data(new_train_data)
                train_loader = DataLoader(TrainDataset(dataset, args.n_negatives), batch_size=args.batch_size,
                                          shuffle=True, num_workers=32)
                group = group_user(dataset.n_users, n_group=args.n_group)
                model.update_edge_index(new_train_data)
                k_step_influ = None

        current_avg_recall = np.mean([valid_metrics[top_k]["recall"] for top_k in args.top_k])

        if current_avg_recall > best_avg_recall:
            best_avg_recall = current_avg_recall
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(model.state_dict(), weight_path)
            print(f"New best model saved with average recall@{args.top_k}: {best_avg_recall:.4f}")

        log = f"Epoch {epoch + 1:02d} [{time.time() - start_time:.1f}s] "
        log += f"Train Loss: {train_loss:.4f} | \n"
        for top_k in args.top_k:
            log += f"[Top@{top_k}] "
            log += " | ".join([f"Valid {k.upper()}: {v:.4f}" for k, v in valid_metrics[top_k].items()])
            log += " | \n"
        print(log)

    load_state = torch.load(weight_path, weights_only=True)
    load_edge, load_weight = load_state.pop("edge_index"), load_state.pop("edge_weight")
    model.load_state_dict(load_state, strict=False)
    model.register_buffer("edge_index", load_edge)
    model.register_buffer("edge_weight", load_weight)
    test_metrics = evaluator.evaluate(test_loader, model)
    return test_metrics


async def run_exps():
    seed_everything(seed=args.seed)
    path = f"{args.weight_path}"
    test_metrics = await main(weight_path=path)
    for top_k in args.top_k:
        print(f"\n[Top@{top_k}] Test Results:")
        print(" | ".join([f"{k.upper()}: {v:.4f}" for k, v in test_metrics[top_k].items()]))


if __name__ == "__main__":
    asyncio.run(run_exps())
