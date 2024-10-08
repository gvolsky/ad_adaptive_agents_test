import gc
import math
import os
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym.vector import SyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
from src.ad.model import Transformer
from src.dataset import SequenceDataset, generate_dataset, generate_probs
from src.envs import BernoulliBandits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")


@dataclass
class Config:
    # wandb params
    project: str = "ad_rel"
    group: str = "ad"
    name: str = "exp"
    run_number: int = 0
    # model params
    embedding_dim: int = 128
    n_filters: int = 64
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 128
    seq_len: int = 300
    stretch_factor: int = 4
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    # training params
    learning_rate: float = 1.1e-5
    warmup_ratio: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1.5e-4
    clip_grad: Optional[float] = 5.0
    batch_size: int = 32
    num_updates: int = 20_000
    log_interval: int = 500
    num_workers: int = 8
    label_smoothing: float = 0.001
    # evaluation params
    num_eval_envs: int = 32
    eval_interval: int = 500
    eval_steps: int = 300
    # general params
    train_seed: int = 0
    eval_seed: int = 100
    # data
    num_train_envs: int = 100_000
    num_arms: int = 10
    traj_name: str = 'game'
    num_iterations: int = 300
    rho: int = 0.5
    data_directory: str = os.path.join(os.path.dirname(__file__), "datafiles")


def cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def cosine_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    _decay_func = partial(
        cosine_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler

def get_rngs(train_seed, eval_seed):
    return np.random.default_rng(train_seed), np.random.default_rng(eval_seed)

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

@torch.no_grad()
def evaluating(vec_env, model, config: Config):
    cumulative_reward = np.zeros(config.num_eval_envs)
    model.eval()
    vec_env.reset(seed=config.eval_seed)

    actions = torch.zeros(
        (config.seq_len, config.num_eval_envs), 
        dtype=torch.long,
        device=DEVICE,
    )
    rewards = torch.zeros(
        (config.seq_len, config.num_eval_envs), 
        dtype=torch.long, 
        device=DEVICE
    )

    for step in trange(config.eval_steps, desc="Eval loop"):
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)

        pred = model(
            actions=actions[-step:].permute(1, 0),
            rewards=rewards[-step:].permute(1, 0)
        )[:, -1]

        pred_actions = pred.argmax(dim=-1).squeeze(-1)

        _, reward, _, _, _ = vec_env.step(pred_actions.cpu().numpy())
        cumulative_reward += reward

        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        actions[-1] = pred_actions
        rewards[-1] = torch.from_numpy(reward).type(torch.long).to(DEVICE)

    model.train()

    return cumulative_reward 

    
def train(config: Config):
    torch.manual_seed(config.train_seed)
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
        mode='offline',
    )

    train_rng, eval_rng = get_rngs(config.train_seed, config.eval_seed)
    get_probs = partial(generate_probs, num_arms=config.num_arms)

    train_probs = get_probs(train_rng, num_envs=config.num_train_envs, type='even') 
    odd_probs = get_probs(eval_rng, num_envs=config.num_eval_envs, type='odd')
    even_probs = get_probs(eval_rng, num_envs=config.num_eval_envs, type='even')
    odd_envs = SyncVectorEnv(
        [lambda prob=prob: BernoulliBandits(prob) for prob in odd_probs]
    )
    even_envs = SyncVectorEnv(
        [lambda prob=prob: BernoulliBandits(prob) for prob in even_probs]
    )

    train_data_path = generate_dataset(train_probs, config.train_seed, config)

    dataset = SequenceDataset(train_data_path, config.seq_len)
    dataloader = cycle(
        DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
            persistent_workers=True,
            drop_last=True
        )
    )

    model = Transformer(config=config).to(DEVICE)
    model.train()

    print(f'''
          Model Parameters: {sum(param.numel() for param in model.parameters())}
    ''')

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )

    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=1000,
        total_steps=config.num_updates,
    )

    bst_reward = 0

    for step in trange(1, config.num_updates + 1, desc="Training loop"):
        acts, rewards, _, _ = next(dataloader)
        acts, rewards = acts.to(torch.long).to(DEVICE), rewards.to(torch.long).to(DEVICE)
        preds = model(acts, rewards)
        loss = F.cross_entropy(
                input=preds.flatten(0, 1),
                target=acts.flatten(0, 1),
                label_smoothing=config.label_smoothing,
            )
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

        loss.backward() 
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % config.log_interval == 0:
            with torch.no_grad():
                pred_actions = torch.argmax(preds, dim=-1).flatten()
                true_actions = acts.flatten()
                accuracy = torch.sum(true_actions == pred_actions) / pred_actions.shape[0]

        if step % config.log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step,
                }
            )

        if step % config.eval_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

            eval_rewards = evaluating(odd_envs, model, config)
            wandb.log({f"odd_eval/reward_env_{i}": reward for i, reward in enumerate(eval_rewards)})
            even_rewards = evaluating(even_envs, model, config)
            wandb.log({f"even_eval/reward_env_{i}": reward for i, reward in enumerate(even_rewards)})
            mean_reward = eval_rewards.mean()
            if eval_rewards.mean() >= bst_reward:
                bst_reward = mean_reward
                checkpoints = os.path.join(config.data_directory, 'checkpoints')
                os.makedirs(checkpoints, exist_ok=True)
                torch.save(
                    model.state_dict(), os.path.join(checkpoints, f'model_{bst_reward}_{uuid.uuid4()}')
                )

    checkpoints = os.path.join(config.data_directory, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoints, f'model_{uuid.uuid4()}')) 

    # in-context observation
    uni_probs = get_probs(eval_rng, num_envs=config.num_eval_envs, type='uniform')
    even_probs = get_probs(train_rng, num_envs=config.num_eval_envs, type='even') 
    odd_max_mean = odd_probs.max(-1) * config.eval_steps
    uni_max_mean = uni_probs.max(-1) * config.eval_steps
    even_max_mean = even_probs.max(-1) * config.eval_steps

    uni_envs = SyncVectorEnv(
        [lambda prob=prob: BernoulliBandits(prob) for prob in uni_probs]
    )
    even_envs = SyncVectorEnv(
        [lambda prob=prob: BernoulliBandits(prob) for prob in even_probs]
    )
    
    even_rw = evaluating(even_envs, model, config)
    odd_rw = evaluating(odd_envs, model, config)
    uni_rw = evaluating(uni_envs, model, config)

    np.savez(
        os.path.join(config.data_directory, f'eval_{config.eval_seed}_{uuid.uuid4()}'),
        even_scale=even_max_mean,
        odd_scale=odd_max_mean,
        uni_scale=uni_max_mean,
        even_reward=even_rw,
        odd_reward=odd_rw,
        uni_reward=uni_rw
        )
    
    wandb.finish()

if __name__ == "__main__":
    for i in range(2):
        print(f'Runs: {i}')
        config = Config()
        config.train_seed = config.run_number = i
        config.eval_seed = 100 + i
        config.name = f'exp_{i}'
        train(config=config)