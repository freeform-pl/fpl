if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_transformer_lowdim_policy import FlowTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.reward_conditioned_lowdim_runner import RewardConditionedLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainRewardConditionedFlowTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: FlowTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: FlowTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner (single instance, reused for all z-values)
        env_runner: RewardConditionedLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, RewardConditionedLowdimRunner)

        # Per-axis z-score targets: each entry is an array of length num_reward_dims
        # Configured via eval_z_positive / eval_z_negative (null = default ±1.5)
        z_pos = cfg.get('eval_z_positive', None)
        z_neg = cfg.get('eval_z_negative', None)
        # Infer num_reward_dims from scores.json
        import json
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        with open(cfg.scores_path, 'r') as f:
            scores_meta = json.load(f)
        num_reward_dims = len(scores_meta['reward_names'])
        env_runner.num_reward_dims = num_reward_dims
        print(f"Reward dims: {num_reward_dims} ({scores_meta['reward_names']})")

        # Build eval z-score targets: list of (label, array_of_length_K)
        z_zero = np.zeros(num_reward_dims, dtype=np.float32)
        if z_pos is not None:
            z_positive = np.array(z_pos, dtype=np.float32)
        else:
            z_positive = np.full(num_reward_dims, 1.5, dtype=np.float32)
        if z_neg is not None:
            z_negative = np.array(z_neg, dtype=np.float32)
        else:
            z_negative = np.full(num_reward_dims, -1.5, dtype=np.float32)

        eval_z_targets = [
            ('z_zero', z_zero),
            ('z_pos', z_positive),
            ('z_neg', z_negative),
        ]
        print(f"Eval z targets: zero={z_zero}, pos={z_positive}, neg={z_negative}")

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # Log z-score distributions of the training dataset
        reward_names = scores_meta['reward_names']
        rollout_z = np.array(scores_meta['rollout_scores_zscore'])
        demo_z = np.array(scores_meta['demo_scores_zscore'])
        all_z = np.concatenate([rollout_z, demo_z], axis=0)

        fig, axes = plt.subplots(1, num_reward_dims, figsize=(4 * num_reward_dims, 3), squeeze=False)
        for k in range(num_reward_dims):
            ax = axes[0, k]
            ax.hist(rollout_z[:, k], bins=30, alpha=0.6, label='rollouts', edgecolor='black')
            ax.hist(demo_z[:, k], bins=30, alpha=0.6, label='demos', edgecolor='black')
            ax.set_title(f'{reward_names[k]} z-scores')
            ax.set_xlabel('z-score')
            ax.set_ylabel('count')
            ax.legend(fontsize=8)
        fig.suptitle('Conditioning Z-Score Distributions (Training Data)')
        fig.tight_layout()
        wandb_run.log({'dataset/zscore_distributions': wandb.Image(fig)})
        plt.close(fig)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    metric_lines = []

                    for z_label, z_target in eval_z_targets:
                        # Swap target rewards on the single runner
                        env_runner.target_rewards = z_target.copy()

                        z_log = env_runner.run(policy)

                        if z_label == 'z_zero':
                            # Use z=0 as the main metrics for checkpointing
                            step_log.update(z_log)
                        for key, value in z_log.items():
                            step_log[f'{z_label}/{key}'] = value

                        for prefix in ['train/', 'test/']:
                            if prefix+'mean_score' in z_log:
                                metric_lines.append(
                                    f"  {z_label} {prefix:<7s} score={z_log[prefix+'mean_score']:.3f}  "
                                    f"success={z_log.get(prefix+'mean_success', 0):.3f}  "
                                    f"speed={z_log.get(prefix+'mean_speed_reward', 0):.3f}  "
                                    f"smoothness={z_log.get(prefix+'mean_smoothness', 0):.3f}"
                                )

                    if metric_lines:
                        print(f"\n[Epoch {self.epoch}] Eval metrics:")
                        for line in metric_lines:
                            print(line)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run flow sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRewardConditionedFlowTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
