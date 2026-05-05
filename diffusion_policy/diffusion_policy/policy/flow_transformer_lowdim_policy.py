from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class FlowTransformerLowdimPolicy(BaseLowdimPolicy):
    """
    Flow matching policy using the same transformer backbone as the diffusion policy.
    Instead of DDPM noise scheduling, uses the flow matching formulation:
      - Training: x_t = t * noise + (1 - t) * actions, predict velocity u_t = noise - actions
      - Sampling: ODE integration from noise (t=1) to data (t=0) with step x_{t+dt} = x_t + dt * v_t
    """
    def __init__(self,
            model: TransformerForDiffusion,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=10,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # beta distribution parameters for time sampling
            beta_a=1.5,
            beta_b=1.0,
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.num_inference_steps = num_inference_steps
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.kwargs = kwargs

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None):
        model = self.model

        # start from pure noise at t=1
        x_t = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        dt = -1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t = 1.0 + i * dt  # goes from 1.0 toward 0.0

            # apply conditioning
            x_t[condition_mask] = condition_data[condition_mask]

            # create timestep tensor for the model
            # the transformer expects integer-like timesteps; we scale [0,1] -> [0,999]
            t_input = torch.full(
                (x_t.shape[0],), t * 999,
                device=x_t.device, dtype=x_t.dtype)

            # predict velocity
            v_t = model(x_t, t_input, cond)

            # euler step: x_{t+dt} = x_t + dt * v_t
            x_t = x_t + dt * v_t

        # final conditioning enforcement
        x_t[condition_mask] = condition_data[condition_mask]

        return x_t

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # compute loss mask
        loss_mask = ~condition_mask

        bsz = trajectory.shape[0]

        # sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # sample time from Beta(a, b) distribution, shifted to [0.001, 1.0]
        time = torch.distributions.Beta(self.beta_a, self.beta_b).sample(
            (bsz,)).to(trajectory.device) * 0.999 + 0.001

        # expand time for broadcasting: (B,) -> (B, 1, 1)
        time_expanded = time[:, None, None]

        # flow matching interpolation: x_t = t * noise + (1 - t) * trajectory
        x_t = time_expanded * noise + (1 - time_expanded) * trajectory

        # target velocity: u_t = noise - trajectory
        u_t = noise - trajectory

        # apply conditioning
        x_t[condition_mask] = trajectory[condition_mask]

        # predict velocity field
        # scale time to match the model's expected timestep range [0, 999]
        t_input = time * 999
        v_t = self.model(x_t, t_input, cond)

        # flow matching loss
        loss = F.mse_loss(v_t, u_t, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
