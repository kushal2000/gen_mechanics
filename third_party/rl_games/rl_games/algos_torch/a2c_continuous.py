import os
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch
import torch.distributed as dist 


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        
        if self.intr_reward_coef_embd is not None and not (self.expl_type.startswith('mixed_expl') and 'disjoint' in self.expl_type):
            input_shape = (self.obs_shape[0] + self.intr_reward_coef_embd.shape[1],)
        else:
            input_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : input_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'type' : 'simple' if 'learn_param' not in self.expl_type else 'extra_param',
        }
        
        if self.expl_type.startswith('mixed_expl'):
            build_config['coef_ids'] = self.intr_reward_coef_embd[::self.intr_coef_block_size,0]
            build_config['coef_id_idx'] = self.obs_shape[0]
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        # Gradient accumulation inside each PPO minibatch: forward/backward over this many contiguous chunks,
        # each chunk's (mean) loss weighted by its share of the minibatch, then ONE clip + optimizer step -- the
        # same update as the whole minibatch at once, with ~1/k of the activation memory. 1 = the original path.
        self.grad_accum_chunks = int(self.config.get('grad_accum_chunks', 1))
        if self.grad_accum_chunks > 1:
            print(f"[a2c_continuous] gradient accumulation: {self.grad_accum_chunks} chunks per minibatch of "
                  f"{self.minibatch_size}", flush=True)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            if self.intr_reward_coef_embd is not None and not (self.expl_type.startswith('mixed_expl') and 'disjoint' in self.expl_type):
                value_input_shape = (self.state_shape[0] + self.intr_reward_coef_embd.shape[1],)
            else:
                value_input_shape = self.state_shape
            cv_config = {
                'state_shape' : value_input_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done,
                'type': 'simple' if 'learn_param' not in self.expl_type else 'extra_param',
                # central_value_config has no mixed_precision key, so without
                # this the critic silently trained in fp32 while the actor used
                # fp16 -- a ~4x cost difference on the same stack.
                'mixed_precision' : self.mixed_precision,
            }
            if self.expl_type.startswith('mixed_expl'):
                cv_config['coef_ids'] = self.intr_reward_coef_embd[::self.intr_coef_block_size,0]
                cv_config['coef_id_idx'] = self.state_shape[0]
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn, override_state=None):
        state = {self.global_rank: self.get_full_state_weights()}
        if override_state is not None:
            state = override_state
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        if self.global_rank in checkpoint:
            weights = checkpoint[self.global_rank]
        elif 0 in checkpoint:
            # Single-GPU checkpoint: all ranks load from rank 0
            weights = checkpoint[0]
        else:
            weights = checkpoint
        self.set_full_state_weights(weights, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        if self.grad_accum_chunks > 1:
            return self._calc_gradients_accum(input_dict)
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)
            
            if self.expl_type.startswith('mixed_expl') and self.config.get('expl_reward_type') == 'entropy':
                ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
                ec_indices = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            elif self.expl_type.startswith('simple') and self.config.get('expl_reward_type') == 'entropy':
                entropy_coef = self.intr_reward_coef
            else:
                entropy_coef = self.entropy_coef
            
            if os.getenv('LOG_OFF_POLICY_GRADS'):
                loss_arr = a_loss.unsqueeze(1) + 0.5 * c_loss * self.critic_coef - (entropy_coef*entropy).unsqueeze(1) + b_loss.unsqueeze(1) * self.bounds_loss_coef
                off_policy_loss = torch.masked_select(loss_arr, input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                on_policy_loss = torch.masked_select(loss_arr, ~input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                grads_off = self.get_grads(off_policy_loss)
                grads_on = self.get_grads(on_policy_loss)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , (entropy_coef*entropy).unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy_loss, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        all_grads = self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0) 

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        if os.getenv('LOG_OFF_POLICY_GRADS'):
            contrib_off = torch.masked_select(contrib, input_dict['off_policy_mask'])
            contrib_on = torch.masked_select(contrib, ~input_dict['off_policy_mask'])
            contrib_off = torch.nan_to_num(contrib_off.mean())
            contrib_on = torch.nan_to_num(contrib_on.mean())
        
            extras = {
                "off_policy_contrib" : contrib_off.item(),
                "on_policy_contrib" : contrib_on.item(),
                "off_policy_grads" : grads_off.detach().cpu(),
                "on_policy_grads" : grads_on.detach().cpu(),
            }
        else:
            # These used to be `all_grads.detach().cpu()` and a same-sized
            # `torch.zeros_like(all_grads).cpu()` -- a full flattened-gradient
            # device-to-host copy TWICE per minibatch, the second of a tensor
            # already known to be zero, on the pageable path, each one a hard
            # host sync. Nothing consumes them unless LOG_OFF_POLICY_GRADS is
            # set (the branch above), and the consumer in a2c_common is now
            # guarded to match. Keeping them cost ~1.7 ms per minibatch and
            # stacked 56 x n_params floats of host memory every epoch.
            extras = {
                "on_policy_contrib" : contrib.mean().item(),
                "off_policy_contrib" : 0,
                "on_policy_grads" : None,
                "off_policy_grads" : None,
            }     
        if self.expl_type.startswith('mixed_expl'):
            bl_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
            bl_idxs = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item() for i in range(self.num_actors // self.intr_coef_block_size)]
        self.train_result = (a_loss, c_loss, torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0], \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, extras)

    def _entropy_coef_for(self, obs_batch):
        """calc_gradients' per-sample entropy coefficient (SAPG: each block's own), for a slice of the batch."""
        if self.expl_type.startswith('mixed_expl') and self.config.get('expl_reward_type') == 'entropy':
            ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
            ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
            ec_indices = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
            return ec_candidates[ec_indices]
        if self.expl_type.startswith('simple') and self.config.get('expl_reward_type') == 'entropy':
            return self.intr_reward_coef
        return self.entropy_coef

    def _calc_gradients_accum(self, input_dict):
        """calc_gradients over grad_accum_chunks contiguous chunks of the minibatch, one optimizer step.

        Every loss term in calc_gradients is a mean over the minibatch, so the minibatch loss is the
        size-weighted sum of the chunks' mean losses and the accumulated gradient is the same (to rounding;
        checked to 1e-7 in experiments/08oct_multi_embodiment_evolution/analysis/toy_transformer_memory.py).
        KL, the clip fraction, diagnostics and per-block entropies are computed on the concatenated outputs,
        as calc_gradients does. One difference: a network that updates input statistics in its forward pass sees
        k smaller updates instead of one -- the combined moments are the same, the batch is normalised with
        statistics that already include its earlier chunks.
        """
        if self.is_rnn or os.getenv('LOG_OFF_POLICY_GRADS'):
            raise NotImplementedError("grad_accum_chunks > 1: RNN and LOG_OFF_POLICY_GRADS are not supported")
        keys = ('old_values', 'old_logp_actions', 'advantages', 'mu', 'sigma', 'returns', 'actions', 'obs')
        n = input_dict['actions'].shape[0]
        bounds = torch.linspace(0, n, self.grad_accum_chunks + 1).round().long().tolist()
        curr_e_clip, lr_mul = self.e_clip, 1.0
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        a_sum = c_sum = e_sum = b_sum = 0.0
        mus, sigmas, neglogps, entropies = [], [], [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo:
                continue
            sub = {k: input_dict[k][lo:hi] for k in keys}
            obs_batch = self._preproc_obs(sub['obs'])
            w = (hi - lo) / n
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                res_dict = self.model({'is_train': True, 'prev_actions': sub['actions'], 'obs': obs_batch})
                action_log_probs, values = res_dict['prev_neglogp'], res_dict['values']
                entropy, mu, sigma = res_dict['entropy'], res_dict['mus'], res_dict['sigmas']
                a_loss = self.actor_loss_func(sub['old_logp_actions'], action_log_probs, sub['advantages'],
                                              self.ppo, curr_e_clip)
                if self.has_value_loss:
                    c_loss = common_losses.critic_loss(self.model, sub['old_values'], values, curr_e_clip,
                                                       sub['returns'], self.clip_value)
                else:
                    c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
                if self.bound_loss_type == 'regularisation':
                    b_loss = self.reg_loss(mu)
                elif self.bound_loss_type == 'bound':
                    b_loss = self.bound_loss(mu)
                else:
                    b_loss = torch.zeros(len(mu), device=self.ppo_device)
                entropy_coef = self._entropy_coef_for(obs_batch)
                losses, _ = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, (entropy_coef*entropy).unsqueeze(1),
                                                   b_loss.unsqueeze(1)], None)
                a_l, c_l, e_l, b_l = losses
                loss = (a_l + 0.5 * c_l * self.critic_coef - e_l + b_l * self.bounds_loss_coef) * w
            self.scaler.scale(loss).backward()
            a_sum, c_sum, b_sum = a_sum + a_l.detach() * w, c_sum + c_l.detach() * w, b_sum + b_l.detach() * w
            e_sum = e_sum + entropy.detach().float().mean() * w
            mus.append(mu.detach()); sigmas.append(sigma.detach())
            neglogps.append(action_log_probs.detach()); entropies.append(entropy.detach())
        self.trancate_gradients_and_step()

        mu, sigma = torch.cat(mus), torch.cat(sigmas)
        action_log_probs, entropy = torch.cat(neglogps), torch.cat(entropies)
        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(mu, sigma, input_dict['mu'], input_dict['sigma'], True)
        self.diagnostics.mini_batch(self, {'values': input_dict['old_values'], 'returns': input_dict['returns'],
                                           'new_neglogp': action_log_probs,
                                           'old_neglogp': input_dict['old_logp_actions'], 'masks': None},
                                    curr_e_clip, 0)
        ratio = torch.exp(input_dict['old_logp_actions'] - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()
        extras = {"on_policy_contrib": contrib.mean().item(), "off_policy_contrib": 0,
                  "on_policy_grads": None, "off_policy_grads": None}
        if self.expl_type.startswith('mixed_expl'):
            obs_full = self._preproc_obs(input_dict['obs'])
            bl_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
            bl_idxs = torch.argmax((obs_full[:,-self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [torch.nan_to_num(entropy[bl_idxs == i].mean()).item()
                                   for i in range(self.num_actors // self.intr_coef_block_size)]
        self.train_result = (a_sum, c_sum, e_sum, kl_dist, self.last_lr, lr_mul, mu, sigma, b_sum, extras)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    def get_grads(self, loss, retain_graph=True):
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None

        loss.backward(retain_graph=retain_graph)

        all_grads_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))

        all_grads = torch.cat(all_grads_list)
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        
        return all_grads

