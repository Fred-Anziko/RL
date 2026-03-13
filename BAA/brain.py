import torch
import numpy as np
import random
from BAA.loss import DTTLossEngine
from BAA.curiosity import CuriosityEngine


class AgenticBrain:
    """
    A 'full-stack' self-regulating brain that manages both
    learning objectives and autonomous action selection.
    """

    def __init__(self, model, lr=1e-4, loss_engine=None, curiosity_engine=None,
                 optimizer=None, training_step=0, action_dim=None):
        self.model = model
        self.loss_engine = loss_engine if loss_engine else DTTLossEngine()
        self.curiosity_engine = curiosity_engine if curiosity_engine else CuriosityEngine(model)
        self.training_step = training_step
        self.action_dim = action_dim if action_dim is not None else model.policy_head.out_features

        # Optimizer covers both model weights AND the learned loss-balancing log_sigmas.
        # log_sigma parameters need a higher lr since they're much smaller scale.
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW([
                {"params": model.parameters(),            "lr": lr},
                {"params": self.loss_engine.parameters(), "lr": lr * 10},
            ])

    def act(self, state, rtg, explore=True, episode_num=1,
            start_epsilon=0.5, min_epsilon=0.05, decay_rate=0.9994):
        """
        Autonomous Action Selection with curiosity-driven exploration.

        state: Current observed state tensor
        rtg: Reward-to-Go tensor
        explore: Whether to apply exploration strategy
        episode_num: Current episode number for decay scheduling
        Returns: action tensor, decision trace, curiosity score
        """
        device = self.model.get_device()
        state = state.to(device)
        rtg = rtg.to(device)

        with torch.no_grad():
            output = self.model(state, rewards_to_go=rtg)
            action = output['action']
            trace = output['routing_trace']
            kl_div = output.get('kl_div')

        curiosity_score = self.curiosity_engine.compute_intrinsic_reward(trace, kl_div=kl_div)

        if explore:
            base_epsilon = max(min_epsilon, start_epsilon * (decay_rate ** (episode_num - 1)))
            epsilon = min(base_epsilon + curiosity_score.item() * 0.1, 1.0)

            if np.random.random() < epsilon:
                action = torch.tanh(torch.randn(1, 1, self.action_dim) * 2.0)

        return action, trace, curiosity_score

    def train_on_buffer(self, batch_size=32):
        """
        Core Self-Regulation: samples from the prioritized replay buffer and
        updates the model. Implements three improvements over the original:

        1. SSL: a second 'unlabeled' batch is sampled and its routing trace is
           used as the SSL consistency target, activating the third learning mode.

        2. Priority updates: per-sample action losses are computed after the
           backward pass and fed back to the buffer so surprising transitions
           are sampled more often.

        3. Auto-pruning: the routing trace from the training batch is fed into
           the RoutingTracker; if any node has become consistently confident,
           it is automatically frozen.

        Returns: training results dictionary, or None if buffer is too small.
        """
        buf = self.model.replay_buffer
        if len(buf) < batch_size:
            if np.random.random() < 0.05:
                print(f"  Training skipped: buffer has {len(buf)}/{batch_size} samples")
            return None

        self.training_step += 1
        device = self.model.get_device()

        # --- Primary (labeled) batch ---
        batch, indices = buf.sample(batch_size)
        if not batch:
            return None

        states  = torch.cat([x["states"].clone()  for x in batch], dim=0).to(device).detach().requires_grad_(True)
        actions = torch.cat([x["actions"].clone() for x in batch], dim=0).to(device).detach().clone()
        rtgs    = torch.cat([x["rtg"].clone()     for x in batch], dim=0).to(device).detach().clone()

        # --- SSL: secondary 'unlabeled' batch ---
        # We sample a second batch and run a no-grad forward pass to get its
        # routing trace. This trace is the SSL consistency target — the model
        # is trained to produce consistent routing between labeled and unlabeled
        # data, activating the path-consistency learning mode.
        unlabeled_routing_trace = None
        if len(buf) >= batch_size * 2:
            unlabeled_batch, _ = buf.sample(batch_size)
            if unlabeled_batch:
                ul_states = torch.cat([x["states"].clone() for x in unlabeled_batch], dim=0).to(device).detach()
                ul_rtgs   = torch.cat([x["rtg"].clone()    for x in unlabeled_batch], dim=0).to(device).detach()
                with torch.no_grad():
                    ul_output = self.model(ul_states, rewards_to_go=ul_rtgs)
                    unlabeled_routing_trace = ul_output["routing_trace"]

        self.optimizer.zero_grad()

        output = self.model(states, rewards_to_go=rtgs)

        results = self.loss_engine(
            model_output=output,
            target_actions=actions,
            target_states=states,
            unlabeled_routing_trace=unlabeled_routing_trace,
        )

        results["loss"].backward()
        self.optimizer.step()

        # --- Priority update ---
        # Compute per-sample action loss (unreduced) to measure how surprising
        # each transition was. Higher loss → higher priority → sampled more often.
        with torch.no_grad():
            per_sample_loss = torch.mean(
                (output["action"].detach() - actions) ** 2,
                dim=[1, 2]  # mean over seq_len and action_dim
            ).cpu().numpy()
        buf.update_priorities(indices, per_sample_loss)

        # --- Auto-pruning ---
        # Feed the routing trace into the tracker. Consistently decisive nodes
        # will be automatically frozen on the next check.
        self.model.record_routing(output["routing_trace"])
        pruned = self.model.check_auto_prune()

        results["active_mode"] = "Task-Mastery (Buffer Replay)"
        results["total_loss"]  = results["loss"].item()
        results["ssl_active"]  = unlabeled_routing_trace is not None
        results["auto_pruned"] = pruned
        return results

    def step_learning(self, experience_batch):
        """
        Live Signal Update: for immediate world-model adjustments.
        experience_batch: dict with keys "states", "actions", "rtg"
        Returns: training results dictionary
        """
        device = self.model.get_device()
        states  = experience_batch.get("states").to(device)
        actions = experience_batch.get("actions")
        if actions is not None:
            actions = actions.to(device)
        rtg = experience_batch.get("rtg")
        if rtg is not None:
            rtg = rtg.to(device)

        self.optimizer.zero_grad()
        output = self.model(states, rewards_to_go=rtg)

        results = self.loss_engine(
            model_output=output,
            target_actions=actions,
            target_states=states,
        )

        results["loss"].backward()
        self.optimizer.step()

        results["active_mode"] = "Signal-Direct (Live)"
        return results
