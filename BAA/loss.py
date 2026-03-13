import torch
import torch.nn as nn
import torch.nn.functional as F


class DTTLossEngine(nn.Module):
    """
    Multi-task loss engine with learned uncertainty weighting (Kendall et al. 2018).

    Instead of hand-tuned static weights, each loss component has a learnable
    log-variance parameter (log_sigma). The effective weight for component i is:

        w_i = 1 / (2 * exp(log_sigma_i)^2)
        regularizer_i = log_sigma_i

    When a loss is noisy or large, the model learns to increase sigma_i,
    automatically reducing that component's contribution. The log(sigma) term
    prevents the model from trivially setting all sigmas to infinity.

    Components:
        0: action_loss     — task prediction (RL / SL)
        1: recon_loss      — world-model reconstruction (UL)
        2: ssl_loss        — routing consistency (SSL)
        3: tree_entropy    — structural sharpness of decisions
        4: tree_balance    — prevents tree collapse to one branch
        5: kl_loss         — Bayesian epistemic regularization
    """

    NUM_TASKS = 6

    def __init__(self):
        super().__init__()
        # Learnable log(sigma) for each loss component, initialized to 0 (sigma=1)
        self.log_sigma = nn.Parameter(torch.zeros(self.NUM_TASKS))

    def _weighted(self, loss_value, task_idx):
        """
        Apply uncertainty weighting to a scalar loss value.
        Returns: weighted_loss (includes regularizer), effective weight
        """
        sigma_sq = torch.exp(self.log_sigma[task_idx]) ** 2
        weighted = loss_value / (2.0 * sigma_sq) + self.log_sigma[task_idx]
        weight = (1.0 / (2.0 * sigma_sq)).detach()
        return weighted, weight.item()

    def forward(self, model_output, target_actions=None, target_states=None, unlabeled_routing_trace=None):
        """
        Compute the multi-task loss with automatic weighting.

        Inputs:
            - model_output: dict from AgenticDTT forward pass
            - target_actions: ground-truth actions (optional, for RL/SL)
            - target_states: ground-truth next states (optional, for UL)
            - unlabeled_routing_trace: routing trace from unlabeled data (optional, for SSL)

        Outputs:
            - metrics: dict with total loss, per-component values, and effective weights
        """
        device = model_output['action'].device
        routing_trace = model_output.get("routing_trace", [])
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}
        weights = {}

        # --- 1. Action Prediction Loss (RL / SL) ---
        if target_actions is not None:
            target_actions = target_actions.to(device)
            model_action = model_output['action']

            if target_actions.shape[-1] == 1 and model_action.shape[-1] > 1:
                batch_size = target_actions.shape[0]
                action_dim = model_action.shape[-1]
                target_flat = target_actions.squeeze(-1).long().contiguous().view(-1)
                logits_flat = model_action.contiguous().view(-1, action_dim)
                l_action = F.cross_entropy(logits_flat, target_flat)
            else:
                l_action = F.mse_loss(model_action, target_actions)

            w_action, eff_w = self._weighted(l_action, 0)
            total_loss = total_loss + w_action
            metrics["action_err"] = l_action.item()
            weights["w_action"] = eff_w
        else:
            metrics["action_err"] = 0.0
            weights["w_action"] = 0.0

        # --- 2. Unsupervised Reconstruction Loss (UL / World Model) ---
        if target_states is not None:
            l_recon = F.mse_loss(model_output['state_pred'], target_states.to(device))
            w_recon, eff_w = self._weighted(l_recon, 1)
            total_loss = total_loss + w_recon
            metrics["recon_err"] = l_recon.item()
            weights["w_recon"] = eff_w
        else:
            metrics["recon_err"] = 0.0
            weights["w_recon"] = 0.0

        # --- 3. SSL Consistency Loss (Path-consistency) ---
        l_consistency = torch.tensor(0.0, device=device)
        if unlabeled_routing_trace is not None:
            for lab_layer, unlab_layer in zip(routing_trace, unlabeled_routing_trace):
                for layer_name in lab_layer:
                    lab_tree = lab_layer[layer_name]
                    unlab_tree = unlab_layer[layer_name]
                    for node_name in lab_tree:
                        p_lab = lab_tree[node_name]['go_right_prob']
                        p_unlab = unlab_tree[node_name]['go_right_prob']
                        l_consistency = l_consistency + F.mse_loss(p_lab, p_unlab.detach())

            w_ssl, eff_w = self._weighted(l_consistency, 2)
            total_loss = total_loss + w_ssl
            weights["w_ssl"] = eff_w
        else:
            weights["w_ssl"] = 0.0
        metrics["ssl_consistency"] = l_consistency.item()

        # --- 4 & 5. Decision Tree Structural Losses ---
        l_entropy = torch.tensor(0.0, device=device)
        l_balance = torch.tensor(0.0, device=device)
        node_count = 0

        for layer_data in routing_trace:
            for layer_name, tree_dec in layer_data.items():
                for node_name, probs in tree_dec.items():
                    p = probs['go_right_prob'].clamp(1e-7, 1 - 1e-7)
                    l_entropy = l_entropy + (-torch.mean(p * torch.log(p) + (1 - p) * torch.log(1 - p)))
                    l_balance = l_balance + (p.mean() - 0.5) ** 2
                    node_count += 1

        if node_count > 0:
            l_entropy = l_entropy / node_count
            l_balance = l_balance / node_count

            w_ent, eff_w_ent = self._weighted(l_entropy, 3)
            w_bal, eff_w_bal = self._weighted(l_balance, 4)
            total_loss = total_loss + w_ent + w_bal
            weights["w_entropy"] = eff_w_ent
            weights["w_balance"] = eff_w_bal
        else:
            weights["w_entropy"] = 0.0
            weights["w_balance"] = 0.0

        metrics["tree_confusion"] = l_entropy.item()
        metrics["tree_imbalance"] = l_balance.item()

        # --- 6. Bayesian KL Divergence Loss ---
        kl_div = model_output.get("kl_div", torch.tensor(0.0, device=device))
        batch_size = model_output['action'].shape[0]
        seq_len = model_output['action'].shape[1]
        kl_scaled = kl_div / (batch_size * seq_len)

        w_kl, eff_w = self._weighted(kl_scaled, 5)
        total_loss = total_loss + w_kl
        metrics["kl_div"] = kl_scaled.item()
        weights["w_kl"] = eff_w

        # Expose effective weights and learned sigmas for monitoring
        metrics["weights"] = weights
        metrics["log_sigmas"] = {
            "action": self.log_sigma[0].item(),
            "recon": self.log_sigma[1].item(),
            "ssl": self.log_sigma[2].item(),
            "entropy": self.log_sigma[3].item(),
            "balance": self.log_sigma[4].item(),
            "kl": self.log_sigma[5].item(),
        }
        metrics["loss"] = total_loss
        return metrics
