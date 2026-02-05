import torch
import torch.nn.functional as F

class DTTLossEngine(torch.nn.Module):
    """ Dynamic Loss Engine for AgenticDTT.
    Inputs:
        - ent_weight: Weight for decision tree entropy loss
        - balance_weight: Weight for decision tree balance loss
        - recon_weight: Weight for reconstruction loss
        - consistency_weight: Weight for SSL consistency loss
    Outputs:
        - metrics: Dictionary of computed loss components
    """
    def __init__(self, ent_weight=0.01, balance_weight=0.05, recon_weight=0.1, consistency_weight=0.1, kl_weight=0.001):
        super().__init__()
        self.ent_weight = ent_weight
        self.balance_weight = balance_weight
        self.recon_weight = recon_weight
        self.consistency_weight = consistency_weight
        self.kl_weight = kl_weight

    def forward(self, model_output, target_actions=None, target_states=None, unlabeled_routing_trace=None):
        """
        Dynamically calculates the loss based on available environmental signals.
        Inputs:
            - model_output: Dictionary containing model predictions and routing trace
            - target_actions: Ground truth actions for supervised loss (optional)
            - target_states: Ground truth states for reconstruction loss (optional)
            - unlabeled_routing_trace: Routing trace from unlabeled data for SSL loss (optional)
        Outputs:
            - metrics: Dictionary of computed loss components
        """
        device = model_output['action'].device
        routing_trace = model_output.get("routing_trace", [])
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        # 1. Action Prediction Loss (RL / SL)
        if target_actions is not None:
            target_actions = target_actions.to(device)
            model_action = model_output['action']
            # target_actions shape: [batch, seq_len, 1] → action index
            # model_action shape: [batch, seq_len, action_dim] → action logits
            
            if target_actions.shape[-1] == 1 and model_action.shape[-1] > 1:
                # Discrete action case: convert to one-hot and use cross-entropy
                batch_size, seq_len = target_actions.shape[0], target_actions.shape[1]
                action_dim = model_action.shape[-1]
                
                # Flatten for loss computation (use contiguous().view() to avoid in-place issues)
                target_flat = target_actions.squeeze(-1).long().contiguous().view(-1)  # [batch*seq_len]
                logits_flat = model_action.contiguous().view(-1, action_dim)  # [batch*seq_len, action_dim]
                
                # Use cross-entropy for discrete actions
                l_action = F.cross_entropy(logits_flat, target_flat)
            else:
                # Continuous action case: use MSE
                l_action = F.mse_loss(model_action, target_actions)
            
            total_loss += l_action
            metrics["action_err"] = l_action.item()
        else:
            metrics["action_err"] = 0.0

        # 2. Unsupervised Reconstruction Loss (UL / World Model)
        if target_states is not None:
            l_recon = F.mse_loss(model_output['state_pred'], target_states.to(device))
            total_loss += self.recon_weight * l_recon
            metrics["recon_err"] = l_recon.item()
        else:
            metrics["recon_err"] = 0.0

        # 3. SSL Consistency Loss (Path-consistency)
        l_consistency = torch.tensor(0.0, device=device)
        if unlabeled_routing_trace is not None:
            for lab_layer, unlab_layer in zip(routing_trace, unlabeled_routing_trace):
                for layer_name in lab_layer:
                    lab_tree = lab_layer[layer_name]
                    unlab_tree = unlab_layer[layer_name]
                    for node_name in lab_tree:
                        p_lab = lab_tree[node_name]['go_right_prob']
                        p_unlab = unlab_tree[node_name]['go_right_prob']
                        # CRITICAL FIX (Issue 13): Detach unlabeled path, not labeled
                        # Gradients should flow through labeled data to update model
                        l_consistency += F.mse_loss(p_lab, p_unlab.detach())
            total_loss += self.consistency_weight * l_consistency
        metrics["ssl_consistency"] = l_consistency.item()

        # 4. Decision Tree Structural Losses
        l_entropy = torch.tensor(0.0, device=device)
        l_balance = torch.tensor(0.0, device=device)
        
        node_count = 0
        for layer_data in routing_trace:
            for layer_name, tree_dec in layer_data.items():
                for node_name, probs in tree_dec.items():
                    # High numerical stability clamp
                    p = probs['go_right_prob'].clamp(1e-7, 1-1e-7)
                    l_entropy += -torch.mean(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                    l_balance += (p.mean() - 0.5) ** 2
                    node_count += 1

        if node_count > 0:
            l_entropy /= node_count
            l_balance /= node_count
            total_loss += self.ent_weight * l_entropy
            total_loss += self.balance_weight * l_balance
            
        # 5. Bayesian KL Divergence Loss (Epistemic Uncertainty)
        kl_div = model_output.get("kl_div", torch.tensor(0.0, device=device))
        
        # Scale KL by batch size to keep it balanced with other average-based losses
        batch_size = model_output['action'].shape[0]
        seq_len = model_output['action'].shape[1]
        kl_scaled = kl_div / (batch_size * seq_len)
        
        total_loss += self.kl_weight * kl_scaled
        metrics["kl_div"] = kl_scaled.item()

        metrics["loss"] = total_loss
        metrics["tree_confusion"] = l_entropy.item()
        metrics["tree_imbalance"] = l_balance.item()

        return metrics