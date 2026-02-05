import torch

class CuriosityEngine:
    """
    Computes intrinsic curiosity rewards based on the uncertainty of the
    Differentiable Decision Tree's routing decisions.
    inputs:
        - model: The agent model containing the DDT
        - alpha: Scaling factor for curiosity reward
        - eta: Learning rate for updating curiosity parameters
    outputs:
        - curiosity_score: Intrinsic reward signal based on routing uncertainty
    """
    def __init__(self, model, alpha=0.1, kl_beta=0.05, eta=0.1):
        self.model = model
        self.alpha = alpha
        self.kl_beta = kl_beta
        self.eta = eta

    def compute_intrinsic_reward(self, routing_trace, kl_div=None):
        """
        Calculates intrinsic reward from structural entropy and Bayesian uncertainty.
        routing_trace: List of layer decision dictionaries
        kl_div: Optional scalar tensor from Bayesian layers
        Returns: curiosity score tensor
        """
        total_entropy = 0
        node_count = 0
        device = None  # Will capture from first probability tensor
        
        for layer_data in routing_trace:
            # layer_data format: {layer_N: {node_X: {probs}}}
            for layer_name, tree_decisions in layer_data.items():
                for node_name, probs in tree_decisions.items():
                    p = probs['go_right_prob']
                    if not isinstance(p, torch.Tensor):
                        p = torch.tensor(p, dtype=torch.float32)
                    
                    if device is None:
                        device = p.device
                    
                    # Batch-safe entropy calculation
                    p = p.clamp(1e-7, 1-1e-7)
                    entropy = -(p * torch.log(p) + (1-p) * torch.log(1-p))
                    total_entropy += entropy.mean()
                    node_count += 1
                
        if node_count == 0:
            # Use captured device or default to CPU
            default_device = device if device is not None else torch.device('cpu')
            entropy_score = torch.tensor(0.0, device=default_device)
        else:
            entropy_score = total_entropy / node_count
            
        # Composite Reward: Discovery (Entropy) + Novelty (Bayesian KL)
        curiosity_score = self.alpha * entropy_score
        
        if kl_div is not None:
            # NORMALIZATION: KL divergence sum scales with parameter count.
            # We normalize by total Bayesian parameters to get "Average KL per Param".
            # This makes the signal invariant to architecture size.
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            kl_avg = kl_div / float(max(1, param_count))
            curiosity_score += self.kl_beta * kl_avg.to(curiosity_score.device)
            
        return curiosity_score

    def get_dynamic_epsilon(self, curiosity_signal):
        """ Converts curiosity signal to exploration epsilon.
        Inputs:
            - curiosity_signal: Scalar curiosity score
        Outputs:
            - epsilon: Exploration probability in [0.1, 1]
        """
        # High curiosity -> High epsilon -> High exploration
        return torch.sigmoid(curiosity_signal).item() * 0.9 + 0.1  # Scale to [0.1, 1]