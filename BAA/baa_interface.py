import torch
import os
import numpy as np
import threading 
from .orchestrator import AgenticDTT
from .brain import AgenticBrain

class BAAAgent:
    def __init__(self, state_dim, action_dim, weight_path="baa_weights.pt", lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_path = weight_path
        
        # Initialize Core Components
        self.model = AgenticDTT(state_dim=state_dim, action_dim=action_dim)
        self.brain = AgenticBrain(self.model, lr=lr)
        self.trajectories = {}
        
        # The Lock: Ensures only one thread touches the weights/data at a time
        self.lock = threading.Lock() 
        
        self.best_reward = -float('inf')
        self.reward_window = []  # Track moving average
        self.episode_count = 1   # Track for epsilon decay
        self.load_weights()

    def get_action(self, state, rtg, session_id="default"):
        with self.lock:
            device = self.model.get_device()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            else:
                state = state.to(device)
                
            rtg_tensor = torch.tensor([[[rtg]]], dtype=torch.float32).to(device)
            
            # Pass episode_count for epsilon decay
            action_tensor, trace, curiosity = self.brain.act(state, rtg_tensor, episode_num=self.episode_count)
            
            # Flatten to 1D [action_dim]
            return action_tensor.view(-1), trace, curiosity

    def record_experience(self, session_id, state, action, reward, done):
        """Trajectory management and training are protected by the lock"""
        with self.lock:
            if session_id not in self.trajectories:
                self.trajectories[session_id] = []
                
            # Convert to CPU tensors for storage efficiency and hindsight compatibility
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            else:
                state = state.detach().cpu()
                
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu()
            else:
                action = torch.tensor(action, dtype=torch.float32)
                
            self.trajectories[session_id].append((state, action, reward))
            
            if done:
                if session_id not in self.trajectories:
                    return None
                episode_data = self.trajectories.pop(session_id)
                
                # Calculate total reward for this segment/episode
                total_episode_reward = sum([x[2] for x in episode_data] if episode_data else [0])
                
                self.model.on_episode_finish(episode_data)
                
                # Autonomous training happens here
                results = self.brain.train_on_buffer(batch_size=32)
                
                # Update reward moving average (size 10)
                self.reward_window.append(total_episode_reward)
                if len(self.reward_window) > 10:
                    self.reward_window.pop(0)
                
                avg_reward = sum(self.reward_window) / len(self.reward_window)
                
                # Check for Best Model
                if len(self.reward_window) >= 5 and avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    print(f"🌟 New Best Average Reward: {avg_reward:.2f} (Saved to {self.weight_path.replace('.pt', '_best.pt')})")
                    self.save_weights(is_best=True)
                
                # Auto-save weights occasionally (10% chance after training)
                if np.random.random() < 0.1:
                    self.save_weights(is_best=False)
                
                self.episode_count += 1
                return results
        return None

    def load_weights(self):
        with self.lock:
            if os.path.exists(self.weight_path):
                try:
                    checkpoint = torch.load(self.weight_path, weights_only=True, map_location='cpu')
                    # Handle both raw state_dicts and BAA checkpoint dictionaries
                    state_dict = checkpoint['agentic_dtt'] if isinstance(checkpoint, dict) and 'agentic_dtt' in checkpoint else checkpoint
                    
                    # --- Smart Legacy Mapping ---
                    # Check if the checkpoint is non-Bayesian (has .weight instead of .weight_mu for routers)
                    current_model_dict = self.model.state_dict()
                    adapted_dict = {}
                    
                    for key, value in state_dict.items():
                        if key in current_model_dict:
                            adapted_dict[key] = value
                        
                        # --- RoPE Attention Mapping ---
                        elif "in_proj_weight" in key:
                            prefix = key.replace(".in_proj_weight", "")
                            # Only map if the new model expects separated keys (RoPE)
                            if f"{prefix}.q_proj.weight" in current_model_dict:
                                embed_dim = value.shape[1]
                                q_w, k_w, v_w = value.split(embed_dim, dim=0)
                                adapted_dict[f"{prefix}.q_proj.weight"] = q_w
                                adapted_dict[f"{prefix}.k_proj.weight"] = k_w
                                adapted_dict[f"{prefix}.v_proj.weight"] = v_w
                                print(f"  Mapped legacy attention weights: {prefix}")

                        elif "in_proj_bias" in key:
                            prefix = key.replace(".in_proj_bias", "")
                            if f"{prefix}.q_proj.bias" in current_model_dict:
                                embed_dim = value.shape[0] // 3
                                q_b, k_b, v_b = value.split(embed_dim, dim=0)
                                adapted_dict[f"{prefix}.q_proj.bias"] = q_b
                                adapted_dict[f"{prefix}.k_proj.bias"] = k_b
                                adapted_dict[f"{prefix}.v_proj.bias"] = v_b
                        
                        else:
                             # Try mapping legacy Linear keys to Bayesian mu keys
                             mu_key = key.replace(".weight", ".weight_mu").replace(".bias", ".bias_mu")
                             if mu_key in current_model_dict:
                                 adapted_dict[mu_key] = value
                    # Load with non-strict to allow unmapped Bayesian 'rho' parameters to keep their defaults
                    missing, unexpected = self.model.load_state_dict(adapted_dict, strict=False)
                    
                    if missing:
                        # Log missing keys (mostly rho parameters which is fine)
                        rho_missing = [k for k in missing if 'rho' in k]
                        other_missing = [k for k in missing if 'rho' not in k]
                        if other_missing:
                            print(f"⚠️  Missing non-variance keys in checkpoint: {other_missing}")
                        # print(f"✓ Loaded weights. Variance parameters ({len(rho_missing)}) initialized to defaults.")
                    
                    print(f"✓ Successfully loaded and adapted weights from {self.weight_path}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load weights: {e}")

    def save_weights(self, is_best=False):
        # Already inside a lock context when called from record_experience
        if is_best:
            best_path = self.weight_path.replace(".pt", "_best.pt")
            torch.save(self.model.state_dict(), best_path)
        else:
            torch.save(self.model.state_dict(), self.weight_path)