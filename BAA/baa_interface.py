import torch
import os
import queue
import threading
import numpy as np
from BAA.orchestrator import AgenticDTT
from BAA.brain import AgenticBrain


class BAAAgent:
    """
    High-level agent interface with decoupled inference and training threads.

    Architecture:
      - Inference thread (get_action): acquires _weight_lock briefly. In practice
        inference is fast and only waits if the optimizer is in the middle of a
        weight update (typically < 5 ms).
      - Training thread (_training_loop): a daemon thread that blocks on
        _episode_queue. When a completed episode arrives, it calls
        on_episode_finish + train_on_buffer, holding _weight_lock only during
        that window. record_experience() returns immediately after queueing.
      - Trajectory collection: protected by a lightweight _traj_lock that is
        completely separate from the weight lock, so data recording never
        blocks inference.
    """

    def __init__(self, state_dim, action_dim, weight_path="baa_weights.pt", lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_path = weight_path

        self.model = AgenticDTT(state_dim=state_dim, action_dim=action_dim)
        self.brain = AgenticBrain(self.model, lr=lr)

        # --- Locks ---
        # _weight_lock: held during the training forward+backward+step window,
        #               and during inference. This prevents torn reads on weights.
        self._weight_lock = threading.Lock()
        # _traj_lock: protects only the trajectories dict (cheap, fast).
        self._traj_lock = threading.Lock()
        # _save_lock: protects weight file writes.
        self._save_lock = threading.Lock()

        self.trajectories = {}
        self.best_reward = -float('inf')
        self.reward_window = []
        self.episode_count = 1

        # --- Background training queue ---
        # Completed episodes are posted here; the training thread drains it.
        self._episode_queue = queue.Queue()
        self._training_results = []      # last N training result dicts for monitoring

        self._training_thread = threading.Thread(
            target=self._training_loop,
            name="BAA-TrainThread",
            daemon=True,
        )

        self.load_weights()
        self._training_thread.start()
        print("✓ Background training thread started.")

    # ------------------------------------------------------------------
    # Inference (called from API / environment loop)
    # ------------------------------------------------------------------

    def get_action(self, state, rtg, session_id="default"):
        """
        Fast inference path. Acquires the weight lock only for the duration
        of the forward pass, which is independent of training time.
        """
        with self._weight_lock:
            device = self.model.get_device()
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            else:
                state = state.to(device)

            rtg_tensor = torch.tensor([[[rtg]]], dtype=torch.float32).to(device)
            action_tensor, trace, curiosity = self.brain.act(
                state, rtg_tensor, episode_num=self.episode_count
            )
            return action_tensor.view(-1), trace, curiosity

    # ------------------------------------------------------------------
    # Experience recording (called from API / environment loop)
    # ------------------------------------------------------------------

    def record_experience(self, session_id, state, action, reward, done):
        """
        Records a transition. Returns immediately when done=False.
        When done=True, posts the completed episode to the training queue
        and returns — training happens asynchronously in the background.
        """
        with self._traj_lock:
            if session_id not in self.trajectories:
                self.trajectories[session_id] = []

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
                total_reward = sum(x[2] for x in episode_data) if episode_data else 0.0

                # Post to background thread — this call returns immediately.
                self._episode_queue.put({
                    "episode_data": episode_data,
                    "total_reward": total_reward,
                })
                return {"queued": True, "episode_length": len(episode_data)}

        return None

    # ------------------------------------------------------------------
    # Background training loop
    # ------------------------------------------------------------------

    def _training_loop(self):
        """
        Runs in a daemon thread. Blocks on the episode queue and trains
        whenever a completed episode is available.
        """
        while True:
            try:
                item = self._episode_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            episode_data = item["episode_data"]
            total_reward = item["total_reward"]

            try:
                with self._weight_lock:
                    # Push relabeled (original + hindsight) samples to replay buffer.
                    self.model.on_episode_finish(episode_data)

                    # Train on the buffer.
                    results = self.brain.train_on_buffer(batch_size=32)

                if results is not None:
                    results['total_loss'] = results['loss'].item()
                    results['active_mode'] = "Task-Mastery (Buffer Replay)"
                    self._training_results.append(results)
                    if len(self._training_results) > 20:
                        self._training_results.pop(0)

                # Update reward tracking (outside the weight lock — no weights involved).
                self.reward_window.append(total_reward)
                if len(self.reward_window) > 10:
                    self.reward_window.pop(0)

                avg_reward = sum(self.reward_window) / len(self.reward_window)

                if len(self.reward_window) >= 5 and avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    print(f"New best average reward: {avg_reward:.2f} — saving best weights.")
                    self.save_weights(is_best=True)

                if np.random.random() < 0.1:
                    self.save_weights(is_best=False)

                self.episode_count += 1

            except Exception as e:
                print(f"Training thread error: {e}")
            finally:
                self._episode_queue.task_done()

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def load_weights(self):
        """Load model weights from disk with legacy key mapping."""
        if not os.path.exists(self.weight_path):
            return

        try:
            checkpoint = torch.load(self.weight_path, weights_only=True, map_location='cpu')
            state_dict = (
                checkpoint['agentic_dtt']
                if isinstance(checkpoint, dict) and 'agentic_dtt' in checkpoint
                else checkpoint
            )

            current_model_dict = self.model.state_dict()
            adapted_dict = {}

            for key, value in state_dict.items():
                if key in current_model_dict:
                    adapted_dict[key] = value

                elif "in_proj_weight" in key:
                    prefix = key.replace(".in_proj_weight", "")
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
                    mu_key = key.replace(".weight", ".weight_mu").replace(".bias", ".bias_mu")
                    if mu_key in current_model_dict:
                        adapted_dict[mu_key] = value

            missing, _ = self.model.load_state_dict(adapted_dict, strict=False)
            other_missing = [k for k in missing if 'rho' not in k]
            if other_missing:
                print(f"  Missing non-variance keys in checkpoint: {other_missing}")

            print(f"✓ Loaded weights from {self.weight_path}")

        except Exception as e:
            print(f"  Failed to load weights: {e}")

    def save_weights(self, is_best=False):
        """Save model weights to disk, protected by a save lock."""
        with self._save_lock:
            path = self.weight_path.replace(".pt", "_best.pt") if is_best else self.weight_path
            torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_training_status(self):
        """Return a snapshot of recent training metrics for the API / debugger."""
        return {
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
            "queue_depth": self._episode_queue.qsize(),
            "recent_results": self._training_results[-5:] if self._training_results else [],
        }
