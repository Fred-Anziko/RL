import torch
import os
import queue
import threading
import time
import numpy as np
from BAA.orchestrator import AgenticDTT
from BAA.brain import AgenticBrain


class BAAAgent:
    """
    High-level agent interface with decoupled inference and training threads,
    plus a watchdog that automatically restarts the training thread if it crashes.

    Architecture:
      - Inference  (_weight_lock, brief):  get_action() acquires the weight lock
        only for the duration of the forward pass. The lock is also checked for
        thread health: if the training thread has been silent for > 60 s, it is
        restarted automatically.
      - Training   (_weight_lock, longer): background daemon thread holds the lock
        during on_episode_finish + train_on_buffer, then releases it.
      - Recording  (_traj_lock, tiny):     record_experience() only touches the
        trajectory dict. When an episode ends, it posts to the queue and returns
        immediately — no training work happens on the calling thread.
      - Watchdog:  _last_heartbeat is updated every training iteration. get_action()
        (called frequently) checks the heartbeat and restarts the thread on stale.
    """

    WATCHDOG_TIMEOUT = 60.0   # seconds of silence before restarting the train thread

    def __init__(self, state_dim, action_dim, weight_path="baa_weights.pt", lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_path = weight_path

        self.model = AgenticDTT(state_dim=state_dim, action_dim=action_dim)
        self.brain = AgenticBrain(self.model, lr=lr)

        # --- Locks ---
        self._weight_lock = threading.Lock()
        self._traj_lock   = threading.Lock()
        self._save_lock   = threading.Lock()

        self.trajectories  = {}
        self.best_reward   = -float('inf')
        self.reward_window = []
        self.episode_count = 1

        # --- Background training ---
        self._episode_queue    = queue.Queue()
        self._training_results = []
        self._last_heartbeat   = time.time()   # updated by training thread each iteration
        self._thread_lock      = threading.Lock()  # guards thread creation

        self.load_weights()
        self._start_training_thread()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _start_training_thread(self):
        """Create and start a new daemon training thread."""
        t = threading.Thread(
            target=self._training_loop,
            name="BAA-TrainThread",
            daemon=True,
        )
        t.start()
        self._training_thread = t
        self._last_heartbeat = time.time()
        print("  Background training thread started.")

    def _restart_training_thread(self):
        """Restart the training thread after a crash or prolonged silence."""
        with self._thread_lock:
            if self._training_thread.is_alive():
                return   # already running — another caller beat us here
            print("  Training thread unresponsive — restarting.")
            self._start_training_thread()

    def _check_watchdog(self):
        """
        Called from get_action(). If the training thread has been silent for
        longer than WATCHDOG_TIMEOUT seconds, restart it.
        """
        if not self._training_thread.is_alive():
            self._restart_training_thread()
        elif (time.time() - self._last_heartbeat) > self.WATCHDOG_TIMEOUT:
            print(f"  Watchdog: training thread silent for >{self.WATCHDOG_TIMEOUT}s — restarting.")
            # The thread may be stuck; daemon threads can't be force-killed,
            # but starting a new one drains the queue going forward.
            self._start_training_thread()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action(self, state, rtg, session_id="default"):
        """
        Fast inference path. Acquires the weight lock only for the forward pass.
        Also runs the watchdog check to ensure the training thread is healthy.
        """
        self._check_watchdog()

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
    # Experience recording
    # ------------------------------------------------------------------

    def record_experience(self, session_id, state, action, reward, done):
        """
        Records a transition. Returns immediately when done=False.
        When done=True, posts the completed episode to the training queue —
        training happens asynchronously in the background thread.
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
                episode_data  = self.trajectories.pop(session_id)
                total_reward  = sum(x[2] for x in episode_data) if episode_data else 0.0
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
        Daemon thread. Blocks on the episode queue and trains whenever a
        completed episode is available. Updates _last_heartbeat every iteration
        so the watchdog can detect if this thread has stalled.
        """
        while True:
            # Heartbeat: update even when idle so watchdog knows we're alive.
            self._last_heartbeat = time.time()

            try:
                item = self._episode_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            episode_data = item["episode_data"]
            total_reward = item["total_reward"]

            try:
                with self._weight_lock:
                    self.model.on_episode_finish(episode_data)
                    results = self.brain.train_on_buffer(batch_size=32)

                # Heartbeat after training step — most important update point.
                self._last_heartbeat = time.time()

                if results is not None:
                    results["total_loss"]  = results["loss"].item()
                    results["active_mode"] = "Task-Mastery (Buffer Replay)"
                    self._training_results.append(results)
                    if len(self._training_results) > 20:
                        self._training_results.pop(0)

                self.reward_window.append(total_reward)
                if len(self.reward_window) > 10:
                    self.reward_window.pop(0)

                avg_reward = sum(self.reward_window) / len(self.reward_window)

                if len(self.reward_window) >= 5 and avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    print(f"  New best avg reward: {avg_reward:.2f} — saving best weights.")
                    self.save_weights(is_best=True)

                if np.random.random() < 0.1:
                    self.save_weights(is_best=False)

                self.episode_count += 1

            except Exception as e:
                # Log the error clearly — the thread continues running.
                print(f"  Training error (episode {self.episode_count}): {type(e).__name__}: {e}")
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

            print(f"  Loaded weights from {self.weight_path}")

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
        """Return a snapshot of recent training metrics."""
        return {
            "episode_count":   self.episode_count,
            "best_reward":     self.best_reward,
            "queue_depth":     self._episode_queue.qsize(),
            "thread_alive":    self._training_thread.is_alive(),
            "last_heartbeat_age_s": round(time.time() - self._last_heartbeat, 1),
            "recent_results":  self._training_results[-5:] if self._training_results else [],
        }
