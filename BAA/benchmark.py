"""
BAA Benchmark — Online Comparison on Pendulum-v1
=================================================
Runs BAA and a random-policy baseline on Pendulum-v1 (continuous control,
3-dimensional observation, 1-dimensional action). Prints episode rewards and
a final summary table so the two can be compared head-to-head.

Usage:
    python3 BAA/benchmark.py [--episodes 100] [--seed 42]

Why Pendulum-v1?
  - Continuous action space (like the humanoid model), so BAA's tanh policy
    head applies directly without modification.
  - Lightweight: no MuJoCo license required, runs on CPU in seconds.
  - Well-understood baselines: a solved agent scores ~-150 to -200.

RTG conditioning:
  - BAA is an RTG-conditioned agent. We set rtg=0 initially and track a
    running moving average of recent returns, updating rtg between episodes.
    This gives the agent a realistic "target to beat" as it learns.
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
import gymnasium as gym

# Make sure BAA imports work from any CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from BAA.baa_interface import BAAAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_random_episode(env):
    """One episode with a uniform random policy. Returns total reward."""
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    return total_reward


def run_baa_episode(env, agent, rtg_target, session_id="bench"):
    """
    One episode with the BAA agent (online learning enabled).
    Returns total reward and episode length.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        # Scale obs to roughly unit range for this environment
        state = obs.tolist()
        action_tensor, _, _ = agent.get_action(state, rtg=rtg_target, session_id=session_id)

        # Pendulum action space: [-2, 2]; BAA outputs tanh in [-1, 1]
        action_np = (action_tensor.detach().cpu().numpy() * 2.0).clip(-2.0, 2.0)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        agent.record_experience(
            session_id=session_id,
            state=state,
            action=action_tensor.detach().cpu(),
            reward=float(reward),
            done=done,
        )

        total_reward += reward
        step += 1

    return total_reward, step


def fmt(val, width=10):
    return f"{val:>{width}.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BAA vs Random benchmark on Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per agent (default 100)")
    parser.add_argument("--seed",     type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N = args.episodes
    print(f"\nBAA Benchmark — Pendulum-v1  |  {N} episodes each  |  seed={args.seed}")
    print("=" * 70)

    # ------------------------------------------------------------------ Random
    print("\n[1/2]  Running random policy baseline...")
    env_rand = gym.make("Pendulum-v1")
    env_rand.reset(seed=args.seed)
    random_rewards = []
    t0 = time.time()
    for ep in range(N):
        r = run_random_episode(env_rand)
        random_rewards.append(r)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:>4}/{N}   reward={r:>8.1f}   "
                  f"avg={np.mean(random_rewards):>8.1f}")
    env_rand.close()
    random_time = time.time() - t0

    # ------------------------------------------------------------------ BAA
    print(f"\n[2/2]  Running BAA (state_dim=3, action_dim=1, online learning)...")
    # Pendulum-v1 observation space: Box(3,)
    agent = BAAAgent(state_dim=3, action_dim=1, weight_path="/tmp/baa_pendulum.pt", lr=5e-4)

    env_baa = gym.make("Pendulum-v1")
    env_baa.reset(seed=args.seed)
    baa_rewards = []
    rtg_target  = 0.0      # start neutral; updated to moving avg each episode
    reward_window = []

    t0 = time.time()
    for ep in range(N):
        r, steps = run_baa_episode(env_baa, agent, rtg_target=rtg_target, session_id=f"ep_{ep}")
        baa_rewards.append(r)

        reward_window.append(r)
        if len(reward_window) > 10:
            reward_window.pop(0)
        rtg_target = float(np.mean(reward_window))   # update RTG estimate

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:>4}/{N}   reward={r:>8.1f}   "
                  f"avg={np.mean(baa_rewards):>8.1f}   "
                  f"buffer={len(agent.model.replay_buffer):>5}   "
                  f"rtg_target={rtg_target:>7.1f}")
    env_baa.close()
    baa_time = time.time() - t0

    # Wait briefly for the training queue to drain
    time.sleep(2.0)

    # ------------------------------------------------------------------ Summary
    rand_arr = np.array(random_rewards)
    baa_arr  = np.array(baa_rewards)

    # Compare first vs second half to see if BAA improved
    baa_first  = baa_arr[:N//2].mean()
    baa_second = baa_arr[N//2:].mean()

    print("\n" + "=" * 70)
    print(f"{'SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Metric':<28} {'Random':>12} {'BAA':>12} {'Delta':>12}")
    print("-" * 70)
    print(f"{'Mean reward':<28} {fmt(rand_arr.mean())} {fmt(baa_arr.mean())} {fmt(baa_arr.mean() - rand_arr.mean())}")
    print(f"{'Std reward':<28} {fmt(rand_arr.std())} {fmt(baa_arr.std())} {'':>12}")
    print(f"{'Best episode':<28} {fmt(rand_arr.max())} {fmt(baa_arr.max())} {fmt(baa_arr.max() - rand_arr.max())}")
    print(f"{'Worst episode':<28} {fmt(rand_arr.min())} {fmt(baa_arr.min())} {fmt(baa_arr.min() - rand_arr.min())}")
    print(f"{'BAA: first half avg':<28} {'':>12} {fmt(baa_first)} {'':>12}")
    print(f"{'BAA: second half avg':<28} {'':>12} {fmt(baa_second)} {fmt(baa_second - baa_first)}")
    print(f"{'Wall time (s)':<28} {random_time:>12.1f} {baa_time:>12.1f} {'':>12}")
    print("=" * 70)

    improvement = baa_second - baa_first
    vs_random   = baa_arr.mean() - rand_arr.mean()
    print(f"\nBAA learning trend (2nd half vs 1st half): {improvement:+.1f}")
    print(f"BAA vs random (mean): {vs_random:+.1f}")

    status = agent.get_training_status()
    print(f"\nTraining status: {status['episode_count']} episodes processed, "
          f"buffer={len(agent.model.replay_buffer)} windows, "
          f"thread_alive={status['thread_alive']}, "
          f"heartbeat_age={status['last_heartbeat_age_s']}s")

    if improvement > 0:
        print("\nConclusion: BAA improved over the course of the benchmark.")
    else:
        print("\nConclusion: BAA did not improve — more episodes or tuning needed.")

    return {
        "random_mean": rand_arr.mean(),
        "baa_mean":    baa_arr.mean(),
        "baa_trend":   improvement,
    }


if __name__ == "__main__":
    main()
