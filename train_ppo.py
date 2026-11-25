"""RecurrentPPO training script for UAV navigation."""

import argparse
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import config
from airsim_env import AirSimUAVEnv
from callbacks import EpisodeLogCallback


def make_env(goal=None):
    env = AirSimUAVEnv(goal=goal)
    return Monitor(env)


def linear_schedule(initial_value: float, final_value: float = 0.0):
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--alpha", type=float, default=config.ALPHA_SMOOTH,
                         help="velocity smoothing coefficient (Table 2)")
    parser.add_argument("--resume", type=str, default=None, help="path to a .zip checkpoint to resume from")
    parser.add_argument("--goal_x", type=float, default=None, help="fixed training goal X (NED, meters)")
    parser.add_argument("--goal_y", type=float, default=None, help="fixed training goal Y (NED, meters)")
    parser.add_argument("--goal_z", type=float, default=None, help="fixed training goal Z (NED, meters, negative=up)")
    parser.add_argument("--fixed_goal", action="store_true",
                         help="train toward config.FIXED_GOAL instead of sampling a new random goal every episode")
    args = parser.parse_args()

    config.ALPHA_SMOOTH = args.alpha

    if not args.fixed_goal:
        config.FIXED_GOAL = None

    goal = None
    if args.goal_x is not None and args.goal_y is not None and args.goal_z is not None:
        goal = (args.goal_x, args.goal_y, args.goal_z)
        print(f"Training toward fixed goal: {goal}")
    elif config.FIXED_GOAL is not None:
        print(f"Training toward config.FIXED_GOAL: {config.FIXED_GOAL}")
    else:
        print("Random goal sampled each episode.")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    env = make_env(goal=goal)

    if args.resume:
        model = RecurrentPPO.load(args.resume, env=env, tensorboard_log=config.LOG_DIR)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=linear_schedule(3e-4, 1e-5),
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                lstm_hidden_size=128,
                n_lstm_layers=1,
                # start with smaller action std (~0.22) to avoid early wall collisions
                log_std_init=-1.5,
            ),
            tensorboard_log=config.LOG_DIR,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=config.MODEL_DIR,
        name_prefix=f"ppo_uav_alpha{args.alpha}",
    )
    episode_log_cb = EpisodeLogCallback(
        best_model_path=os.path.join(config.MODEL_DIR, f"ppo_uav_best_alpha{args.alpha}.zip"))
    callback = CallbackList([checkpoint_cb, episode_log_cb])

    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="ppo")

    final_path = os.path.join(config.MODEL_DIR, f"ppo_uav_final_alpha{args.alpha}.zip")
    model.save(final_path)
    print(f"Saved final PPO model to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
