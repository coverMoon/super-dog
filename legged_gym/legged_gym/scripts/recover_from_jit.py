import argparse
import os

import torch

from legged_gym.envs import *
from legged_gym.utils import class_to_dict, task_registry
from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import ActorCritic, HIMActorCritic


def build_actor_critic(task_name, device):
    env_cfg, train_cfg = task_registry.get_cfgs(task_name)

    num_actor_obs = env_cfg.env.num_observations
    num_critic_obs = env_cfg.env.num_privileged_obs
    if num_critic_obs is None:
        num_critic_obs = num_actor_obs
    num_actions = env_cfg.env.num_actions

    policy_cfg = class_to_dict(train_cfg.policy)
    policy_class = eval(train_cfg.runner.policy_class_name)

    if policy_class is HIMActorCritic:
        actor_critic = policy_class(
            num_actor_obs,
            num_critic_obs,
            env_cfg.env.num_one_step_observations,
            num_actions,
            **policy_cfg,
        ).to(device)
    else:
        actor_critic = policy_class(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            **policy_cfg,
        ).to(device)

    return actor_critic, train_cfg


def build_algorithm(actor_critic, train_cfg, device):
    algorithm_cfg = class_to_dict(train_cfg.algorithm)
    algorithm_class = eval(train_cfg.runner.algorithm_class_name)
    return algorithm_class(actor_critic, device=device, **algorithm_cfg)


def recover_weights(actor_critic, jit_module):
    target_state = actor_critic.state_dict()
    source_state = jit_module.state_dict()

    recovered = []
    skipped = []

    for source_key, source_tensor in source_state.items():
        target_key = None

        if source_key in target_state and target_state[source_key].shape == source_tensor.shape:
            target_key = source_key
        elif source_key.startswith("estimator."):
            candidate = "estimator.encoder." + source_key[len("estimator."):]
            if candidate in target_state and target_state[candidate].shape == source_tensor.shape:
                target_key = candidate
        else:
            candidate = "actor." + source_key
            if candidate in target_state and target_state[candidate].shape == source_tensor.shape:
                target_key = candidate

        if target_key is None:
            skipped.append(source_key)
            continue

        target_state[target_key] = source_tensor.detach().to(target_state[target_key].device, dtype=target_state[target_key].dtype)
        recovered.append((source_key, target_key))

    actor_critic.load_state_dict(target_state, strict=False)
    return recovered, skipped


def save_checkpoint(path, actor_critic, algorithm, iteration, jit_path, recovered, skipped):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": algorithm.optimizer.state_dict(),
        "iter": iteration,
        "infos": {
            "recovered_from_jit": os.path.abspath(jit_path),
            "recovered_param_count": len(recovered),
            "skipped_param_count": len(skipped),
        },
    }

    if hasattr(actor_critic, "estimator") and hasattr(actor_critic.estimator, "optimizer"):
        checkpoint["estimator_optimizer_state_dict"] = actor_critic.estimator.optimizer.state_dict()

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Recover a trainable checkpoint from an exported TorchScript policy.")
    parser.add_argument("--task", required=True, help="Registered task name, e.g. black / black_bridge / black_arm")
    parser.add_argument("--jit", required=True, help="Path to exported TorchScript policy, e.g. policy.pt")
    parser.add_argument("--output", required=True, help="Output checkpoint path, e.g. ./recovered/model_from_jit.pt")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number written into the recovered checkpoint")
    parser.add_argument("--device", default="cpu", help="Device used to instantiate the model")
    args = parser.parse_args()

    jit_path = os.path.abspath(args.jit)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(jit_path):
        raise FileNotFoundError(f"JIT policy not found: {jit_path}")

    actor_critic, train_cfg = build_actor_critic(args.task, args.device)
    algorithm = build_algorithm(actor_critic, train_cfg, args.device)

    jit_module = torch.jit.load(jit_path, map_location="cpu")
    recovered, skipped = recover_weights(actor_critic, jit_module)

    if not recovered:
        raise RuntimeError(
            "No parameters were recovered from the JIT file. "
            "This usually means the task config no longer matches the exported policy architecture."
        )

    save_checkpoint(
        output_path,
        actor_critic,
        algorithm,
        args.iteration,
        jit_path,
        recovered,
        skipped,
    )

    print(f"Recovered checkpoint saved to: {output_path}")
    print(f"Recovered tensors: {len(recovered)}")
    print(f"Skipped tensors: {len(skipped)}")
    if skipped:
        print("Skipped keys:")
        for key in skipped:
            print(f"  - {key}")


if __name__ == "__main__":
    main()
