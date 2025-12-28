import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

from env.swarm_nav_env import SwarmNavEnv


def env_creator(config):
    return SwarmNavEnv(config)


register_env("SwarmNavEnv", env_creator)


if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)

    dummy_env = SwarmNavEnv({
        "num_agents": 4,
        "arena_size": 10.0,
        "max_steps": 300,
    })

    config = (
        PPOConfig()
        .environment(
            env="SwarmNavEnv",
            env_config={
                "num_agents": 4,
                "arena_size": 10.0,
                "max_steps": 300,
            }
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=200,
            enable_connectors=True,
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=dummy_env.observation_space,
                    action_space=dummy_env.action_space,
                )
            },
            policy_mapping_fn=lambda aid, *args, **kwargs: "shared_policy",
        )
        .training(
            gamma=0.99,
            lr=1e-4,
            entropy_coeff=0.001,
            clip_param=0.15,
            train_batch_size=8000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    for i in range(2000):
        result = algo.train()
        print(f"Iter {i:04d} | reward_mean: {result['episode_reward_mean']:.2f}")

    algo.save("checkpoints/swarm_nav")
    ray.shutdown()
