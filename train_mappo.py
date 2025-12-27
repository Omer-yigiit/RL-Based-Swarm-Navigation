import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from env.swarm_nav_env import SwarmNavEnv

def env_creator(cfg):
    return SwarmNavEnv(cfg)

if __name__ == "__main__":
    ray.init(
        num_cpus = 1,
        include_dashboard=False,
        local_mode= True,
        ignore_reinit_error=True,
        log_to_driver = False
    )

    register_env("SwarmNav-v1", env_creator)

    temp_env = SwarmNavEnv({"num_agents": 4})

    policies = {
        "shared_policy": (
            None,
            temp_env.observation_space,
            temp_env.action_space,
            {}
        )
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment("SwarmNav-v1", env_config={"num_agents": 4})
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            clip_param=0.2,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
    )

    algo = config.build()

    for i in range(500):
        result = algo.train()
        print(
            f"[{i}] "
            f"reward_mean={result['episode_reward_mean']:.2f}"
        )

        if i % 50 == 0:
            algo.save("checkpoints_mappo/iter_{i}")
