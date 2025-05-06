# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy from custom dict with OmniSafe."""

# import omnisafe


# if __name__ == '__main__':
#     # env_id = 'SafetyAntVelocity-v1'
#     # env_id = 'SafetySwimmerVelocity-v1'
#     # env_id = 'SafetyHumanoidVelocity-v1'
#     # env_id = 'SafetyWalker2dVelocity-v1'
#     # env_id = 'SafetyHalfCheetahVelocity-v1'

#     # env_id = 'Glucose'
#     env_id = 'BiGlucose'
#     # env_id = 'CSTR'

#     custom_cfgs = {
#         'seed': 0,
#         'train_cfgs': {
#             # 'total_steps': 1_000_000,
#             'total_steps': 50_000,
#             'vector_env_nums': 1,
#             'parallel': 1,
#         },
#         'algo_cfgs': {
#             # 'steps_per_epoch': 2000,
#             'steps_per_epoch': 100,
#             'update_iters': 1,
#         },
#         'logger_cfgs': {
#             'use_wandb': False,
#         },
#     }

#     # agent = omnisafe.Agent('SACPID', env_id, custom_cfgs=custom_cfgs)
#     agent = omnisafe.Agent('PPOSimmerPID', env_id, custom_cfgs=custom_cfgs)

#     # exit()

#     agent.learn()

#     agent.plot(smooth=1)
#     agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
#     agent.evaluate(num_episodes=1)


import argparse
import omnisafe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True, help='Environment ID')
    parser.add_argument('--seed', type=int, required=True, help='Seed')
    args = parser.parse_args()
    env_id = args.env_id
    seed = args.seed

    custom_cfgs = {
        'seed': seed,
        'train_cfgs': {
            # 'total_steps': 1000000,
            'total_steps': 50_000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            # 'steps_per_epoch': 2000,
            'steps_per_epoch': 100,
            'update_iters': 1,
        },
        'logger_cfgs': {
            'use_wandb': False,
        },
    }

    agent = omnisafe.Agent('SACPID', env_id, custom_cfgs=custom_cfgs)
    # agent = omnisafe.Agent('PPOSimmerPID', env_id, custom_cfgs=custom_cfgs)

    agent.learn()
    # agent.plot(smooth=1)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    # agent.evaluate(num_episodes=1)
