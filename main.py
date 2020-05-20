import maml_rl.envs
import gym
import numpy as np
import torch
import json
import time

from matplotlib import pyplot as plt
from maml_rl.metalearner import MetaLearner
from maml_rl.multi_task_metalearner import Multi_MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy, hyper_normal_mlp
from maml_rl.policies.hyper_normal_mlp import Hyper_Policy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
#from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1', 'HalfCheetahDirBullet-v0','AntPosBullet-v0','AntDirBullet-v0',
        'AntVelBullet-v0','HalfCheetahVelBullet-v0', '2DNavigation-v0','Sparse2DNavigation-v0'])

    # writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    if args.no_hyper:
        file_name = args.env_name + "_regular_maml_"
    else:
        file_name = args.env_name + "_hyper_maml_"

    if args.multi_task_critic:
        file_name = file_name + "_multi_task_extra"

    file_name = file_name + str(args.seed)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers, seed=args.seed)
    if continuous_actions:
        if args.no_hyper:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers)
            print("regular policy")
        else:
            policy = Hyper_Policy(args.task_dim,
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                args.num_hyper_layers)
            args.fast_lr=args.fast_hyper_lr
            print("hyper policy")
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    if args.multi_task_critic:
        metalearner = Multi_MetaLearner(sampler, policy, baseline, gamma=args.gamma,
            fast_lr=args.fast_lr, tau=args.tau, device=args.device)
    else:
        metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
            fast_lr=args.fast_lr, tau=args.tau, device=args.device)

        
    before_rewards = []
    after_rewards = []
    test_b_rewards = []
    test_a_rewards = []
    total_steps = 0
    episode_num = 0
    t0 = time.time()
    
    #test tasks
    tasks, task_norm = sampler.sample_unseen_task(num_of_unseen=args.num_of_tasks)
    #train tasks
    unseen_task, _ = sampler.sample_unseen_task(tasks, num_of_unseen=args.test_tasks)
  #  unseen_task = sampler.sample_unseen_task(num_of_unseen=args.test_tasks)
    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(tasks, num_tasks=args.meta_batch_size)
      #  tasks, task_norm = sampler.sample_tasks(unseen_task, num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks, task_norm, first_order=args.first_order)
        metalearner.step(episodes, args.device, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)
        total_steps += args.ls_max_steps
        test_before, test_after = metalearner.evaluate_task(unseen_task,  task_norm, episode_num)
        test_a_rewards.append(test_after); test_b_rewards.append(test_before)
        before_rewards.append(total_rewards([ep.rewards for __, ep, _ in episodes]))
        after_rewards.append(total_rewards([ep.rewards for __, _, ep, in episodes]))
        print("Episode Num: {}  before: {:.3f}  after: {:.3f} (test_before: {:.3f}  test_after: {:.3f} ) --  time: {} sec".format(
					episode_num, before_rewards[-1], after_rewards[-1],test_b_rewards[-1],test_a_rewards[-1], int(time.time() - t0)))
        episode_num += 1

        # save evaluations
        np.save("./results/%s_after_rewards" % (file_name), after_rewards)
        np.save("./results/%s_before_rewards" % (file_name), before_rewards)
        np.save("./results/%s_test_after_rewards" % (file_name), test_a_rewards)
        np.save("./results/%s_test_before_rewards" % (file_name), test_b_rewards)

    
    title = ""
    for task in unseen_task:
        title += "_" + str(task.values())
    plt.title(args.env_name + "unseen_task: " + title)
    plt.plot(range(len(before_rewards)), before_rewards, label="before")
    plt.plot(range(len(after_rewards)), after_rewards, label="after")
    plt.plot(range(len(test_b_rewards)), test_b_rewards, label="test_before")
    plt.plot(range(len(test_a_rewards)), test_a_rewards, label="test_after")
    plt.legend()
    plt.savefig(file_name + ".png") 
    

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')
    # General
    parser.add_argument('--env-name', type=str, default='AntDirBullet-v0',#'Sparse2DNavigation-v0',#'AntPosBullet-v0',#'AntVelBullet-v0',#"HalfCheetahVelBullet-v0",'2DNavigation-v0','HalfCheetahDirBullet-v0',#
        help='name of the environment')
    parser.add_argument("--no_hyper", action="store_true")	# use regular critic
    parser.add_argument("--multi_task_critic", action="store_true", default=True)	# use multi task critic
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--seed', type=int, default=0,
        help='set seed')
    parser.add_argument('--test_tasks', type=int, default=5,
        help='number of unseen task or testing')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    parser.add_argument('--num-hyper-layers', type=int, default=3,
        help='number of hidden layers in hyper net')

    # Task-specific
    parser.add_argument('--task-dim', type=int, default=2,
        help='value of the discount factor gamma')
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--max-horizon', type=int, default=200,
        help='max horizon H for sample')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--fast-hyper-lr', type=float, default=5e-5,
        help='learning rate for the 1-step gradient update of hyper MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=500,
        help='number of batches')
    parser.add_argument('--num-of-tasks', type=int, default=100,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=2,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=20,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    print(args.device)
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    
    print(args)
    main(args)
