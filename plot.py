import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def plot(args):
    dirr = "results/results/"
    args.env = "Sparse2DNavigation-v0"#"HalfCheetahVelBullet-v0"#"HalfCheetahDirBullet-v0"#"AntDirBullet-v0"#"AntPosBullet-v0"#"AntVelBullet-v0"#
    args.hyper_file = args.env + "_hyper_maml_"
    args.reg_file = args.env + "_regular_maml_"
    test = '_test'
    last1 = 500
    last2 = 500
    last3 = 500
    last4 = 500
    last5 = 500
    a0 = np.load(dirr + args.reg_file + "0_test_before_rewards.npy")[:last1]
  #  a1 = np.load(dirr + args.reg_file + "1_before_rewards.npy")[:last1]
  #  a2 = np.load(dirr + args.reg_file + "2_before_rewards.npy")[:last1]
    b0 = np.load(dirr + args.reg_file + "0_test_after_rewards.npy")[:last2]
  #  b1 = np.load(dirr + args.reg_file + "1_after_rewards.npy")[:last2]
  #  b2 = np.load(dirr + args.reg_file + "2_after_rewards.npy")[:last2]
    c0 = np.load(dirr + args.hyper_file + "0_test_before_rewards.npy")[:last3]
  #  c1 = np.load(dirr + args.hyper_file + "1_before_rewards.npy")[:last3]
  #  c2 = np.load(dirr + args.hyper_file + "2_before_rewards.npy")[:last3]
    d0 = np.load(dirr + args.hyper_file + "0_test_after_rewards.npy")[:last4]
  #  d1 = np.load(dirr + args.hyper_file + "1_after_rewards.npy")[:last4]
  #  d2 = np.load(dirr + args.hyper_file + "2_after_rewards.npy")[:last4]
    e0 = np.load(dirr + args.hyper_file + "_multi_task_0_test_before_rewards.npy")[:last5]
  #  e1 = np.load(dirr + args.hyper_file + "_multi_task_1_before_rewards.npy")[:last5]
  #  e2 = np.load(dirr + args.hyper_file + "_multi_task_2_before_rewards.npy")[:last5]

    #total = np.vstack([a0,a1 ,a2])
    total = np.vstack([a0])
    min_a = gaussian_filter1d(np.min(total,axis=0), sigma=2)
    max_a = gaussian_filter1d(np.max(total, axis=0),sigma=2)
    avg_a = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
    #total = np.vstack([b0, b1 ,b2])
    total = np.vstack([b0])
    min_b = gaussian_filter1d(np.min(total,axis=0),sigma=2)
    max_b = gaussian_filter1d(np.max(total, axis=0),sigma=2)
    avg_b = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
    #total = np.vstack([c0, c1, c2])
    total = np.vstack([c0])
    min_c = gaussian_filter1d(np.min(total,axis=0),sigma=2)
    max_c = gaussian_filter1d(np.max(total, axis=0),sigma=2)
    avg_c = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
    #total = np.vstack([d0, d1, d2])
    total = np.vstack([d0])
    min_d = gaussian_filter1d(np.min(total,axis=0),sigma=2)
    max_d = gaussian_filter1d(np.max(total, axis=0),sigma=2)
    avg_d = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
    #total = np.vstack([e0, e1 ,e2])
    total = np.vstack([e0])
    min_e = gaussian_filter1d(np.min(total,axis=0),sigma=2)
    max_e = gaussian_filter1d(np.max(total, axis=0),sigma=2)
    avg_e = gaussian_filter1d(np.mean(total, axis=0),sigma=2)

    plt.title(args.env + test)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.plot(range(len(avg_a)), avg_a, label="regular_MAML_before", color='#CC4F1B')
    plt.plot(range(len(avg_b)), avg_b, label="regular_MAML_after_one_step", color='g')
    plt.plot(range(len(avg_c)), avg_c, label="Hyper_MAML_before", color='#1B2ACC')
    plt.plot(range(len(avg_d)), avg_d, label="Hyper_MAML_after_one_step", color='m')
    plt.plot(range(len(avg_e)), avg_e, label="Hyper_TRPO_multi_task",color='y')
    plt.fill_between(range(len(avg_b)), min_b, max_b, facecolor='lightgreen', alpha=0.9)
    plt.fill_between(range(len(avg_d)), min_d, max_d, facecolor='plum', alpha=0.9)
    plt.fill_between(range(len(avg_e)), min_e, max_e, facecolor='lightyellow', alpha=0.7)
    plt.legend()
    plt.savefig(args.env+test+".png")
    plt.close()  

if __name__ == "__main__":
    	
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_file", type=str)				# hyper net file
    parser.add_argument("--reg_file", type=str)					# regular net file
    parser.add_argument("--env", type=str)					    # env
    args = parser.parse_args()
    plot(args)
