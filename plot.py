import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.lines import Line2D
import matplotlib
plt.style.use('ggplot')

def grad_plot(args):
  fontsize=20
  matplotlib.rc('font', size=fontsize)
  fig, axs = plt.subplots(1, 3,constrained_layout=True, figsize=(20,4))
  envs =["HalfCheetahDirBullet-v0", "AntDirBullet-v0", "AntVelBullet-v0"]
  envs_name =["HalfCheetah-Dir", "Ant-Dir", "Ant-Vel"]
  for i,env in enumerate(envs):
      hyper_name = env + "_hyper_maml_0_value"
      standard_name = env +"_regular_maml_0_value"
      standard_task_name = env +"_regular_maml_task_0_value"
      reverse_name = env + "_hyper_reverse_0_value"
      reverse = False

      hyper_b = np.load("./values/%s_before.npy" % (hyper_name),allow_pickle=True)
      standard_b = np.load("values/%s_before.npy" % (standard_name),allow_pickle=True)
      standard_task_b = np.load("values/%s_before.npy" % (standard_task_name),allow_pickle=True)
      
      if reverse:
        reverse_b = np.load("values/%s_before.npy" % (reverse_name),allow_pickle=True)
        std_br = np.std(reverse_b,axis=1)
        mean_br = np.mean(reverse_b,axis=1)
        cov_br = std_br / mean_br

      
      std_bh = np.std(hyper_b,axis=1)
      std_bs = np.std(standard_b,axis=1)
      std_bst = np.std(standard_task_b,axis=1)
      
      
      mean_bh = np.abs(np.mean(hyper_b,axis=1))
      mean_bs = np.abs(np.mean(standard_b,axis=1))
      mean_bst = np.abs(np.mean(standard_task_b,axis=1))
      cov_bh = hyper_b#std_bh / mean_bh
      cov_bs = standard_b#std_bs / mean_bs
      cov_bst = standard_task_b

      z_h = std_bh / mean_bh
      z_s=std_bs / mean_bs
      z_st=std_bst / mean_bst
      
      width = 0.2
      c = "red"
      axs[i].boxplot(cov_bh.transpose(1,0),positions=[1.,2.,3.,4.],widths=width, notch=True, patch_artist=True,
                showfliers = False,
                showmeans=True,
                boxprops=dict(facecolor=c, color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                medianprops=dict(color=c),
                meanprops=dict(markerfacecolor='w', marker='D')
                )

      c = "blue"
      axs[i].boxplot(cov_bs.transpose(1,0),positions=[1.2,2.2,3.2,4.2],  widths=width, notch=True, patch_artist=True,
          showfliers = False,
          showmeans=True,
          boxprops=dict(facecolor=c, color=c),
          capprops=dict(color=c),
          whiskerprops=dict(color=c),
          medianprops=dict(color=c),
          meanprops=dict(markerfacecolor='k', marker='D')
          )
      c = "orange"
      axs[i].boxplot(cov_bst.transpose(1,0),positions=[1.4,2.4,3.4,4.4],  widths=width, notch=True, patch_artist=True,
          showfliers = False,
          showmeans=True,
          boxprops=dict(facecolor=c, color=c),
          capprops=dict(color=c),
          whiskerprops=dict(color=c),
          medianprops=dict(color=c),
          meanprops=dict(markerfacecolor='k', marker='D')
          )
      if reverse:
        c = "green"
        axs[i].boxplot(reverse_b.transpose(1,0),positions=[1.6, 2.6],  widths=width, notch=True, patch_artist=True,
            showfliers = False,
            showmeans=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            medianprops=dict(color=c),
            meanprops=dict(markerfacecolor='k', marker='D')
            )


      axs[i].set_title(envs_name[i], fontsize=20)

      
      x_labels = ["50 iter", "150 iter", "300 iter", "450 iter"]
      for ax in axs.flat:
        ax.set_xticks(np.arange(1,len(x_labels)+1))
        ax.set_xticklabels(x_labels)
      #  ax.set_ylim(-100,450)
      #  ax.set_ylim(0.5,1.5)
      
  #    for ax in axs.flat:
  #      ax.label_outer()  
  axs[0].set_ylabel(ylabel='reward')
  custom_lines = [Line2D([0], [0], color='blue', lw=2),
              Line2D([0], [0], color='red', lw=2),
              Line2D([0], [0], color='orange', lw=2),
              Line2D([0], [0], color='green', lw=2)]
  if reverse:
    leg = axs[0].legend(custom_lines,('Standard MAML std 150 = %f'%(std_bs[1]), 'Hyper MAML std 150 = %f'%(std_bh[1]),'Hyper reverse MAML std 150 = %f'%(std_br[1])),loc='upper left')
  else:
    leg = axs[0].legend(custom_lines,('MAML', 'Hyper MAML','Task MAML'),loc='lower right')
    
    
  plt.savefig("grad_var.pdf")

  plt.close() 


def normlaize_plot():
    fontsize=20
    matplotlib.rc('font', size=fontsize)
    seed = 0
    norm_reg_b = []
    norm_hyp_b = []
    norm_reg_z_b = []
    norm_reg_a = []
    norm_hyp_a = []
    norm_reg_z_a = []
    norm_mult = []
    dirr = "results/"
    envs_name =  ["HalfCheetah-Vel", "Ant-Vel", "HalfCheetah-Vel-Medium","HalfCheetah-Dir","Ant-Dir"]
    test = '_test'
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for i,env in enumerate(["HalfCheetahVelBullet-v0", "AntVelBullet-v0","HalfCheetahVelHardBullet-v0","HalfCheetahDirBullet-v0", "AntDirBullet-v0"]):
          args.hyper_file = env + "_hyper_maml_"
          args.regtask_file = env + "_regular_maml_task_"
          args.reg_file = env + "_regular_maml_"
          args.multi_file = env + "_multi_task__hyper_maml_"

          a0 = np.load(dirr  + args.reg_file + "0" + test + "_before_rewards.npy")
          a1 = np.load(dirr + args.reg_file + "1" + test + "_before_rewards.npy")
          a2 = np.load(dirr + args.reg_file + "2" + test + "_before_rewards.npy")
          b0 = np.load(dirr + args.reg_file + "0" + test + "_after_rewards.npy")
          b1 = np.load(dirr + args.reg_file + "1" + test + "_after_rewards.npy")
          b2 = np.load(dirr + args.reg_file + "2" + test + "_after_rewards.npy")
          
          c0 = np.load(dirr + args.hyper_file + "0" + test + "_before_rewards.npy")
          c1 = np.load(dirr + args.hyper_file + "1" + test + "_before_rewards.npy")
          c2 = np.load(dirr + args.hyper_file + "2" + test + "_before_rewards.npy")
          d0 = np.load(dirr + args.hyper_file + "0" + test + "_after_rewards.npy")
          d1 = np.load(dirr + args.hyper_file + "1" + test + "_after_rewards.npy")
          d2 = np.load(dirr + args.hyper_file + "2" + test + "_after_rewards.npy")
          
          f0 = np.load(dirr + args.regtask_file + "0" + test + "_before_rewards.npy")
          f1 = np.load(dirr + args.regtask_file + "1" + test + "_before_rewards.npy")
          f2 = np.load(dirr + args.regtask_file + "2" + test + "_before_rewards.npy")
          g0 = np.load(dirr + args.regtask_file + "0" + test + "_after_rewards.npy")
          g1 = np.load(dirr + args.regtask_file + "1" + test + "_after_rewards.npy")
          g2 = np.load(dirr + args.regtask_file + "2" + test + "_after_rewards.npy")

          e0 = np.load(dirr + args.multi_file + "0" + test + "_before_rewards.npy")
          e1 = np.load(dirr + args.multi_file + "1" + test + "_before_rewards.npy")
          e2 = np.load(dirr + args.multi_file + "2" + test + "_before_rewards.npy")

      #    if env in "AntDirBullet-v0":
      #      total = np.vstack([a1 ,a2])
      #    else:
          total = np.vstack([a0,a1 ,a2])
          min_a = gaussian_filter1d(np.min(total,axis=0), sigma=2)
          max_a = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_a = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
    #      if env in "AntDirBullet-v0":
    #        total = np.vstack([ b1 ,b2])
    #      else:
          total = np.vstack([b0, b1 ,b2])
          min_b = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_b = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_b = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          
          total = np.vstack([c0, c1, c2])
          min_c = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_c = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_c = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          total = np.vstack([d0, d1, d2])
          min_d = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_d = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_d = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          
          total = np.vstack([e0, e1 ,e2])
          min_e = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_e = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_e = gaussian_filter1d(np.mean(total, axis=0),sigma=2)

          if env in "AntDirBullet-v0":
            total = np.vstack([f0[:437],f1[:437], f2[:437]])
            total1 = np.vstack([f1[437:],f2[437:]])
            min_f = np.concatenate([np.min(total,axis=0),np.min(total1,axis=0)])
            max_f = np.concatenate([np.max(total,axis=0),np.max(total1,axis=0)])
            avg_f = np.concatenate([np.mean(total,axis=0),np.mean(total1,axis=0)])
          else:
          
            total = np.vstack([f0,f1 ,f2])
            min_f = gaussian_filter1d(np.min(total,axis=0),sigma=2)
            max_f = gaussian_filter1d(np.max(total, axis=0),sigma=2)
            avg_f = gaussian_filter1d(np.mean(total, axis=0),sigma=2)

          if env in "AntDirBullet-v0":
            total = np.vstack([g0[:437],g1[:437], g2[:437]])
            total1 = np.vstack([g1[437:],g2[437:]])
            min_g = np.concatenate([np.min(total,axis=0),np.min(total1,axis=0)])
            max_g = np.concatenate([np.max(total,axis=0),np.max(total1,axis=0)])
            avg_g = np.concatenate([np.mean(total,axis=0),np.mean(total1,axis=0)])
          else:
            total = np.vstack([g0,g1 ,g2])
            min_g = gaussian_filter1d(np.min(total,axis=0),sigma=2)
            max_g = gaussian_filter1d(np.max(total, axis=0),sigma=2)
            avg_g = gaussian_filter1d(np.mean(total, axis=0),sigma=2)


          base = avg_b[0]
          top = avg_b[-1] - base

          norm_reg_b.append((avg_a - base) / top)
          norm_hyp_b.append((avg_c - base) / top)
          norm_reg_z_b.append((avg_f - base) / top)
          norm_reg_a.append((avg_b - base) / top)
          norm_hyp_a.append((avg_d - base) / top)
          norm_reg_z_a.append((avg_g - base) / top)
          norm_mult.append((avg_e - base) / top)

    norm_max_hyp_b = np.max(norm_hyp_b,axis=0)
    norm_min_hyp_b = np.min(norm_hyp_b,axis=0)
    norm_mean_hyp_b = np.mean(norm_hyp_b,axis=0)
    norm_max_reg_b = np.max(norm_reg_b,axis=0)
    norm_min_reg_b = np.min(norm_reg_b,axis=0)
    norm_mean_reg_b = np.mean(norm_reg_b,axis=0)
    norm_max_mult = np.max(norm_mult,axis=0)
    norm_min_mult = np.min(norm_mult,axis=0)
    norm_mean_mult = np.mean(norm_mult,axis=0)
    norm_max_reg_z_b = np.max(norm_reg_z_b,axis=0)
    norm_min_reg_z_b = np.min(norm_reg_z_b,axis=0)
    norm_mean_reg_z_b = np.mean(norm_reg_z_b,axis=0)
    
    norm_max_hyp_a = np.max(norm_hyp_a,axis=0)
    norm_min_hyp_a = np.min(norm_hyp_a,axis=0)
    norm_mean_hyp_a = np.mean(norm_hyp_a,axis=0)
    norm_max_reg_a = np.max(norm_reg_a,axis=0)
    norm_min_reg_a = np.min(norm_reg_a,axis=0)
    norm_mean_reg_a = np.mean(norm_reg_a,axis=0)
    norm_max_reg_z_a = np.max(norm_reg_z_a,axis=0)
    norm_min_reg_z_a = np.min(norm_reg_z_a,axis=0)
    norm_mean_reg_z_a = np.mean(norm_reg_z_a,axis=0)
    
    ax.set_xlabel("iteration")
    ax.set_ylabel("norm reward")
    ax.plot(range(len(norm_max_reg_b)), norm_mean_reg_b, label="MAML before adapt",color='cyan',linewidth=3)
    ax.plot(range(len(norm_max_reg_a)), norm_mean_reg_a, label="MAML", color='yellow',linewidth=3)
    ax.plot(range(len(norm_max_reg_z_b)), norm_mean_reg_z_b, label="Contex MAML before adapt", color='black',linewidth=3)
    ax.plot(range(len(norm_max_reg_z_a)), norm_mean_reg_z_a, label="Contex MAML", color='m',linewidth=3)
    ax.plot(range(len(norm_mean_hyp_b)), norm_mean_hyp_b, label="Hyper MAML before adapt", color='#1B2ACC',linewidth=3)
    ax.plot(range(len(norm_mean_hyp_a)), norm_mean_hyp_a, label="Hyper MAML (Ours)", color='#CC4F1B',linewidth=3)
    ax.plot(range(len(norm_mean_mult)), norm_mean_mult, label="Hyper TRPO multi task (Ours)",color='g',linewidth=3)
  #  ax.fill_between(range(len(norm_max_reg_a)), norm_min_reg_a, norm_max_reg_a,facecolor='lightyellow', alpha=0.3)
  #  ax.fill_between(range(len(norm_max_reg_z_a)), norm_min_reg_z_a, norm_max_reg_z_a, facecolor='#089FFF', alpha=0.3)
  #  ax.fill_between(range(len(norm_mean_hyp_a)), norm_min_hyp_a, norm_max_hyp_a, facecolor='#FF9848', alpha=0.3)
  #  ax.fill_between(range(len(norm_mean_mult)),norm_min_mult, norm_max_mult, facecolor='lightgreen', alpha=0.3)
    ax.grid(True)
    text = ax.text(-0.2,1.01, "", transform=ax.transAxes)
    lgd = ax.legend(loc='upper center',ncol=2, bbox_to_anchor=(0.5,-0.17),prop={'size': 20})
    
    plt.savefig("./normalize_MAML.pdf", bbox_extra_artists=(lgd,text), bbox_inches='tight')
    plt.close()  


def plot(args):
    fontsize=20
    matplotlib.rc('font', size=20)
    for test in ['_test', '']:
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(20, 10)
        gs = fig.add_gridspec(2, 6)
        ax1 = fig.add_subplot(gs[0,:2])
        ax2 = fig.add_subplot(gs[0,2:4])
        ax3 = fig.add_subplot(gs[0,4:])
        ax4 = fig.add_subplot(gs[1,:2],sharex=ax1)
        ax5 = fig.add_subplot(gs[1,2:4],sharex=ax2)
        ax6 = fig.add_subplot(gs[1,4:])
        ax6.axis('off')
        axs = fig.axes
        
      #  fig, axs = plt.subplots(2, 3,constrained_layout=True, figsize=(15,10), sharex=True)
        envs_name =  ["HalfCheetah-Vel", "Ant-Vel", "HalfCheetah-Vel-Medium","HalfCheetah-Dir","Ant-Dir"]
        dirr = "results/"
      
        for i,env in enumerate(["HalfCheetahVelBullet-v0", "AntVelBullet-v0","HalfCheetahVelHardBullet-v0","HalfCheetahDirBullet-v0", "AntDirBullet-v0"]):
          args.hyper_file = env + "_hyper_maml_"
          args.regtask_file = env + "_regular_maml_task_"
          args.reg_file = env + "_regular_maml_"
          args.multi_file = env + "_multi_task__hyper_maml_"
          last1 = 500
          last2 = 500
          last3 = 500
          last4 = 500
          last5 = 500
          a0 = np.load(dirr  + args.reg_file + "0" + test + "_before_rewards.npy")[:last1]
          a1 = np.load(dirr + args.reg_file + "1" + test + "_before_rewards.npy")[:last1]
          a2 = np.load(dirr + args.reg_file + "2" + test + "_before_rewards.npy")[:last1]
          b0 = np.load(dirr + args.reg_file + "0" + test + "_after_rewards.npy")[:last2]
          b1 = np.load(dirr + args.reg_file + "1" + test + "_after_rewards.npy")[:last2]
          b2 = np.load(dirr + args.reg_file + "2" + test + "_after_rewards.npy")[:last2]
          
          c0 = np.load(dirr + args.hyper_file + "0" + test + "_before_rewards.npy")[:last3]
          c1 = np.load(dirr + args.hyper_file + "1" + test + "_before_rewards.npy")[:last3]
          c2 = np.load(dirr + args.hyper_file + "2" + test + "_before_rewards.npy")[:last3]
          d0 = np.load(dirr + args.hyper_file + "0" + test + "_after_rewards.npy")[:last4]
          d1 = np.load(dirr + args.hyper_file + "1" + test + "_after_rewards.npy")[:last4]
          d2 = np.load(dirr + args.hyper_file + "2" + test + "_after_rewards.npy")[:last4]
          
          f0 = np.load(dirr + args.regtask_file + "0" + test + "_before_rewards.npy")[:last3]
          f1 = np.load(dirr + args.regtask_file + "1" + test + "_before_rewards.npy")[:last3]
          f2 = np.load(dirr + args.regtask_file + "2" + test + "_before_rewards.npy")[:last3]
          g0 = np.load(dirr + args.regtask_file + "0" + test + "_after_rewards.npy")[:last4]
          g1 = np.load(dirr + args.regtask_file + "1" + test + "_after_rewards.npy")[:last4]
          g2 = np.load(dirr + args.regtask_file + "2" + test + "_after_rewards.npy")[:last4]

          e0 = np.load(dirr + args.multi_file + "0" + test + "_before_rewards.npy")[:last5]
          e1 = np.load(dirr + args.multi_file + "1" + test + "_before_rewards.npy")[:last5]
          e2 = np.load(dirr + args.multi_file + "2" + test + "_before_rewards.npy")[:last5]

          axs[i].set_title(envs_name[i],fontsize=fontsize+2)

  #        if env in "AntDirBullet-v0":
  #          total = np.vstack([a1 ,a2])
  #        else:
          total = np.vstack([a0,a1 ,a2])
          min_a = gaussian_filter1d(np.min(total,axis=0), sigma=2)
          max_a = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_a = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
   #       if env in "AntDirBullet-v0":
    #        total = np.vstack([ b1 ,b2])
     #     else:
          total = np.vstack([b0, b1 ,b2])
          min_b = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_b = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_b = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          
          total = np.vstack([c0, c1, c2])
          min_c = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_c = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_c = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          total = np.vstack([d0, d1, d2])
          min_d = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_d = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_d = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
          
          total = np.vstack([e0, e1 ,e2])
          min_e = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_e = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_e = gaussian_filter1d(np.mean(total, axis=0),sigma=2)

  #        if env in "AntDirBullet-v0":
  #          total = np.vstack([f1 ,f2])
  #        else:
          total = np.vstack([f0,f1 ,f2])
          min_f = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_f = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_f = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
  #        if env in "AntDirBullet-v0":
  #          total = np.vstack([g1 ,g2])
  #        else:
          total = np.vstack([g0,g1 ,g2])
          min_g = gaussian_filter1d(np.min(total,axis=0),sigma=2)
          max_g = gaussian_filter1d(np.max(total, axis=0),sigma=2)
          avg_g = gaussian_filter1d(np.mean(total, axis=0),sigma=2)


            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
          
          axs[i].plot(range(len(avg_a)), avg_a, label="MAML before adapt", color='g')
          axs[i].plot(range(len(avg_b)), avg_b, label="MAML", color='#1B2ACC',linewidth=2.0)
          axs[i].plot(range(len(avg_f)), avg_f, label="Context MAML before adapt",color='cyan',linewidth=2.0)
          axs[i].plot(range(len(avg_g)), avg_g, label="Context MAML", color='black',linewidth=2.0)
          axs[i].plot(range(len(avg_c)), avg_c, label="Hyper MAML before adapt", color='m',linewidth=2.0)
          axs[i].plot(range(len(avg_d)), avg_d, label="Hyper MAML", color='#CC4F1B',linewidth=2.0)
          axs[i].plot(range(len(avg_e)), avg_e, label="Hyper TRPO multi task",color='y',linewidth=2.0)
          axs[i].fill_between(range(len(avg_g)), min_g, max_g,facecolor='black', alpha=0.3)
          axs[i].fill_between(range(len(avg_b)), min_b, max_b, facecolor='#089FFF', alpha=0.3)
          axs[i].fill_between(range(len(avg_d)), min_d, max_d, facecolor='#FF9848', alpha=0.3)
          axs[i].fill_between(range(len(avg_e)), min_e, max_e, facecolor='lightyellow', alpha=0.3)
          axs[i].grid(True)
          
          if axs[i].is_last_row():
                axs[i].set_xlabel('iteration')
      #    else:
      #          plt.setp(axs[i].get_xticklabels(), visible=False)

          if axs[i].is_first_col():
                axs[i].set_ylabel('reward')
     #     else:
     #           plt.setp(axs[i].get_yticklabels(), visible=False)


        ax3.set_xlabel('iteration')
        h,l=axs[0].get_legend_handles_labels()
        axs[-1].legend(h,l,prop={'size': 25},loc='center')
        plt.savefig('MAML' + test+".pdf")
        plt.close()  

def single_plot(args):
      fontsize=20
      matplotlib.rc('font', size=fontsize)
      fig = plt.figure(figsize=(10,7))
      env_name = "HalfCheetah-Vel-Medium"
      dirr = "results/"
      env = "HalfCheetahVelHardBullet-v0"
      test = "_test"
      args.hyper_file = env + "_hyper_maml_"
      args.reg_file = env + "_regular_maml_task_"

      a0t = np.load(dirr  + args.reg_file + "0" + test + "_before_rewards.npy")
      a1t = np.load(dirr + args.reg_file + "1" + test + "_before_rewards.npy")
      a2t = np.load(dirr + args.reg_file + "2" + test + "_before_rewards.npy")
      b0t = np.load(dirr + args.reg_file + "0" + test + "_after_rewards.npy")
      b1t = np.load(dirr + args.reg_file + "1" + test + "_after_rewards.npy")
      b2t = np.load(dirr + args.reg_file + "2" + test + "_after_rewards.npy")
      
      c0t = np.load(dirr + args.hyper_file + "0" + test + "_before_rewards.npy")
      c1t = np.load(dirr + args.hyper_file + "1" + test + "_before_rewards.npy")
      c2t = np.load(dirr + args.hyper_file + "2" + test + "_before_rewards.npy")
      d0t = np.load(dirr + args.hyper_file + "0" + test + "_after_rewards.npy")
      d1t = np.load(dirr + args.hyper_file + "1" + test + "_after_rewards.npy")
      d2t = np.load(dirr + args.hyper_file + "2" + test + "_after_rewards.npy")

      a0 = np.load(dirr  + args.reg_file + "0" + "_before_rewards.npy")
      a1 = np.load(dirr + args.reg_file + "1" +  "_before_rewards.npy")
      a2 = np.load(dirr + args.reg_file + "2" +  "_before_rewards.npy")
      b0 = np.load(dirr + args.reg_file + "0" + "_after_rewards.npy")
      b1 = np.load(dirr + args.reg_file + "1" +  "_after_rewards.npy")
      b2 = np.load(dirr + args.reg_file + "2" +  "_after_rewards.npy")

      c0 = np.load(dirr + args.hyper_file + "0" +"_before_rewards.npy")
      c1 = np.load(dirr + args.hyper_file + "1" + "_before_rewards.npy")
      c2 = np.load(dirr + args.hyper_file + "2" + "_before_rewards.npy")
      d0 = np.load(dirr + args.hyper_file + "0" + "_after_rewards.npy")
      d1 = np.load(dirr + args.hyper_file + "1" + "_after_rewards.npy")
      d2 = np.load(dirr + args.hyper_file + "2" +  "_after_rewards.npy")

      total = np.vstack([a0,a1 ,a2])
      min_a = gaussian_filter1d(np.min(total,axis=0), sigma=2)
      max_a = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_a = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      total = np.vstack([b0, b1 ,b2])
      min_b = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_b = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_b = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      total = np.vstack([a0t, a1t, a2t])
      min_at = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_at = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_at = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      total = np.vstack([b0t, b1t, b2t])
      min_bt = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_bt= gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_bt = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      
      total = np.vstack([c0,c1 ,c2])
      min_c = gaussian_filter1d(np.min(total,axis=0), sigma=2)
      max_c = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_c = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      
      total = np.vstack([d0, d1 ,d2])
      min_d = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_d = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_d = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      total = np.vstack([c0t, c1t, c2t])
      min_ct = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_ct = gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_ct = gaussian_filter1d(np.mean(total, axis=0),sigma=2)
      total = np.vstack([d0t, d1t, d2t])
      min_dt = gaussian_filter1d(np.min(total,axis=0),sigma=2)
      max_dt= gaussian_filter1d(np.max(total, axis=0),sigma=2)
      avg_dt = gaussian_filter1d(np.mean(total, axis=0),sigma=2)

    #  plt.plot(range(len(avg_a)), avg_a, label="MAML",color='cyan')
      plt.plot(range(len(avg_b)), avg_b, label="Context MAML", color='black',linewidth=2.5)
    #  plt.plot(range(len(avg_at)), avg_at, label="Test MAML",color='g')
      plt.plot(range(len(avg_bt)), avg_bt, label="Test Context MAML", color='y',linewidth=2.5)
    #  plt.plot(range(len(avg_c)), avg_c, label="Hyper MAML", color='m')
      plt.plot(range(len(avg_d)), avg_d, label="Hyper MAML", color='#1B2ACC',linewidth=2.5)
    #  plt.plot(range(len(avg_ct)), avg_ct, label="Test Hyper MAML", color='y')
      plt.plot(range(len(avg_dt)), avg_dt, label="Test Hyper MAML", color='#CC4F1B',linewidth=2.5)
      plt.fill_between(range(len(avg_b)), min_b, max_b,facecolor='black', alpha=0.3)
      plt.fill_between(range(len(avg_bt)), min_bt, max_bt, facecolor='lightyellow', alpha=0.3)
      plt.fill_between(range(len(avg_d)), min_d, max_d,facecolor='#1B2ACC', alpha=0.3)
      plt.fill_between(range(len(avg_dt)), min_dt, max_dt, facecolor='#CC4F1B', alpha=0.3)

      plt.grid(True)
      

      plt.xlabel('iteration')
      plt.ylabel('reward')
  
      plt.legend(prop={'size': 19})
      plt.savefig("HalfCheetahVelMediumMAML.pdf")
      plt.close()  

if __name__ == "__main__":
    	
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_file", type=str)				# hyper net file
    parser.add_argument("--reg_file", type=str)					# regular net file
    parser.add_argument("--env", type=str)					    # env
    args = parser.parse_args()
   # plot(args)
   # single_plot(args)
    grad_plot(args)
   # normlaize_plot()
