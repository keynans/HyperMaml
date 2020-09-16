
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
import numpy as np
        
class HalfCheetahDirEnv(HalfCheetahBulletEnv):
    """
    half cheetah bullet env with fwd back
    forward: x=1e3, y=0
    backword: x=-1e3, y=0
    """
    def __init__(self, task={}):
        super(HalfCheetahDirEnv, self).__init__()
        self._task = task
        self.max = 1e3
        self.min = -1e3
        self.walk_target_x, self.walk_target_y = task.get('direction', (self.max ,0.0))
        self.robot.walk_target_x, self.robot.walk_target_y = task.get('direction', (self.max ,0.0))

    def sample_tasks(self, tasks, num_tasks):
        goals = np.random.choice(tasks, num_tasks)
        goals = goals.tolist()
        return goals

    def sample_unseen_task(self, tasks):
        #dir has no unsee task so return fwsd or bck
        direction = (2 * self.np_random.binomial(1, p=0.5) - 1) * self.max 
        unseen_task = [{'direction': (direction,0.0)}]
        return unseen_task

    def reset_task(self, task):
        self._task = task
        self.walk_target_x, self.walk_target_y = task['direction']
        self.robot.walk_target_x, self.robot.walk_target_y = task['direction']

    def step(self,action):
        s, r, done, data = HalfCheetahBulletEnv.step(self,action)
        yaw = self.robot.body_rpy[2]
        s[1] = 0.
        s[2] = yaw
        return s, r, done, data 

    def reset(self):
        state = HalfCheetahBulletEnv.reset(self)
        yaw = self.robot.body_rpy[2]
        state[1] = 0.
        state[2] = yaw
        return state

class HalfCheetahVelEnv(HalfCheetahBulletEnv):
    """
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a penalty equal to the 
    difference between its current velocity and the target velocity. The tasks 
    are generated by sampling the target velocities from the uniform 
    distribution on [0, 2].
    """
    def __init__(self, task={}):
        super(HalfCheetahVelEnv, self).__init__()
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        self.max = 3.0
        self.min = 0.0

    def sample_tasks(self, tasks, num_tasks):
        goals = np.random.choice(tasks, num_tasks)
        goals = goals.tolist()
        return goals
    
    def sample_unseen_task(self, tasks):
        velocity = self.np_random.uniform(self.min, self.max)
        velocities = [vel['velocity'] for vel in tasks]
        while velocity in velocities:
            velocity = self.np_random.uniform(self.min, self.max)
        unseen_task = [{'velocity': velocity}]
        return unseen_task


    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            xposbefore = self.robot.body_xyz[0]
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        forward_vel = (self.robot.body_xyz[0] - xposbefore) / self.robot.scene.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(a))
        reward = (forward_reward - ctrl_cost)
        self.reward += reward
  
        self.HUD(state, a, done)

        yaw = self.robot.body_rpy[2]
        state[1] = 0.
        state[2] = yaw

        return state, reward, bool(done), {}

    def reset(self):
        state = HalfCheetahBulletEnv.reset(self)
        yaw = self.robot.body_rpy[2]
        state[1] = 0.
        state[2] = yaw
        return state