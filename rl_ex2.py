#Hopping into RL with double gyre

import gym
import numpy as np

from gym.spaces import Box
 

def doublegyreVEC(t, yin, ctrl, A, eps, om):
    x = yin[0]
    y = yin[1]
    u_x = ctrl[0]
    u_y = ctrl[1]

    a = eps * np.sin(om * t);
    b = 1 - 2 * a;
    
    f = a * x**2 + b * x;
    df = 2 * a * x + b;

    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y) + .1*u_x;
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df + .1*u_y;

    return np.array([u,v])

def rk1singlestep(fun,dt,t0,y0, u0):
    f1 = fun(t0,y0,u0);

    yout = y0 + (dt)*(f1)
    return yout


class Ocean(gym.Env):
    def __init__(self, start = [1,1,0], goal = [0.5,0.5], dt = .1, A = .1, eps = .25, om = 2*np.pi/10, R = 70):
        
        self.action_space = Box(low = np.array([-1, -1]), high = np.array([1, 1]), dtype=np.float32)
        self.observation_space = Box(low = np.array([0., 0, 0]), high = np.array([1, 1, 1]), dtype = np.float32)
        
        self.goal = goal
        self.start = start
        self.dt = dt
        
        self.A = A
        self.eps = eps
        self.om = om
        self.ubtime = 0
        
        self.R = R
        
        
    def step(self, u):
       x_pos = 2*self.state[0]
       y_pos = self.state[1]
       t_inst = 10*self.state[2]

       dt = self.dt
       goal = self.goal
       A = self.A
       eps = self.eps
       om = self.om
       
       done = False
       #self.last_u = u  # for rendering
       costs = (((2*x_pos-goal[0])**2 + (y_pos-goal[1])**2) + self.R*((.1*u[0]**2)+(.1*u[1]**2))) 

       yout = rk1singlestep(lambda t, y, u: doublegyreVEC(t,y,u,A,eps,om),dt,t_inst,np.array([x_pos, y_pos]),u)
       
       self.ubtime += dt
       
       # if ((2*x_pos-goal[0])**2 + (y_pos-goal[1])**2) < .0001:
       #     costs -= 20000
       #     done = True
           
       # if x_pos<0 or x_pos>2 or y_pos<0 or y_pos>1:
       #     costs += 30000
       #     done = True
           
       if self.ubtime>80:
           done = True
           
       self.state = np.array([yout[0]/2, yout[1], ((t_inst+dt)/10)%1], dtype = np.float32)

       return self.state, -costs, done, {}

    def render(self):
        pass
    
    def reset(self, rdom = True):
        if rdom == False:
            start = self.start
        
        else:
            start = [np.random.random(), np.random.random(), 0]
            
        self.state = np.array(start, dtype = np.float32)
        self.ubtime = 0
        return self.state
    

#%%
from stable_baselines3.common.env_checker import check_env

env = Ocean()
check_env(env)
#%%
ep_trajs = []
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    traj = []
    
    while not done:
        #env.render()
        action = env.action_space.sample() 
        n_state, reward, done, info = env.step(action)
        traj += [n_state]
        score+=reward
    
    ep_trajs += [np.array(traj)]
    print('Episode:{} Score:{}'.format(episode, score))

#%%

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(-0, 1))

ep = 2
ax.scatter(2*ep_trajs[ep][0,0], ep_trajs[ep][0,1])
ax.plot(2*ep_trajs[ep][:,0], ep_trajs[ep][:,1])

#%%
# from stable_baselines3 import A2C

# #model = PPO('MlpPolicy',env,verbose=1,tensorboard_log = log_path)

# model = A2C('MlpPolicy', env, verbose=1,gamma = .6)
#%%
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, buffer_size= 200000,learning_starts=10000, gamma = .98)
#%%
model.learn(total_timesteps=200000, log_interval=10)

#%%
#model.learn(total_timesteps = 5000000)
#%%
env = Ocean()

ep_trajs = []
ep_acs = []
episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0 
    traj = []
    acs = []
    
    while not done:
        #env.render()
        action, _ = model.predict(obs, deterministic=True)
        n_state, reward, done, info = env.step(action)
        obs = n_state
        traj += [n_state]
        acs += [action]
        score+=reward
    
    ep_trajs += [np.array(traj)]
    ep_acs += [np.array(acs)]
    print('Episode:{} Score:{}'.format(episode, score))
    
#%%
   
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

ep = 4
ax.scatter(2*ep_trajs[ep][0,0], ep_trajs[ep][0,1])
ax.plot(2*ep_trajs[ep][:-1,0], ep_trajs[ep][:-1,1])
plt.quiver(2*ep_trajs[ep][:-1,0], ep_trajs[ep][:-1,1], ep_acs[ep][:-1,0], ep_acs[ep][:-1,1])

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

ep = 2
ax.scatter(2*ep_trajs[ep][0,:,0], ep_trajs[ep][0,:,1])
ax.plot(2*ep_trajs[ep][:-1,:,0], ep_trajs[ep][:-1,:,1])
plt.quiver(2*ep_trajs[ep][:-1,:,0], ep_trajs[ep][:-1,:,1], ep_acs[ep][:-1,:,0], ep_acs[ep][:-1,:,1])


    
#%%
recall = np.load('/Users/kartikkrishna/Documents/Code/swarmLCS/lcs-au19win20/ftle_th15.npz')
data = [recall[key] for     key in recall]
x01, y01, solfor, solbac = data


#%%
##%% Part 1 - Initialize grid of particles through vector field

dx = .005 #try .005
xvec = np.arange(0.0, 2.0, dx)
yvec = np.arange(0, 1, dx)

x03, y03 = np.meshgrid(xvec, yvec)
yIC = np.zeros((2, len(yvec), len(xvec)))
yIC[0], yIC[1] = x03, y03

def ctrl_VEC(t, yin):
    
    sh = np.shape(yin[0]) 
    x = yin[0].flatten()
    y = yin[1].flatten()
    t = np.zeros(len(x)) + t%10
    
    pts = np.array([x/2,y,t/10]).T
    
    u = np.zeros(x.shape); 
    v = u.copy()
    
    u = .1*model.predict(pts)[0][:,0]
    v = .1*model.predict(pts)[0][:,1]
    
    u = u.reshape(sh[0],sh[1])
    v = v.reshape(sh[0],sh[1])

    return np.array([u,v])




#%%

def ctrl_doublegyreVEC(t, yin, A, eps, om):
    
    sh = np.shape(yin[0]) 
    x = yin[0].flatten()
    y = yin[1].flatten()
    t = np.zeros(len(x)) + t%10
    
    pts = np.array([x/2,y,t/10]).T
    
    U = .1*model.predict(pts)[0]
    
    u = np.zeros(x.shape); 
    v = u.copy()
    
    a = eps * np.sin(om * t);
    b = 1 - 2 * a;
    
    f = a * x**2 + b * x;
    df = 2 * a * x + b;
    
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y) + U[:,0]
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df + U[:,1]
    
    u = u.reshape(sh[0],sh[1])
    v = v.reshape(sh[0],sh[1])

    return np.array([u,v])

def rk4singlestep(fun,dt,t0,y0):
    f1 = fun(t0,y0);
    f2 = fun(t0+dt/2,y0+(dt/2)*f1);
    f3 = fun(t0+dt/2,y0+(dt/2)*f2);
    f4 = fun(t0+dt,y0+dt*f3);

    yout = y0 + (dt/6)*(f1+2*f2+2*f3+f4)
    return yout

# %% Calculate FTLE

dt = 0.025;  # timestep for integrator (try .005)
dt2 = .2 # timestep for frame
Tinf = 10;     # duration of integration (maybe use 15) (forward time horiz)
Tinb = 15;     # duration of integration (maybe use 15) (backward time horiz)
T = 10 # total time over which simulation runs

solfor1 = []
solbac1 = []


A= .1; eps = .1; omega=2*np.pi/10

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tinf+m, dt):
        yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
        yin_for = yout
    
    xT = yin_for[0]
    yT = yin_for[1]

    # Finite difference to compute the gradient
    dxTdx0, dxTdy0 = np.gradient(xT, dx, dx)
    dyTdx0, dyTdy0 = np.gradient(yT, dx, dx)

    D = np.eye(2)
    sigma = xT.copy()*0
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            D[0,0] = dxTdx0[j,i];
            D[0,1] = dxTdy0[j,i];
            D[1,0] = dyTdx0[j,i];
            D[1,1] = dyTdy0[j,i];
            sigma[j,i] = abs((1./Tinf)) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solfor1 += [sigma]
    
    # # Backward time LCS
    # yin_bac = yIC
    
    # for i in np.arange(0+m, -Tinb+m, -dt):
    #     yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),-dt,i,yin_bac)
    #     yin_bac = yout
    
    # xT = yin_bac[0]
    # yT = yin_bac[1]

    # # Finite difference to compute the gradient
    # dxTdx0, dxTdy0 = np.gradient(xT, dx, dx)
    # dyTdx0, dyTdy0 = np.gradient(yT, dx, dx)

    # D = np.eye(2)
    # sigma = xT.copy()*0
    # for i in range(len(xvec)):
    #     for j in range(len(yvec)):
    #         D[0,0] = dxTdx0[j,i];
    #         D[0,1] = dxTdy0[j,i];
    #         D[1,0] = dyTdx0[j,i];
    #         D[1,1] = dyTdy0[j,i];
    #         sigma[j,i] = (1./Tinb) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    # sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    # solbac1 += [sigma]
    
    print("Time = " + str(m))

#%% Integrate patches through

dx_p = .005 #try .005
xvec4 = np.arange(1, 1.25, dx_p)
yvec4 = np.arange(0.6, .8, dx_p)

x04, y04 = np.meshgrid(xvec4, yvec4)
yIC = np.zeros((2, len(yvec4), len(xvec4)))
yIC[0], yIC[1] = x04, y04

all_xT1 = []
all_yT1 = []

yin_for = yIC

tempx = []
tempy = []
c = 0
for i in np.arange(0, 100, dt):
    yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
    yin_for = yout
    c+=1
    if c%int(dt2/dt) == 0:
        tempx += [yin_for[0]]
        tempy += [yin_for[1]]

all_xT1 += [np.array(tempx)]
all_yT1 += [np.array(tempy)]

all_xT1, all_yT1= np.array(all_xT1)[0], np.array(all_yT1)[0]

# second patch
xvec5 = np.arange(1.5, 1.75, dx_p)
yvec5 = np.arange(0.4, 0.6, dx_p)

x04, y04 = np.meshgrid(xvec5, yvec5)
yIC = np.zeros((2, len(yvec5), len(xvec5)))
yIC[0], yIC[1] = x04, y04

all_xT = []
all_yT = []

yin_for = yIC

tempx = []
tempy = []
c = 0
for i in np.arange(0, 100, dt):
    yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
    yin_for = yout
    c+=1
    if c%int(dt2/dt) == 0:
        tempx += [yin_for[0]]
        tempy += [yin_for[1]]

all_xT += [np.array(tempx)]
all_yT += [np.array(tempy)]

all_xT2, all_yT2 = np.array(all_xT)[0], np.array(all_yT)[0]


#%% Calculate FTLE cost function

dt = 0.025;  # timestep for integrator (try .005)
dt2 = .2 # timestep for frame
Tinf = 15;     # duration of integration (maybe use 15) (forward time horiz)
Tinb = 6;     # duration of integration (maybe use 15) (backward time horiz)
T = 10 # total time over which simulation runs

goal = env.goal

solfor2 = []
solbac2 = []

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tinf+m, dt):
        yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
        yin_for = yout
    
    xT = yin_for[0]
    yT = yin_for[1]
    sigma = (xT-goal[0])**2 + (yT-goal[1])**2
    solfor2 += [sigma]
    
    # Backward time LCS
    # yin_bac = yIC
    
    # for i in np.arange(0+m, -Tinb+m, -dt):
    #     yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),-dt,i,yin_bac)
    #     yin_bac = yout
    
    # xT = yin_bac[0]
    # yT = yin_bac[1]
    # sigma = (xT-goal[0])**2 + (yT-goal[1])**2
    
    # solbac1 += [sigma]
    
    print("Time = " + str(m))
    
#%%
from matplotlib import animation

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

#qui = ax.quiver(x01,y01,u_x[0], u_y[0], color = 'grey')



def animate(i):
    
    # ax.clear()
    # particles1 = []
    # particles2 = []

    # for k in range(len(xvec4)*len(yvec4)):
    #         particle1, = ax.plot([], [], marker='o', linewidth = 0, color='purple')
    #         particles1 += particle1,
        
    # for k in range(len(xvec5)*len(yvec5)):
    #         particle2, = ax.plot([], [], marker='o', linewidth = 0, color='yellow')
    #         particles2 += particle2,
            
    # m = 0
    # for j in range(0, len(yvec4)):
    #     for k in range(0, len(xvec4)):
    #         particles1[m].set_data(all_xT1[i,j,k], all_yT1[i,j,k])
    #         m+=1
            
    # m = 0
    # for j in range(0, len(yvec5)):
    #     for k in range(0, len(xvec5)):
    #         particles2[m].set_data(all_xT2[i,j,k], all_yT2[i,j,k])
    #         m+=1
    
    #ax.collections = [] # clear lines streamplot
    #ax.patches = [] # clear arrowheads streamplot
    
    #i=10*i
    #ax.streamplot(x03,y03, ctrl_VEC(i*.2,yIC)[0],ctrl_VEC(i*.2,yIC)[1])
    
    #plt.imshow((ctrl_VEC(i*.2,yIC)[0]**2 + ctrl_VEC(i*.2,yIC)[1]**2)**1, extent = (0,2,0,1), origin = 'lower')
    #plt.imshow((solfor2[i])**.5, extent = (0,1.9,0,1), origin = 'lower', cmap = 'Spectral')
  
    #ax.contour(x01, y01, (solfor[i]), origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x01, y01, (solbac[i]), origin = 'lower', cmap = 'autumn', alpha = 1)
    
    
    #ax.contourf(x03, y03, np.log(solfor1[i%50]), origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x03, y03, (solbac1[i%50]), origin = 'lower', cmap = 'autumn', alpha = 1)
    
    #plt.imshow(np.log(solfor1[i%50]),origin = 'lower')
    plt.title('DDPG , $t$ = ' + str(.2*i) + ', R/Q = ' + str(70))
    ax.scatter(.5, .5, marker = 'X', color='green', s=90)

    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    #return qui

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=4, blit=False, repeat=False)

#%% Save RL model for recall later

#model.save('paperDG_RQ70_DDPG')

#%% Load model

#model = DDPG.load('/Users/kartikkrishna/Downloads/bigdata/RL_sb3/paperDG_RQ70_DDPG.zip')