# Code to check how Cost varies along space and time

import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib import animation

# %% Paramteters from Shadden Physica D
A = .1
eps = .25
omega = 2*np.pi/10 

# %% Define model and parameters.
xgrid = 90
ygrid = 45
#nparl = 20 # number of Parallel threads (can't exceed 20 here)
Delta = .1
Nt = 30
#nloops = int(ygrid/nparl) # number of loops
Nage = xgrid*ygrid # Number of agents
#Bage = int(xgrid) # Number of agents per batch (one row of agents)
Bage = 1

xpos,ypos = np.meshgrid(np.linspace(0,2,xgrid), np.linspace(0,1,ygrid))
xpos, ypos = xpos.flatten(), ypos.flatten()

Nx = 2*Bage+1
Nu = 2*Bage

def dgyre(x, u, A = A, eps = eps, om = omega):
    """Continuous-time ODE model."""
    
    a = []
    b = []
    f = []
    df = []
    for i in range(0, Bage):
        a += [eps * np.sin(om * x[2*Bage])];
        b += [1 - 2 * a[i]];
    
        f += [a[i] * x[2*i]**2 + b[i] * x[2*i]];
        df += [2 * a[i] * x[2*i] + b[i]];
        
    dxdt = []
    for i in range(0, 2*Bage, 2):

        dxdt += [
        -np.pi * A * np.sin(np.pi * f[int(i/2)]) * np.cos(np.pi * x[i+1]) + u[i],
        np.pi * A * np.cos(np.pi * f[int(i/2)]) * np.sin(np.pi * x[i+1]) * df[int(i/2)] + u[i+1],
               ]
    
    dxdt+=[1]
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
ocean = mpc.DiscreteSimulator(dgyre, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(dgyre, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)

# Define stage cost and terminal weight.
Q = 1*np.eye(Nx)
Q[-1, -1] = 0
R = 80*np.eye(Nu)

goal_abs = np.array([.5, .5])
goal = np.tile(goal_abs, Bage)
goal = np.concatenate((goal, np.array([0])))

def lfunc(x,u):
    """Standard quadratic stage cost."""
    return (mpc.mtimes(u.T, R, u) + mpc.mtimes((x-goal).T, Q, (x-goal)))
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")


# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -.1*np.ones((Nu,))}
ub = {"u" : .1*np.ones((Nu,))}

def get_trajs(idx, tstart):
    
    xi = np.array([xpos[idx], ypos[idx]])
    xi = xi.flatten('F')
    xi = np.append(xi, [tstart])
    #xi = np.array([2 ,1 ,2 , .9, 1.9 ,1, 2 ,.8 , 0])
    N = {"x":Nx, "u":Nu, "t":Nt}
    solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=xi, lb=lb, ub=ub,
                  verbosity=-1)

## %% Now simulate.
    x = np.zeros((2,Nx))
    x[0,:] = xi
    pred = []
    upred = [] 

    # Fix initial state.
    #solver.fixvar("x", 0, x[0,:])  
    
    # Solve nlp.
    solver.solve()   
    
    # Print stats.
    solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1])
    # calling solver variables. Can pull out predicted trajectories from here.
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    x[1,:] = xi
    
        # Retrieving trajectory predictions
    pred2 = []
    pred3 = []    
    for i in pred:
        pred2 = []
        for j in i:
            temp = np.array(j)
            pred2 += [temp]
        pred3 += [pred2]
    
    pred3 = np.array(pred3)[0,:,:,0]
    
    # Retrieving energy predictions
    upred2 = []
    upred3 = []    
    for i in upred:
        upred2 = []
        for j in i:
            temp = np.array(j)
            upred2 += [temp]
        upred3 += [upred2]
    
    upred3 = np.array(upred3)[0,:,:,0]
    return [x[0,:], pred3, upred3]
# %% The parallel loop

results = []

dt3 = .2
for tstep in np.arange(0, 10+dt3, dt3):
    single_sim = []
    pool = mp.Pool(20) # We have 20 cores totally
    single_sim += pool.starmap(get_trajs, [(idx, tstep) for idx in range(0, Nage, 1)])
    pool.close()
    print('tstep '+ str(tstep)+ ' done')
    
    results += [single_sim]
    
#%%
import pickle

with open('rq60_nt10_gp5p5.pkl', 'wb') as f:
    pickle.dump(results, f)
    
#%%

dt3 = .2

fname = 'rq80_nt30_gp5p5.pkl'
import pickle
with open('/Users/kartikkrishna/Downloads/bigdata/control_vec_flds/' + fname, 'rb') as f:
    results2 = pickle.load(f)

#results = results + results2
results = results2
#%% Load LCS Data
recall = np.load('/Users/kartikkrishna/Documents/Code/swarmLCS/lcs-au19win20/ftle_th6.npz')
data = [recall[key] for     key in recall]
x0, y0, solfor, solbac = data
    
    
#%% forward trajectory Animation


fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))
ax.scatter(goal[0], goal[1])

particles = []
trajs = []
fortrajs = []

for i in range(Nage):
    particle, = ax.plot([], [], marker='o', linewidth = 0, color='black')
    #traj, = ax.plot([],[], color='grey', alpha=0.2)
    fortraj, = ax.plot([],[], color='green', alpha=0.6, linestyle = '--') # forward trajectory
    
    particles += particle,
    #trajs += traj,
    fortrajs += fortraj,


def update(num):
    
    ax.collections = []
    ax.contour(x0, y0, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x03, y03,(solfor1[num]), origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x03, y03, solbac1[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    ax.contour(x0, y0, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('t = ' + str(int(num*dt3)))
    
    for i in range(0, Nage, 250):
        particles[i].set_data(results[num][i][0][0], results[num][i][0][1])
        #trajs[int(i/2)].set_data(x[:num+1, i],x[:num+1, i+1])
        fortrajs[i].set_data(results[num][i][1][:,0], results[num][i][1][:,1])
      
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
    
    return Q, particles, trajs, fortrajs

anim = animation.FuncAnimation(fig,update,interval=100, blit=False)

#%% Storing control for grid

temp3 = []
for k in range(len(results)):
    temp1 = results[k]
    temp2 = []
    for l in range(len(temp1)):
        temp2 += [temp1[l][2]]
    temp3 += [temp2]

temp3 = np.array(temp3)

en_pred = (temp3[:,:,:,0]**2 + temp3[:,:,:,1]**2)**.5
en_pred = np.sum(en_pred, axis = 2)*dt3
en_pred = en_pred.reshape(-1,ygrid,xgrid)

#%% State error on grid

temp6 = []
for k in range(len(results)):
    temp4 = results[k]
    temp5 = []
    for l in range(len(temp1)):
        temp5 += [temp4[l][1]]
    temp6 += [temp5]

temp6 = np.array(temp6)

state_pred = (temp6[:,:,:,0] - goal[0])**2 + (temp6[:,:,:,1]-goal[1])**2
state_pred = np.sum(state_pred, axis = 2)*dt3
state_pred = state_pred.reshape(-1,ygrid,xgrid)

#%% Plotting costs along horizon

fig2 = plt.figure()

def update2(num):
    plt.title('t = ' + str(int(num*dt3)))
    plt.imshow(state_pred[num], origin = 'lower')


anim = animation.FuncAnimation(fig2,update2,interval=10, blit=False)

#%% Plotting the control vector field
u_x = temp3[:,:,0,0] ; u_y = temp3[:,:,0,1]
u_x = u_x.reshape(-1, ygrid,xgrid); u_y = u_y.reshape(-1, ygrid,xgrid)

#x01 = np.linspace(-2,4,xgrid); y01 = np.linspace(-1,2,ygrid)
x01 = np.linspace(0,2,xgrid); y01 = np.linspace(0,1,ygrid)
x01, y01 = np.meshgrid(x01,y01)

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

#qui = ax.quiver(x01,y01,u_x[0], u_y[0], color = 'grey')

def animate(i):
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot

    #qui = ax.quiver(x01,y01,u_x[i], u_y[i], color = 'grey')
    ax.streamplot(x01,y01,u_x[i], u_y[i], density=1,arrowsize=1,color ='grey')
    ax.contour(x0, y0, solfor[i], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[i], origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('MPC $T_H = $' + str(Nt*Delta) + ', $t_0$ = ' + str(round(i*dt3,2)))
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)

    #return qui

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=False, repeat=False)

#%% 
from scipy.interpolate import RegularGridInterpolator

x02 = x01[0]
y02 = y01[:,0]
t02 = np.arange(0,10.2,.2)

# Control is 0 outside domain!
U_X = RegularGridInterpolator((t02, y02, x02), u_x, bounds_error = False, fill_value = 0)
U_Y = RegularGridInterpolator((t02, y02, x02), u_y, bounds_error = False, fill_value = 0)

def ctrl_doublegyreVEC(t, yin, A, eps, om):
    
    sh = np.shape(yin[0]) 
    x = yin[0].flatten()
    y = yin[1].flatten()
    t = np.zeros(len(x)) + t%10
    
    pts = np.array([t,y,x]).T
    
    u = np.zeros(x.shape); 
    v = u.copy()
    
    a = eps * np.sin(omega * t);
    b = 1 - 2 * a;
    
    f = a * x**2 + b * x;
    df = 2 * a * x + b;
    
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y) + U_X(pts)
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df + U_Y(pts)
    
    # u = U_X(pts)
    # v = U_Y(pts)
    
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

##%% Part 1 - Initialize grid of particles through vector field

dx = .005 #try .005
xvec = np.arange(0.0, 2, dx)
yvec = np.arange(0, 1, dx)

x03, y03 = np.meshgrid(xvec, yvec)
yIC = np.zeros((2, len(yvec), len(xvec)))
yIC[0], yIC[1] = x03, y03


# %% Calculate FTLE

dt = 0.025;  # timestep for integrator (try .005)
dt2 = .2 # timestep for frame
Tinf = 15;     # duration of integration (maybe use 15) (forward time horiz)
Tinb = 15;     # duration of integration (maybe use 15) (backward time horiz)
T = 10 # total time over which simulation runs

solfor1 = []
solbac1 = []

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
    
    # Backward time LCS
    yin_bac = yIC
    
    for i in np.arange(0+m, -Tinb+m, -dt):
        yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),-dt,i,yin_bac)
        yin_bac = yout
    
    xT = yin_bac[0]
    yT = yin_bac[1]

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
            sigma[j,i] = (1./Tinb) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solbac1 += [sigma]
    
    print("Time = " + str(m))
    
#%% Calculate FTLE cost function

dt = 0.025;  # timestep for integrator (try .005)
dt2 = .2 # timestep for frame
Tinf = 6;     # duration of integration (maybe use 15) (forward time horiz)
Tinb = 6;     # duration of integration (maybe use 15) (backward time horiz)
T = 10 # total time over which simulation runs

solfor1 = []
solbac1 = []

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tinf+m, dt):
        yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
        yin_for = yout
    
    xT = yin_for[0]
    yT = yin_for[1]
    sigma = (xT-goal[0])**2 + (yT-goal[1])**2
    solfor1 += [sigma]
    
    # Backward time LCS
    yin_bac = yIC
    
    for i in np.arange(0+m, -Tinb+m, -dt):
        yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),-dt,i,yin_bac)
        yin_bac = yout
    
    xT = yin_bac[0]
    yT = yin_bac[1]
    sigma = (xT-goal[0])**2 + (yT-goal[1])**2
    
    solbac1 += [sigma]
    
    print("Time = " + str(m))
    
#%% find delta J

# xT = []
# yT = []

# #all_xT = []
# #all_yT = []
# for m in np.arange(0, T, dt2):
    
#     # Forward time LCS
#     yin_for = yIC
    
#     tempx = []
#     tempy = []
#     for i in np.arange(0+m, Tinf+m, dt):
#         yout = rk4singlestep(lambda t, y: ctrl_doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
#         yin_for = yout
#         #tempx += [yin_for[0]]
#         #tempy += [yin_for[1]]
    
#     #all_xT += [np.array(tempx)]
#     #all_yT += [np.array(tempy)]
#     xT += [yin_for[0]]
#     yT += [yin_for[1]]
#     print("Time = " + str(m))

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
#xvec5 = np.arange(1.5, 1.75, dx_p)
xvec5 = np.arange(.5, .75, dx_p)
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
#%%
# xT = np.array(xT); yT = np.array(yT)

# dxTx = xT[:,1:,:] - xT[:,:-1,:]; dyTx = yT[:,1:,:] - yT[:,:-1,:]

# lmbTx = xT - goal[0]; lmbTy = xT - goal[1]

# dJx = (lmbTx[:,:-1,:]*dxTx + lmbTy[:,:-1,:]*dyTx)


#%%

# all_xT = np.array(all_xT)
# all_yT = np.array(all_yT)

# blah = all_xT[0][0]
# plt.imshow(all_xT[0][0])

#%% Plot a lot of things
plt.style.use('default')
fig2= plt.figure()

ax2 = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

particles1 = []
particles2 = []

for k in range(len(xvec4)*len(yvec4)):
        particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='purple')
        particles1 += particle1,
    
for k in range(len(xvec5)*len(yvec5)):
        particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='yellow')
        particles2 += particle2,
        
def animate2(i):
    ax2.collections = [] # clear lines streamplot
    ax2.patches = [] # clear arrowheads streamplot
    
    m = 0
    for j in range(0, len(yvec4)):
        for k in range(0, len(xvec4)):
            particles1[m].set_data(all_xT1[i,j,k], all_yT1[i,j,k])
            m+=1
            
    m = 0
    for j in range(0, len(yvec5)):
        for k in range(0, len(xvec5)):
            particles2[m].set_data(all_xT2[i,j,k], all_yT2[i,j,k])
            m+=1
        
    #i = 10*i
    #plt.imshow((all_xT[4][i] - goal[0])**2 + (all_yT[0][i] - goal[1])**2, extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    #plt.imshow(lmbTy[i]**2 + lmbTy[i]**2, extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    #plt.imshow(lmbTy[i], extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    #plt.imshow((-dJx[i]), extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    
    
    #plt.imshow(solfor1[0], extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    
    #qui = ax.quiver(x01,y01,u_x[i], u_y[i], color = 'grey')
    #stream = ax.streamplot(x01,y01,u_x[i], u_y[i], density=3,arrowsize=1,color ='grey')
    #ax2.contour(x03, y03, (solfor1[i]), origin = 'lower', cmap = 'winter', alpha = 1)
    
    #plt.imshow(solfor1[i%20], extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    ax2.contour(x03, y03, ((solfor1[i%50])), origin = 'lower', cmap = 'winter', alpha = 1)
    ax2.contour(x03, y03, (solbac1[i%50]), origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('time = ' + str(i*dt2))
    #i = int(i*2.5)
    plt.imshow(state_pred[i], extent = (0, 2, 0, 1), origin = 'lower', cmap = 'Spectral')
    #plt.imshow(en_pred[i], extent = (0, 2, 0, 1), origin = 'lower', cmap = 'viridis')
    #ax2.contour(x0, y0, solfor[i], origin = 'lower', cmap = 'summer', alpha = 1)
    #ax2.contour(x0, y0, solbac[i], origin = 'lower', cmap = 'spring', alpha = 1)

    #plt.title('time = ' + str(i*dt))
    
        
    ax2.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
    ax2.set_xlim(0,2)
    ax2.set_ylim(0,1)
    
    #return stream
    #return qui

anim =   animation.FuncAnimation(fig2, animate2, interval=2000, blit=False, repeat=False)

#%% Save data
#np.savez('450trace.npz', x, goal, trajs, fortrajs, R)
