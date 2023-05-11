# LCS vs varying R/Q ratio or T_H

import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib import animation

#%% Variables

A = .1
eps = .25
omega = 2*np.pi/10

# Define model and parameters.
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

goal_abs = np.array([.5, .5])
goal = np.tile(goal_abs, Bage)
goal = np.concatenate((goal, np.array([0])))

# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -.1*np.ones((Nu,))}
ub = {"u" : .1*np.ones((Nu,))}

#%%
full_res = []
for par in range(0,11):
    # Paramteters from Shadden Physica D
    A = .1
    eps = .25
    omega = 2*np.pi/10
    
    # Define model and parameters.
    xgrid = 90
    ygrid = 45
    #nparl = 20 # number of Parallel threads (can't exceed 20 here)
    Delta = .1
    Nt = 10
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
    R = np.eye(Nu)*par*10
    
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
    # The parallel loop
    
    results = []
    
    dt3 = .2
    for tstep in np.arange(0, 10+dt3, dt3):
        single_sim = []
        pool = mp.Pool(20) # We have 20 cores totally
        single_sim += pool.starmap(get_trajs, [(idx, tstep) for idx in range(0, Nage, 1)])
        pool.close()
        print('param ' + str(par)+ ',tstep '+ str(tstep)+ ' done')
        
        results += [single_sim]
    
    full_res += [results]
    
# Store variables

import pickle

with open('rq0t100s10_sweep_th1_gp5p5.pkl', 'wb') as f:
     pickle.dump(full_res, f)

#%%
dt3 = .2

fname = 'rq0t100s10_sweep_th3_gp5p5.pkl'
import pickle
with open('/Users/kartikkrishna/Downloads/bigdata/rq_vs_lcs/' + fname, 'rb') as f:
    full_res = pickle.load(f)

#%% Load LCS Data
recall = np.load('/Users/kartikkrishna/Documents/Code/swarmLCS/lcs-au19win20/ftle_th15.npz')
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
    tslice = 0
    ax.collections = []
    ax.contour(x0, y0, solfor[tslice], origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x03, y03,(solfor1[num]), origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x03, y03, solbac1[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    ax.contour(x0, y0, solbac[tslice], origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('R/Q ' + str(10*int(num+1)))
    
    for i in range(0, Nage, 100):
        particles[i].set_data(full_res[num][tslice][i][0][0], full_res[num][tslice][i][0][1])
        #trajs[int(i/2)].set_data(x[:num+1, i],x[:num+1, i+1])
        fortrajs[i].set_data(full_res[num][tslice][i][1][:,0], full_res[num][tslice][i][1][:,1])
      
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
    
    return particles, trajs, fortrajs

anim = animation.FuncAnimation(fig,update,interval=100, blit=False)

#%% Storing control for grid
full_en_pred = []
ctrl_fields = []
for num in range(len(full_res)):
    temp3 = []
    for k in range(len(full_res[0])):
        temp1 = full_res[num][k]
        temp2 = []
        for l in range(len(temp1)):
            temp2 += [temp1[l][2]]
        temp3 += [temp2]
    
    temp3 = np.array(temp3)
    ctrl_fields += [temp3]
    
    en_pred = (temp3[:,:,:,0]**2 + temp3[:,:,:,1]**2)**.5
    en_pred = np.sum(en_pred, axis = 2)*dt3
    en_pred = en_pred.reshape(-1,ygrid,xgrid)
    
    full_en_pred += [en_pred]

#%% State error on grid

full_er_pred = []
for num in range(len(full_res)):
    temp6 = []
    for k in range(len(full_res[0])):
        temp4 = full_res[num][k]
        temp5 = []
        for l in range(len(temp1)):
            temp5 += [temp4[l][1]]
        temp6 += [temp5]
    
    temp6 = np.array(temp6)
    
    state_pred = (temp6[:,:,:,0] - goal[0])**2 + (temp6[:,:,:,1] - goal[1])**2 
    state_pred = np.sum(state_pred, axis = 2)*dt3
    state_pred = state_pred.reshape(-1,ygrid,xgrid)
    
    full_er_pred += [state_pred]

#%% Plotting costs along horizon

fig2 = plt.figure()

def update2(num):
    tslice = 0
    plt.title('Energy spent for R/Q = ' + str(10*int(num+1)) + ', t = ' + str(tslice*dt3))
    plt.imshow(full_en_pred[num][tslice], origin = 'lower')
    #plt.imshow(full_er_pred[num][tslice] + (num+1)*10*full_en_pred[num][tslice], origin = 'lower')

anim = animation.FuncAnimation(fig2,update2,interval=1000, blit=False)

#%% storing the control vector field

u_xs = []
u_ys = []

for i in range(len(full_res)):
    u_x = ctrl_fields[i][:,:,0,0] ; u_y = ctrl_fields[i][:,:,0,1]
    u_x = u_x.reshape(-1, ygrid,xgrid); u_y = u_y.reshape(-1, ygrid,xgrid)
    u_xs += [u_x]
    u_ys += [u_y]

#x01 = np.linspace(-2,4,xgrid); y01 = np.linspace(-1,2,ygrid)
x01 = np.linspace(0,2,xgrid); y01 = np.linspace(0,1,ygrid)
x01, y01 = np.meshgrid(x01,y01)

#%%Plotting the control law
fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

#qui = ax.quiver(x01,y01,u_x[0], u_y[0], color = 'grey')

def animate(i):
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot

    #qui = ax.quiver(x01,y01,u_x[i], u_y[i], color = 'grey')
    ax.streamplot(x01,y01,u_xs[i][0], u_ys[i][0], density=1,arrowsize=1,color ='grey')
    ax.contour(x0, y0, solfor[0], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[0], origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('MPC $R/Q = $' + str((i+1)*10) + ', $t_0$ = 0')
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)

    #return qui

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=5000, blit=False, repeat=False)

#%%Plotting  for a single R/Q ratio
fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

#qui = ax.quiver(x01,y01,u_x[0], u_y[0], color = 'grey')

def animate(i):
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot

    #qui = ax.quiver(x01,y01,u_x[i], u_y[i], color = 'grey')
    ax.streamplot(x01,y01,u_xs[2][i], u_ys[2][i], density=1,arrowsize=1,color ='grey')
    ax.contour(x0, y0, solfor[i], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[i], origin = 'lower', cmap = 'autumn', alpha = 1)
    plt.title('MPC $R/Q = 0$' + ', $t_0$ = ' + str((i*dt3)))
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)

    #return qui

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=False, repeat=False)
#%% FTLE

tslice = 0
from scipy.interpolate import RegularGridInterpolator
ftle_vs_rq = []
for rqs in range(11):
    x02 = x01[0]
    y02 = y01[:,0]
    t02 = np.arange(0,10.2,.2)
    
    # Control is 0 outside domain!
    U_X = RegularGridInterpolator((t02, y02, x02), u_xs[rqs], bounds_error = False, fill_value = 0)
    U_Y = RegularGridInterpolator((t02, y02, x02), u_ys[rqs], bounds_error = False, fill_value = 0)
    
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
    
    
    # Calculate FTLE
    
    dt = 0.025;  # timestep for integrator (try .005)
    dt2 = .2 # timestep for frame
    Tinf = 15;     # duration of integration (maybe use 15) (forward time horiz)
    Tinb = 0;     # duration of integration (maybe use 15) (backward time horiz)
    T = .2 # total time over which simulation runs
    
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
        
        print("Time = " + str(m) + ', Param = ' + str(rqs))
    
    ftle_vs_rq += [solfor1]
    
#%%

fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

#qui = ax.quiver(x01,y01,u_x[0], u_y[0], color = 'grey')

def animate(i):
    tslice = 0
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot
    skip=(slice(None,None,2),slice(None,None,2))
    #qui = ax.quiver(x01[skip],y01[skip],u_xs[i+1][tslice][skip] - u_xs[i][tslice][skip], u_ys[i+1][tslice][skip] - u_ys[i][tslice][skip], color = 'grey', scale = .5)
    #ax.streamplot(x01[skip],y01[skip],u_xs[i+1][tslice][skip] - u_xs[i][tslice][skip], u_ys[i+1][tslice][skip] - u_ys[i][tslice][skip], density=1,arrowsize=1,color ='grey')
    #ax.imshow((u_xs[i+1][tslice][skip] - u_xs[i][tslice][skip])**2 + (u_ys[i+1][tslice][skip] - u_ys[i][tslice][skip])**2, origin = 'lower', extent = (0,2,0,1))
    
    ax.contour(x0, y0, solfor[tslice], origin = 'lower', cmap = 'summer', alpha = 1)
    ax.contour(x03, y03, ((ftle_vs_rq[i][tslice])), origin = 'lower', cmap = 'winter', alpha = 1)
    #ax.contour(x0, y0, solbac[tslice], origin = 'lower', cmap = 'autumn', alpha = 1)
    #plt.title('$\Delta U$, MPC $T_H = $' + str(Nt*Delta) + ', $t_0$ = ' + str(round(tslice*dt3,2)) + ', R/Q = ' + str(18+(i+1)))
    
    #plt.contourf(x01,y01,full_er_pred[i][tslice], origin = 'lower')
    #plt.imshow(full_er_pred[i][tslice], origin = 'lower', extent = (0,2,0,1))
    #plt.imshow(full_er_pred[i+1][tslice] - full_er_pred[i][tslice], origin = 'lower', extent = (0,2,0,1))
    # qui = ax.quiver(x01[skip],y01[skip],u_xs[i][tslice][skip], u_ys[i][tslice][skip], scale = 2,color ='grey')
    #ax.streamplot(x01[skip],y01[skip],u_xs[i][tslice][skip], u_ys[i][tslice][skip], density=1,arrowsize=1,color ='grey')
    
    plt.title('$\Delta U$, MPC $T_H = $' + str(Nt*Delta) + ', $t_0$ = ' + str(round(tslice*dt3,2)) + ', R/Q = ' + str(10*(i)))
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)

    #return qui

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=400, blit=False, repeat=False)

#%%

fig = plt.figure()
ax2 = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))

for i in range(len(ftle_vs_rq)):
    ax2.contour(x03, y03, (ftle_vs_rq[i][tslice]), origin = 'lower', cmap = 'winter', vmin = (.07*i),vmax = (.07*(i+1)), alpha = .1*i)
    #ax2.contourf(x03, y03, np.log(ftle_vs_rq[i][tslice]), origin = 'lower', cmap = 'winter', alpha =1)
    
ax2.contour(x0, y0, solfor[tslice], origin = 'lower', cmap = 'winter', alpha = 1)

#%%

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
