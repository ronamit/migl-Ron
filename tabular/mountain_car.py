
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


#--------------------------------------------------------------------
# Random seed
#--------------------------------------------------------------------

np.random.seed(42) # for reproducible results

#--------------------------------------------------------------------
# Enviornment parmaters
#--------------------------------------------------------------------

P_MIN = -1.2
P_MAX = 0.6
V_MIN = -0.07
V_MAX = 0.07

P_GOAL = 0.5

maxSteps = 500

# discretize the state space:
deltaPos = 0.1
deltaVel = 0.01

nTilesP = int(np.ceil((P_MAX - P_MIN) / deltaPos))
nTilesV = int(np.ceil((V_MAX - V_MIN) / deltaVel))


# Actions:
actionSet = [-1,0,1]
nActions = len(actionSet)

#--------------------------------------------------------------------
# Algorithm paramters
#--------------------------------------------------------------------

# algName = 'SARSA'
algName = 'SARSA_LAMBDA'
# algName = 'SARSA_Kernel'

nEpisodes = 1000
epsilon = 0.1  # Exploration
gamma = 0.95  # Discount

lambdaParam = 0.9 # for SARSA(lambda)

# RBF Kernel (for SARSA_Kernel)
sizeG = 1 # Gaussian of size [2*sizeG+1 X 2*sizeG + 1] - note - effects running time
sigmmaG = 0.5 # STD of gaussian
#--------------------------------------------------------------------
#  Utility functions
#--------------------------------------------------------------------

def GetTileIndex(p,v):
    """ Return discrete tile index of a continuous state """
    pI = int( (p - P_MIN) // deltaPos)
    vI = int( (v - V_MIN) // deltaVel)
    return pI, vI


# Create Kernel
def CreateKernel():
    kernelSize = 2*sizeG+1
    kernel = np.zeros((kernelSize, kernelSize), dtype=np.float)
    kernelShifts = range(-sizeG, sizeG+1)
    for i1, shift1 in enumerate(kernelShifts):
        for i2, shift2 in enumerate(kernelShifts):
            kernel[i1,i2] = np.exp(-(shift1**2 + shift2**2)/(2*sigmmaG**2))
    kernel = kernel / np.linalg.norm(kernel) # normalize phi to have unit norm
    return kernel, kernelShifts

if algName == 'SARSA_Kernel':
    kernel, kernelShifts = CreateKernel()
    print(' Kernel: ' + kernel)

def InnerProdWithKernel(w, pG, vG, aG):
    prodOut = 0
    for i1,shift1 in enumerate(kernelShifts):
        for i2,shift2  in enumerate(kernelShifts):
            pInd = pG + shift1
            vInd = vG + shift2
            if (0 <= pInd < nTilesP) and (0 <= vInd < nTilesV):
                prodOut += w[pInd, vInd, aG] * kernel[i1,i2]
    return prodOut


def AddByKernel(w, pG, vG, aG, factor):
    for i1,shift1 in enumerate(kernelShifts):
        for i2,shift2  in enumerate(kernelShifts):
            pInd = pG + shift1
            vInd = vG + shift2
            if (0 <= pInd < nTilesP) and (0 <= vInd < nTilesV):
                w[pInd, vInd, aG] += kernel[i1,i2] * factor
    return w

#--------------------------------------------------------------------
# Dynamics
#--------------------------------------------------------------------

def mountainCarSim(p, v, u):
    """ Take current state (p,v) and action u to and outputs the next state (pNext, vNext) """

    vNext = v + 0.001 * u - 0.0025 * np.cos(3 * p)

    # Apply boundary
    vNext = min(max(vNext, V_MIN), V_MAX)

    pNext = p + vNext

    # Apply boundary
    pNext = min(max(pNext, P_MIN), P_MAX)

    # Inelastic wall on the left side
    if pNext <= P_MIN:
        vNext = 0.0

    return pNext, vNext


# --------------------------------------------------------------------
#  GetAction
# --------------------------------------------------------------------
def  GetActionInd(w, pI, vI, exploreFlag):
    Q_per_action = w[pI, vI, :]
    epsilonFlag =  (np.random.uniform(0,1) <= epsilon) # true w.p epslion(np.all(Q_per_action == 0))
    sameQ = np.all(Q_per_action == Q_per_action[0], axis = 0) # If Q is the same for all actions, choose randomly

    if sameQ  or (exploreFlag and epsilonFlag):
        aI = np.random.randint(nActions)
    else:
        aI = np.argmax(Q_per_action)

    return aI


# --------------------------------------------------------------------
#  GetReward
# --------------------------------------------------------------------
def  GetReward(pNext):
    if pNext > P_GOAL:
        r = 5
        goalReachFlag = True
    else:
        r = -1
        goalReachFlag = False
    return r, goalReachFlag

# --------------------------------------------------------------------
#  Run episode (+ SARSA learning) method
# --------------------------------------------------------------------
def RunEpisode(p, v, w, beta, learnFlag, exploreFlag):

    pTrajectory = np.empty((maxSteps, 1), dtype=np.float)
    pTrajectory.fill(np.nan)

    # Init the eligibility trace:
    elTrace = np.zeros((nTilesP, nTilesV, nActions), dtype=np.float)

    iStep = 0
    goalReachFlag = False
    totalReward = 0
    # Run episode:
    while (iStep < maxSteps and not goalReachFlag):

        (pI, vI) = GetTileIndex(p,v)
        # Find current action:
        aI = GetActionInd(w, pI, vI, exploreFlag)
        u = actionSet[aI]

        # Find next state and next action:
        (pNext, vNext) = mountainCarSim(p, v, u)
        (pNextI, vNextI) = GetTileIndex(pNext, vNext)

        aNextI = GetActionInd(w, pNextI, vNextI, exploreFlag=False)
        # Note: we don't explore on on the next action, we only explore on the current action ot update different state-action pairs

        # Get current reward:
        (r, goalReachFlag) = GetReward(pNext)

        # Learning Step:
        if learnFlag:
            if algName == 'SARSA':
                # since phi(s,a) is an indicator, only one element of w is changed
                delta_t = r + gamma * w[pNextI, vNextI, aNextI] - w[pI, vI, aI]
                w[pI, vI, aI] += beta * delta_t

            elif  algName == 'SARSA_LAMBDA':
                # Based on Sutton 7.5
                # since phi(s,a) is an indicator, only one element of w is changed
                delta_t = r + gamma * w[pNextI, vNextI, aNextI] - w[pI, vI, aI]
                elTrace[pI, vI, aI] += 1
                w += beta * delta_t * elTrace
                elTrace *= gamma * lambdaParam

            elif  algName == 'SARSA_Kernel':
                # Based on Sutton 7.5
                phi_dot_w = InnerProdWithKernel(w, pI, vI, aI)
                phiNext_dot_w = InnerProdWithKernel(w, pNextI, vNextI, aNextI)
                delta_t = r + gamma * phiNext_dot_w - phi_dot_w
                w = AddByKernel(w, pI, vI, aI, factor= beta * delta_t)


            else:
                raise ValueError('Unknown algName')

        pTrajectory[iStep] = p
        p, v = pNext, vNext
        totalReward += r
        iStep += 1
    nSteps = iStep
    return w, totalReward, goalReachFlag, pTrajectory, nSteps


def PlotTrajectory(w, axes, iPlot):
    ''' Trajectory of car ( p vs. time) for the greedy policy starting from 0, 0 , '''
    p, v = 0, 0
    _, _, _, pTrajectory, nSteps = RunEpisode(p, v, w, beta=0, learnFlag=False, exploreFlag=False)
    ax1 = axes[iPlot % len(axes)]
    ax1.clear()
    ax1.plot(range(nSteps), pTrajectory[:nSteps], label='Episode : {0}'.format(iEpisode))
    ax1.set_title('Ep. {0}'.format(iEpisode))
    ax1.grid(True)
    ax1.set_xlabel('time')
    ax1.set_ylabel('Pos')
    iPlot += 1
    return iPlot
# --------------------------------------------------------------------
# Run episodes
# --------------------------------------------------------------------
# Inits:
vTotalReward = np.zeros((nEpisodes,1), dtype=np.float)
vIsGoalReached = np.zeros((nEpisodes,1), dtype=np.bool)
vWeightsNorm = np.zeros((nEpisodes,1), dtype=np.float)

# Figure for trajectories:
fig, axes = plt.subplots(nrows=4, ncols=3,  sharex = True)
axes = axes.flatten()
iPlot = 0

# Initial Weights
w = np.zeros((nTilesP,nTilesV, nActions), dtype=np.float)
print('Running {} learning episodes with {} ...'.format(nEpisodes,  algName))

for iEpisode in range(nEpisodes):

    # Initial state:
    p = np.random.uniform(-0.5, 0.2)
    v = np.random.uniform(-0.02, 0.02)

    # learning rate:
    beta = 100 / (1000 + iEpisode)

    # Run episode:
    w, totalReward, goalReachFlag, _, _ = RunEpisode(p, v, w, beta, learnFlag=True, exploreFlag=True)
    vTotalReward[iEpisode] = totalReward
    vIsGoalReached[iEpisode] = goalReachFlag
    vWeightsNorm[iEpisode] = np.linalg.norm(w) # L2 norm

    # Plot car trajectory:
    if (iEpisode % 100  == 0):
        iPlot = PlotTrajectory(w, axes, iPlot)

print('Done')

# --------------------------------------------------------------------
# Evaluate final performance
# --------------------------------------------------------------------
nEval = 1000
evalReward = np.zeros((nEval,1), dtype=np.float)
evalGoalFlag =  np.zeros((nEval,1), dtype=np.bool)
evalTimeToGoal = np.zeros((nEval,1), dtype=np.float)

print('Running {} evaluation episodes...'.format(nEval))
for iEpisode in range(nEval):
    # Initial state:
    p = np.random.uniform(-0.5, 0.2)
    v = np.random.uniform(-0.02, 0.02)

    # Run episode:
    _, totalReward, goalReachFlag, _, nSteps = RunEpisode(p, v, w, beta=0, learnFlag=False, exploreFlag=False)

    evalReward[iEpisode] = totalReward
    evalGoalFlag[iEpisode] = goalReachFlag
    evalTimeToGoal[iEpisode] = nSteps


meanTotalReward= np.mean(evalReward[evalGoalFlag])
goalReachRate = np.mean(evalGoalFlag)
meanTimeToGoal = np.mean(evalTimeToGoal)

print('Algorithm: ', algName, ' number of episodes: ', nEpisodes)
print('The mean success rate of the final policy:', goalReachRate)
print('The mean time to goal (for successful episodes) of the final policy:', meanTimeToGoal)
print('The mean reward of the final policy:', meanTotalReward)

# --------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------

# Fine-tune figure; make subplots farther from each other.
fig.subplots_adjust(hspace=0.3, wspace=0.3)

plt.figure()
plt.plot(range(nEpisodes), vTotalReward, 'g.')
plt.title('Total Reward in Episdoe vs. Episode')
plt.grid(True)
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.figure()
plt.plot(range(nEpisodes), vIsGoalReached, 'b.')
plt.title('Is Goal Reached vs. Episode')
plt.grid(True)
plt.xlabel('Episode')
plt.ylabel('Is Goal Reached')


plt.figure()
plt.plot(range(nEpisodes), vWeightsNorm)
plt.title('Weights L2 Norm vs. Episode')
plt.grid(True)
plt.xlabel('Episode')
plt.ylabel('Norm')


# Plot the value.
statesValue = np.max(w, axis = 2, keepdims=False) # V(s) =  Q(s,pi(s)) where pi(s) =  argmax_a Q(s,a)

# Check if a state s was visited (then the Q(s,a) should be updated and non-constant)
isVisited = (np.argmin(w, axis = 2) != np.argmax(w, axis = 2))


fig = plt.figure()
ax = fig.gca(projection='3d')
pVec = np.linspace(P_MIN, P_MAX, nTilesP)
vVec = np.linspace(V_MIN, V_MAX, nTilesV)
vGrid, pGrid  = np.meshgrid(vVec, pVec)
statesValue[~isVisited] = np.min(statesValue)
surf = ax.plot_surface(pGrid, vGrid, statesValue, cmap=cm.viridis, linewidth=0, antialiased=False)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.suptitle('State Value')
# ax.set_zlim(-1.01, 1.01) # Customize the z axis.
fig.colorbar(surf, shrink=0.5, aspect=5, label='mean reward') # Add a color bar which maps values to colors


# Plot the policy
policyPerState = np.argmax(w, axis = 2) # pi(s) =  argmax_a Q(s,a)
# Set a unique symbol to state with undefined policy
policyPerState[~isVisited] = 3

cmap = colors.ListedColormap(['red', 'black', 'green', 'white']) # make a color map of fixed colors
ticks=[0,1,2,3]
extent = [P_MIN, P_MAX, V_MIN, V_MAX]
fig = plt.figure()
plt.imshow(policyPerState, interpolation='nearest', aspect='auto', cmap=cmap, extent = extent)
plt.suptitle('Final  Policy')
plt.xlabel('Position')
plt.ylabel('Velocity')
cbar = plt.colorbar(cmap=cmap, ticks=ticks, orientation='horizontal')
cbar.ax.set_xticklabels( ['Accelarate -1', 'Accelarate 0', 'Accelarate +1', 'Undefined'])  # horizontal colorbar


plt.figure()
plt.imshow(np.array(isVisited, dtype=np.float), interpolation='nearest', aspect='auto', extent = extent, cmap=plt.get_cmap('gray', 2))
plt.suptitle('Is state-action updated?')
plt.xlabel('Position')
plt.ylabel('Velocity')
cbar = plt.colorbar(ticks=[0,1], orientation='horizontal')
cbar.ax.set_xticklabels( ['No', 'Yes'])  # horizontal colorbar



plt.show()


