import numpy as np
import math

####################
#     Paramters    #
####################
theta_scale = 180
d_scale = 200

time_scale = 7                                                                  #6+1  

sensor = np.array([17.09433584,   5.20951119,   3.20735776,   2.71438153,   2.57237491,   2.24333728])/100.0


####################
#     Varaible     #
####################
X = np.zeros((d_scale, theta_scale))                                            #state space, divided by 360*2000 grids

#for each grid at each moment, there are 3 values: current_d, cuttent_thetam, bel_value
D = np.zeros((time_scale, d_scale, theta_scale), dtype='float')                    #current_d at specific time and statespace
T = np.zeros((time_scale, d_scale, theta_scale), dtype='float')                    #current_theta at specific time and statespace
Bel = np.zeros((time_scale, d_scale, theta_scale), dtype='float')                  #current_Bel_value at specific time and statespace


P = np.zeros((time_scale, d_scale, theta_scale), dtype='float') 

#state space
d_border = 0.5

#########################
#     Help function     #
#########################
def next_probability(d_predict,sense):
  
    sigma = 0.2
      
    #sensor beam model
    res = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (sense - d_predict)**2 / (2 * sigma**2)) #sensor_value-d_predict
    
    return res



def next_d(d, theta):
    
    #parameter for motion
    r = 0.5
    delta_theta = 0.4
    
    #predict next d
    d_predict = d + r*math.cos(theta+delta_theta) - r*math.cos(theta)

    return d_predict

#########################
#         Main          #
#########################

#Initialize Bel-value at t=0
propability_init = 1.0/(d_scale*theta_scale)
Bel[0] +=propability_init

#Initialize current_d at t=0 for each grid
for i in range(d_scale):
    for j in range(theta_scale):
        D[0][i][j] = d_border*i/d_scale   #i-related 
        T[0][i][j] = 2*math.pi*j/theta_scale   #j-related    


#Compute each coordinate at different t
for t in range(time_scale-1):
                    
    for i in range(d_scale): 
        for j in range(theta_scale):
            D[t+1][i][j] = next_d(D[t][i][j], T[t][i][j])
            T[t+1][i][j] = T[t][i][j] + 0.4


#belief recursion
for t in range(time_scale-1):

    #normalize
    n = np.sum(Bel[t])
    n_inverse = 1.0/n
                    
    for i in range(d_scale): 
        for j in range(theta_scale):
        
            #prediction step
            #bel value does not change since action is unique
           
            #update step
            p = next_probability(D[t+1][i][j], sensor[t])
            Bel[t+1][i][j] = n_inverse * p *Bel[t][i][j]

            P[t][i][j] = p



print("Helloworld")

#print result and collect data for pyplot
x = []
y = []
z = []

print("In state space")
for t in range(1,7):
    #print("np.max",np.max(Bel[i]))
    #print(np.where(Bel[i] == np.max(Bel[i])))
    i_max, j_max = np.where(Bel[t] == np.max(Bel[t]))
    #print("    max i j Bel:",i_max, j_max, np.max(Bel[t]))
    print("    max i j Bel:",i_max * d_border/d_scale, j_max * 360/theta_scale, np.max(Bel[t])) 

    x.append(i_max * d_border/d_scale)
    y.append(j_max * 360/theta_scale )
    z.append(np.max(Bel[t]))


#pyplot-3d
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(x, y, z, label='trend of the most likely point in state space')
ax.scatter(x, y, z, c='r', label='the optimized points (d, theta, Bel) at each step')

ax.set_xlabel('d: 0~0.5 meter')
ax.set_ylabel('theta: 0~360 degree')
ax.set_zlabel('Bel value')
ax.legend()

plt.show()

#pyplot-2d
fig = plt.figure(1)
plt.subplot(211)  
plt.scatter(x,y, c='r',label='the optimized points (d, theta) at each step') 
ax.set_xlabel('d: 0~0.5 meter')
ax.set_ylabel('theta: 0~360 degree')
ax.legend()

plt.show()

#Question 1.2 result
print("result for q1.2:",D[5][150][51], T[5][150][51]*360/(2*math.pi))
