import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import matplotlib.animation as animation

#constants
Y=1.4
R=287
C_F_L=0.5

#generate the grid
nx=int(input('Enter number of grid points \n'))
delta_x=(3/(nx-1))


x=np.linspace(0,3,nx)
#Create nozzle shape
A=1+(2.2*(x-1.5)**2)


U_1=np.zeros(nx)
U_2=np.full(nx,0.59)	#initialize U_2 for use in V initialization

U_3=np.zeros(nx)

F_2=np.zeros(nx)
F_3=np.zeros(nx)
J_2=np.zeros(nx)
rho=np.zeros(nx)
V=np.zeros(nx)
T=np.zeros(nx)
a=np.zeros(nx)
T_U=np.zeros(nx) #Temperature in terms of U for calculation of speed of sound
delta_t=np.zeros(nx)
dU1_dt_n=np.zeros(nx)
dU2_dt_n=np.zeros(nx)
dU3_dt_n=np.zeros(nx)
dU1_dt_n1=np.zeros(nx)
dU2_dt_n1=np.zeros(nx)
dU3_dt_n1=np.zeros(nx)
dU1_dt_av=np.zeros(nx)
dU2_dt_av=np.zeros(nx)
dU3_dt_av=np.zeros(nx)
U_1p=np.zeros(nx)
U_2p=np.zeros(nx)
U_3p=np.zeros(nx)
U_1n=np.zeros(nx)
U_2n=np.zeros(nx)
U_3n=np.zeros(nx)
rho_p=np.zeros(nx)
V_p=np.zeros(nx)
T_p=np.zeros(nx)
F_1p=np.zeros(nx)
F_2p=np.zeros(nx)
F_3p=np.zeros(nx)
J_2p=np.zeros(nx)
rho_n=np.zeros(nx)
V_n=np.zeros(nx)
T_n=np.zeros(nx)
p=np.zeros(nx)


#initialize rho,T,V


for i in range(nx):
	if 0<=x[i]<=0.5:
		rho[i]=1
		T[i]=1
		V[i]=((U_2[i])/(rho[i]*A[i]))

	elif 0.5<x[i]<=1.5:
		rho[i]=1.0-0.366*(x[i]-0.5)

		T[i]=1.0-0.167*(x[i]-0.5)
		V[i]=((U_2[i])/(rho[i]*A[i]))
	elif 1.5<x[i]<=2.1:
		rho[i]=0.634-0.702*(x[i]-1.5)

		T[i]=0.833-0.4908*(x[i]-1.5)
		V[i]=((U_2[i])/(rho[i]*A[i]))
	elif 2.1<x[i]<=3.0:
		rho[i]=0.5892+0.10228*(x[i]-2.1)

		T[i]=0.93968+0.0622*(x[i]-2.1)
		V[i]=((U_2[i])/(rho[i]*A[i]))






#Initialize U
for i in range(nx):
	U_1[i]=rho[i]*A[i]
	
	U_3[i]=rho[i]*(((T[i]/(Y-1))+(Y*V[i]**2)/(2)))*A[i]
nt=int(input('Enter number of time steps :\n'))


def MCormack(U_1,U_2,U_3,rho,V,T,nt,nx,A):

	rho_all = []
	T_all = []
	p_all = []

	for n in range(nt):
		#initialize F,J vectors
		for i in range(nx):
			F_1=U_2.copy()
			F_2[i]=((U_2[i]**2)/(U_1[i]))+((Y-1)/(Y))*(U_3[i]-((Y/2)*(U_2[i]**2/U_1[i])))
			
			F_3[i]=Y*(U_2[i]*U_3[i])/U_1[i]-(Y*(Y - 1)/2)*(U_2[i]**3)/(U_1[i]**2)
		for i in range(nx-1):
			J_2[i]=((Y-1)/Y)*(U_3[i]-(Y/2)*(U_2[i]**2/U_1[i]))*((np.log(A[i+1])-np.log(A[i]))/delta_x)

		



#J[nx] or last point value of J must be found by using boundary conditions or maybe it isnt required

#Predictor step
		for i in range(nx-1):
			dU1_dt_n[i]=-((F_1[i+1]-F_1[i])/(delta_x))
			dU2_dt_n[i]=-((F_2[i+1]-F_2[i])/(delta_x))+J_2[i]
			dU3_dt_n[i]=-((F_3[i+1]-F_3[i])/(delta_x)) 

#we need time step value now
		for k in range(nx):
			a[k]=np.sqrt((Y*R*T[k]))
			delta_t[k]=C_F_L*(delta_x/(a[k]+V[k]))
		dt=10*min(delta_t)

#get predicted values of U

		for i in range(nx-1):

			U_1p[i]=U_1[i]+(dU1_dt_n[i]*dt)
			U_2p[i]=U_2[i]+(dU2_dt_n[i]*dt)
			U_3p[i]=U_3[i]+(dU3_dt_n[i]*dt)
#apply boundary conditions
		U_1p[0]=A[0]
		U_2p[0]=2*U_2p[1]-(U_2p[2])
		U_3p[0]=U_1p[0]*(1/(Y-1)+(Y/2)*((U_2p[0])/(U_1p[0]))**2)
		U_1p[nx-1]=2*U_1p[nx-2]-(U_1p[nx-3])
		U_2p[nx-1]=2*U_2p[nx-2]-(U_2p[nx-3])
		

#from this get predicted values of rho, T, V
		for i in range(nx):
			rho_p[i]=U_1p[i]/A[i]
			V_p[i]=U_2p[i]/U_1p[i]
			
		#apply U3 BC at exit
		U_3p[nx-1]=((0.6784*A[nx-1])/(Y-1))+((Y/2)*U_2p[nx-1]*V_p[nx-1])
		for i in range(nx):
			T_p[i]=(Y-1)*((U_3p[i]/U_1p[i])-((Y/2)*(U_2p[i]/U_1p[i])**2))
#get predicted values of F and J  vectors
		for i in range(nx):
			F_1p=U_2p.copy()
			F_2p[i]=((U_2p[i]**2)/(U_1p[i]))+((Y-1)/(Y))*(U_3p[i]-((Y/2)*(U_2p[i]**2/U_1p[i])))
			
			F_3p[i]=Y*(U_2p[i]*U_3p[i])/U_1p[i]-(Y*(Y - 1)/2)*(U_2p[i]**3)/(U_1p[i]**2)
		for i in range(1,nx):
			J_2p[i]=((Y-1)/Y)*(U_3p[i]-(Y/2)*(U_2p[i]**2/U_1p[i]))*((np.log(A[i])-np.log(A[i-1]))/delta_x)
#corrector step

		for i in range(1,nx):
			dU1_dt_n1[i]=-((F_1p[i]-F_1p[i-1])/(delta_x))
			dU2_dt_n1[i]=-((F_2p[i]-F_2p[i-1])/(delta_x))+J_2p[i]
			dU3_dt_n1[i]=-((F_3p[i]-F_3p[i-1])/(delta_x))

#average time derivative
		for i in range(1,nx-1):
			dU1_dt_av[i]=0.5*(dU1_dt_n1[i]+dU1_dt_n[i])
			dU2_dt_av[i]=0.5*(dU2_dt_n1[i]+dU2_dt_n[i])
			dU3_dt_av[i]=0.5*(dU3_dt_n1[i]+dU3_dt_n[i])
#final update to U
		for i in range(1,nx-1):
			U_1n[i]=U_1[i]+(dt*dU1_dt_av[i])
			U_2n[i]=U_2[i]+(dt*dU2_dt_av[i])
			U_3n[i]=U_3[i]+(dt*dU3_dt_av[i])
		U_1n[0]=A[0]
		U_2n[0]=2*U_2n[1]-(U_2n[2])
		U_3n[0]=U_1n[0]*(1/(Y-1)+(Y/2)*((U_2n[0])/(U_1n[0]))**2)
		U_1n[nx-1]=2*U_1n[nx-2]-(U_1n[nx-3])
		U_2n[nx-1]=2*U_2n[nx-2]-(U_2n[nx-3])
		
		for i in range(nx):
			rho_n[i]=U_1n[i]/A[i]
			V_n[i]=U_2n[i]/U_1n[i]
			
		U_3n[nx-1]=((0.6784*A[nx-1])/(Y-1))+((Y/2)*U_2n[nx-1]*V_n[nx-1])
		for i in range(nx):
			T_n[i]=(Y-1)*((U_3n[i]/U_1n[i])-((Y/2)*(U_2n[i]/U_1n[i])**2))
		
		# Calculate pressure for this time step
		p_current = np.zeros(nx)
		for i in range(nx):
			p_current[i] = rho_n[i] * T_n[i]
		
		# Store data for animation
		rho_all.append(rho_n.copy())
		T_all.append(T_n.copy())
		p_all.append(p_current.copy())
		
		rho=rho_n.copy()
		T=T_n.copy()
		V=V_n.copy()
		U_1=U_1n.copy()
		U_2=U_2n.copy()
		U_3=U_3n.copy()
	return U_1,U_2,U_3,rho,T,V,rho_all,T_all,p_all

U_1,U_2,U_3,rho,T,V,rho_all,T_all,p_all=MCormack(U_1,U_2,U_3,rho,V,T,nt,nx,A)

for i in range(nx):
	p[i]=rho[i]*T[i]

# Convert to numpy arrays for animation
rho_all = np.array(rho_all)
T_all = np.array(T_all)
p_all = np.array(p_all)

# Create figure and axes
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
lines = []

# Set up plots with proper axis limits
data_arrays = [rho_all, p_all, T_all]
labels = ['Density', 'Pressure', 'Temperature']

for i, (data, label) in enumerate(zip(data_arrays, labels)):
    ax[i].set_ylabel(label)
    ax[i].grid(True)
    ax[i].set_xlim(0, 3)
    # Set y-limits based on data range with some padding
    y_min, y_max = np.min(data), np.max(data)
    y_range = y_max - y_min
    ax[i].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    line, = ax[i].plot([], [], lw=2, color='blue')
    lines.append(line)

ax[-1].set_xlabel('Nozzle Length (x)')
fig.suptitle('Flow Evolution in Nozzle')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    lines[0].set_data(x, rho_all[frame])
    lines[1].set_data(x, p_all[frame])
    lines[2].set_data(x, T_all[frame])
    
    # Update title to show current time step
    fig.suptitle(f'Flow Evolution in Nozzle - Time Step: {frame+1}/{len(rho_all)}')
    
    return lines

# Create animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(rho_all),
    interval=200, blit=False, repeat=True
)

plt.tight_layout()
plt.show()

# Optional: Save animation as GIF (uncomment if needed)
# ani.save('nozzle_flow_animation.gif', writer='pillow', fps=5)