import numpy as np 
import math 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


Y=1.4

R=287
C_F_L=0.5
Cx=0.6
theta=math.radians(5.352)

def inverse_prandtl_meyer(f, Y):
    return brentq(lambda M:np.sqrt((Y+1)/(Y-1))*math.atan(np.sqrt((Y-1)/(Y+1)*(M**2-1)))-np.arctan(np.sqrt(M**2-1))-f,1.01,5)

#Initial data line
n=int(input('Enter number of grid points'))
L=int(input('Enter distance till which space marching must occur'))
neta=np.linspace(0,1,n)
dn=1/(n-1)
M=np.full(n,2.0)
rho=np.full(n,1.23)
u=np.full(n,678.0)
T=np.full(n,286.0)
p=np.full(n,101000.0)
v=np.full(n,0.0)
df=pd.DataFrame({'M':M,'rho':rho,'u':u,'v':v,'T':T,'P':p,})
F1=np.zeros(n)
F2=np.zeros(n)
F3=np.zeros(n)
F4=np.zeros(n)
G1=np.zeros(n)
G2=np.zeros(n)
G3=np.zeros(n)
G4=np.zeros(n)
F1p=np.zeros(n)
F2p=np.zeros(n)
F3p=np.zeros(n)
F4p=np.zeros(n)
G1p=np.zeros(n)
G2p=np.zeros(n)
G3p=np.zeros(n)
G4p=np.zeros(n)
F1n=np.zeros(n)
F2n=np.zeros(n)
F3n=np.zeros(n)
F4n=np.zeros(n)
G1n=np.zeros(n)
G2n=np.zeros(n)
G3n=np.zeros(n)
G4n=np.zeros(n)
rhop=np.zeros(n)
rhon=np.zeros(n)
dF1_dzeta=np.zeros(n-1)#top most grid point flowfield value does not change, ie at y=40 it does not change
dF2_dzeta=np.zeros(n-1)
dF3_dzeta=np.zeros(n-1)
dF4_dzeta=np.zeros(n-1)
dF1_dzeta1=np.zeros(n-1)
dF2_dzeta1=np.zeros(n-1)
dF3_dzeta1=np.zeros(n-1)
dF4_dzeta1=np.zeros(n-1)
dF1_dzetaav=np.zeros(n-1)
dF2_dzetaav=np.zeros(n-1)
dF3_dzetaav=np.zeros(n-1)
dF4_dzetaav=np.zeros(n-1)
y=np.linspace(0,40,n)
# Storage for animation data
x_positions=[]
u_data=[]
y_data=[]
v_data=[]
M_data=[]
rho_data=[]
p_data=[]

x=0
#Predictor step for i+1 values(for all j interior points)
#initialize F and G vectors
for j in range(n):
    F1[j]=rho[j]*u[j]
    F2[j]=rho[j]*u[j]**2+p[j]
    F3[j]=rho[j]*u[j]*v[j]
    F4[j]=(Y/(Y-1))*p[j]*u[j]+rho[j]*u[j]*((u[j]**2+v[j]**2)/2)
    G1[j]=rho[j]*v[j]
    G2[j]=rho[j]*u[j]*v[j]
    G3[j]=rho[j]*v[j]**2+p[j]
    G4[j]=(Y/(Y-1))*p[j]*v[j]+rho[j]*v[j]*((u[j]**2+v[j]**2)/2)
    F1p[j]=F1[j]
    F2p[j]=F2[j]
    F3p[j]=F3[j]
    F4p[j]=F4[j]
    G1p[j]=G1[j]
    G2p[j]=G2[j]
    G3p[j]=G3[j]
    G4p[j]=G4[j]

while x<L:
    if x<=10:
        #Start predictor step
        h=40
        dz=C_F_L*(3**0.5)
        #dndx=0 for x<10m
        #do first for boundary point or j=0
        dF1_dzeta[0]=(1/h)*((G1[0]-G1[0+1])/(dn))
        dF2_dzeta[0]=(1/h)*((G2[0]-G2[0+1])/(dn))
        dF3_dzeta[0]=(1/h)*((G3[0]-G3[0+1])/(dn))
        dF4_dzeta[0]=(1/h)*((G4[0]-G4[0+1])/(dn))
        F1p[0]=(dF1_dzeta[0]*dz)+F1[0]
        F2p[0]=(dF2_dzeta[0]*dz)+F2[0]
        F3p[0]=(dF3_dzeta[0]*dz)+F3[0]
        F4p[0]=(dF4_dzeta[0]*dz)+F4[0]
        A=(F3p[0]**2)/(2*F1p[0])-F4p[0]
        B=(Y/(Y-1))*F1p[0]*F2p[0]
        C=-((Y+1)/(2*(Y-1)))*(F1p[0]**3)
        rhop[0]=(-B+np.sqrt(B**2-4*A*C))/(2*A)
        G1p[0]=rhop[0]*(F3p[0]/F1p[0])
        G2p[0]=F3p[0]
        G3p[0]=(rhop[0]*(F3p[0]/F1p[0])**2)+F2p[0]-((F1p[0]**2)/rhop[0])
        G4p[0]=(Y/(Y-1))*(F2p[0]-(F1p[0]**2)/rhop[0])*(F3p[0]/F1p[0])+(rhop[0]/2)*(F3p[0]/F1p[0])*((F1p[0]/rhop[0])**2+(F3p[0]/F1p[0])**2)
        #corrector step
        dF1_dzeta1[0]=(1/h)*((G1p[0]-G1p[0+1])/(dn))
        dF2_dzeta1[0]=(1/h)*((G2p[0]-G2p[0+1])/(dn))
        dF3_dzeta1[0]=(1/h)*((G3p[0]-G3p[0+1])/(dn))
        dF4_dzeta1[0]=(1/h)*((G4p[0]-G4p[0+1])/(dn))
        dF1_dzetaav[0]=(dF1_dzeta[0]+dF1_dzeta1[0])/(2)
        dF2_dzetaav[0]=(dF2_dzeta[0]+dF2_dzeta1[0])/(2)
        dF3_dzetaav[0]=(dF3_dzeta[0]+dF3_dzeta1[0])/(2)
        dF4_dzetaav[0]=(dF4_dzeta[0]+dF4_dzeta1[0])/(2)
        F1n[0]=F1[0]+(dF1_dzetaav[0]*dz)
        F2n[0]=F2[0]+(dF2_dzetaav[0]*dz)
        F3n[0]=F3[0]+(dF3_dzetaav[0]*dz)
        F4n[0]=F4[0]+(dF4_dzetaav[0]*dz)
        An=(F3n[0]**2)/(2*F1n[0])-F4n[0]
        Bn=(Y/(Y-1))*F1n[0]*F2n[0]
        Cn=-((Y+1)/(2*(Y-1)))*(F1n[0]**3)
        rhon[0]=(-Bn+np.sqrt(Bn**2-4*An*Cn))/(2*An)
        G1n[0]=rhon[0]*(F3n[0]/F1n[0])
        G2n[0]=F3n[0]
        G3n[0]=rhon[0]*(F3n[0]/F1n[0])**2+F2n[0]-(F1n[0]**2)/rhon[0]
        G4n[0]=(Y/(Y-1))*(F2n[0]-(F1n[0]**2)/rhon[0])*(F3n[0]/F1n[0])+(rhon[0]/2)*(F3n[0]/F1n[0])*((F1n[0]/rhon[0])**2+(F3n[0]/F1n[0])**2)
        F1[0]=F1n[0]
        F2[0]=F2n[0]
        F3[0]=F3n[0]
        F4[0]=F4n[0]
        G1[0]=G1n[0]
        G2[0]=G2n[0]
        G3[0]=G3n[0]
        G4[0]=G4n[0]
        rho[0]=rhon[0]
        u[0]=F1[0]/rho[0]
        v[0]=F3[0]/F1[0]
        p[0]=F2[0]-(F1[0]*u[0])
        T[0]=(p[0])/(rho[0]*R)
        M[0]=((u[0]**2+v[0]**2)**0.5)/(Y*R*T[0])**0.5
        M_cal=M[0]

        phi=math.atan(v[0]/u[0])
        f_cal=((Y+1)/(Y-1))**0.5*math.atan(((Y-1)/(Y+1)*(M_cal**2-1))**0.5)-math.atan((M_cal**2-1)**0.5)
        f_act=f_cal+phi
        M_act=inverse_prandtl_meyer(f_act,Y)
        p_act=p[0]*((1+((Y-1)/2)*M_cal**2)/(1+((Y-1)/2)*M_act**2))**(Y/(Y-1))
        T_act=T[0]*((1+((Y-1)/2)*M_cal**2)/(1+((Y-1)/2)*M_act**2))
        rho_act=p_act/(R*T_act)
        p[0]=p_act
        rho[0]=rho_act
        T[0]=T_act
        M[0]=M_act
        u[0]=F1[0]/rho[0]
        v[0]=0
        #do for all interior points
        for j in range(1,n-1):
            dF1_dzeta[j]=(1/h)*((G1[j]-G1[j+1])/(dn))
            dF2_dzeta[j]=(1/h)*((G2[j]-G2[j+1])/(dn))
            dF3_dzeta[j]=(1/h)*((G3[j]-G3[j+1])/(dn))
            dF4_dzeta[j]=(1/h)*((G4[j]-G4[j+1])/(dn))
            F1p[j]=(dF1_dzeta[j]*dz)+F1[j]
            F2p[j]=(dF2_dzeta[j]*dz)+F2[j]
            F3p[j]=(dF3_dzeta[j]*dz)+F3[j]
            F4p[j]=(dF4_dzeta[j]*dz)+F4[j]
            A=(F3p[j]**2)/(2*F1p[j])-F4p[j]
            B=(Y/(Y-1))*F1p[j]*F2p[j]
            C=-((Y+1)/(2*(Y-1)))*(F1p[j]**3)
            rhop[j]=(-B+np.sqrt(B**2-4*A*C))/(2*A)
            G1p[j]=rhop[j]*(F3p[j]/F1p[j])
            G2p[j]=F3p[j]
            G3p[j]=rhop[j]*(F3p[j]/F1p[j])**2+F2p[j]-(F1p[j]**2)/rhop[j]
            G4p[j]=(Y/(Y-1))*(F2p[j]-(F1p[j]**2)/rhop[j])*(F3p[j]/F1p[j])+(rhop[j]/2)*(F3p[j]/F1p[j])*((F1p[j]/rhop[j])**2+(F3p[j]/F1p[j])**2)
            #corrector step
            dF1_dzeta1[j]=(1/h)*((G1p[j-1]-G1p[j])/(dn))
            dF2_dzeta1[j]=(1/h)*((G2p[j-1]-G2p[j])/(dn))
            dF3_dzeta1[j]=(1/h)*((G3p[j-1]-G3p[j])/(dn))
            dF4_dzeta1[j]=(1/h)*((G4p[j-1]-G4p[j])/(dn))
            dF1_dzetaav[j]=(dF1_dzeta[j]+dF1_dzeta1[j])/(2)
            dF2_dzetaav[j]=(dF2_dzeta[j]+dF2_dzeta1[j])/(2)
            dF3_dzetaav[j]=(dF3_dzeta[j]+dF3_dzeta1[j])/(2)
            dF4_dzetaav[j]=(dF4_dzeta[j]+dF4_dzeta1[j])/(2)
            F1n[j]=F1[j]+(dF1_dzetaav[j]*dz)
            F2n[j]=F2[j]+(dF2_dzetaav[j]*dz)
            F3n[j]=F3[j]+(dF3_dzetaav[j]*dz)
            F4n[j]=F4[j]+(dF4_dzetaav[j]*dz)
            An=(F3n[j]**2)/(2*F1n[j])-F4n[j]
            Bn=(Y/(Y-1))*F1n[j]*F2n[j]
            Cn=-((Y+1)/(2*(Y-1)))*(F1n[j]**3)
            rhon[j]=(-Bn+np.sqrt(Bn**2-4*An*Cn))/(2*An)
            G1n[j]=rhon[j]*(F3n[j]/F1n[j])
            G2n[j]=F3n[j]
            G3n[j]=rhon[j]*(F3n[j]/F1n[j])**2+F2n[j]-(F1n[j]**2)/rhon[j]
            G4n[j]=(Y/(Y-1))*(F2n[j]-(F1n[j]**2)/rhon[j])*(F3n[j]/F1n[j])+(rhon[j]/2)*(F3n[j]/F1n[j])*((F1n[j]/rhon[j])**2+(F3n[j]/F1n[j])**2)
            F1[j]=F1n[j]
            F2[j]=F2n[j]
            F3[j]=F3n[j]
            F4[j]=F4n[j]
            G1[j]=G1n[j]
            G2[j]=G2n[j]
            G3[j]=G3n[j]
            G4[j]=G4n[j]
            rho[j]=rhon[j]
            u[j]=F1[j]/rho[j]
            v[j]=F3[j]/F1[j]
            p[j]=F2[j]-(F1[j]*u[j])
            T[j]=(p[j])/(rho[j]*R)
            M[j]=((u[j]**2+v[j]**2)**0.5)/(Y*R*T[j])**0.5
    
    else:
        # start predictor step
        h=40+((x-10)*math.tan(theta))
        bottom=-((x-10)*math.tan(math.radians(5.352)))
        y=np.linspace(bottom,40,n)
        dy=(y[-1]-y[0])/(n-1)
        denom=np.zeros(n)
        mu=np.zeros(n)
        for j in range(n):
            mu[j]=math.asin(1/M[j])
            denom[j]=max(abs(math.tan(theta+mu[j])),abs(math.tan(theta-mu[j])))
        dz1=np.zeros(n)
        for j in range(n):
            dz1[j]=(C_F_L*dy)/denom[j]
        dz=min(dz1)
        dndx=np.zeros(n-1)
        for j in range(n-1):
            dndx[j]=(1-neta[j])*(math.tan(theta)/h)

        dF1_dzeta[0]=((dndx[0])*((F1[0]-F1[1])/(dn)))+(1/h)*((G1[0]-G1[0+1])/(dn))
        dF2_dzeta[0]=((dndx[0])*((F2[0]-F2[1])/(dn)))+(1/h)*((G2[0]-G2[0+1])/(dn))
        dF3_dzeta[0]=((dndx[0])*((F3[0]-F3[1])/(dn)))+(1/h)*((G3[0]-G3[0+1])/(dn))
        dF4_dzeta[0]=((dndx[0])*((F4[0]-F4[1])/(dn)))+(1/h)*((G4[0]-G4[0+1])/(dn))
        F1p[0]=(dF1_dzeta[0]*dz)+F1[0]
        F2p[0]=(dF2_dzeta[0]*dz)+F2[0]
        F3p[0]=(dF3_dzeta[0]*dz)+F3[0]
        F4p[0]=(dF4_dzeta[0]*dz)+F4[0]
        A=(F3p[0]**2)/(2*F1p[0])-F4p[0]
        B=(Y/(Y-1))*F1p[0]*F2p[0]
        C=-((Y+1)/(2*(Y-1)))*(F1p[0]**3)
        rhop[0]=(-B+np.sqrt(B**2-4*A*C))/(2*A)
        G1p[0]=rhop[0]*(F3p[0]/F1p[0])
        G2p[0]=F3p[0]
        G3p[0]=(rhop[0]*(F3p[0]/F1p[0])**2)+F2p[0]-((F1p[0]**2)/rhop[0])
        G4p[0]=(Y/(Y-1))*(F2p[0]-(F1p[0]**2)/rhop[0])*(F3p[0]/F1p[0])+(rhop[0]/2)*(F3p[0]/F1p[0])*((F1p[0]/rhop[0])**2+(F3p[0]/F1p[0])**2)
        #corrector step
        dF1_dzeta1[0]=((dndx[0])*((F1p[0]-F1p[1])/(dn)))+(1/h)*((G1p[0]-G1p[0+1])/(dn))
        dF2_dzeta1[0]=((dndx[0])*((F2p[0]-F2p[1])/(dn)))+(1/h)*((G2p[0]-G2p[0+1])/(dn))
        dF3_dzeta1[0]=((dndx[0])*((F3p[0]-F3p[1])/(dn)))+(1/h)*((G3p[0]-G3p[0+1])/(dn))
        dF4_dzeta1[0]=((dndx[0])*((F4p[0]-F4p[1])/(dn)))+(1/h)*((G4p[0]-G4p[0+1])/(dn))
        dF1_dzetaav[0]=(dF1_dzeta[0]+dF1_dzeta1[0])/(2)
        dF2_dzetaav[0]=(dF2_dzeta[0]+dF2_dzeta1[0])/(2)
        dF3_dzetaav[0]=(dF3_dzeta[0]+dF3_dzeta1[0])/(2)
        dF4_dzetaav[0]=(dF4_dzeta[0]+dF4_dzeta1[0])/(2)
        F1n[0]=F1[0]+(dF1_dzetaav[0]*dz)
        F2n[0]=F2[0]+(dF2_dzetaav[0]*dz)
        F3n[0]=F3[0]+(dF3_dzetaav[0]*dz)
        F4n[0]=F4[0]+(dF4_dzetaav[0]*dz)
        An=(F3n[0]**2)/(2*F1n[0])-F4n[0]
        Bn=(Y/(Y-1))*F1n[0]*F2n[0]
        Cn=-((Y+1)/(2*(Y-1)))*(F1n[0]**3)
        rhon[0]=(-Bn+np.sqrt(Bn**2-4*An*Cn))/(2*An)
        G1n[0]=rhon[0]*(F3n[0]/F1n[0])
        G2n[0]=F3n[0]
        G3n[0]=rhon[0]*(F3n[0]/F1n[0])**2+F2n[0]-(F1n[0]**2)/rhon[0]
        G4n[0]=(Y/(Y-1))*(F2n[0]-(F1n[0]**2)/rhon[0])*(F3n[0]/F1n[0])+(rhon[0]/2)*(F3n[0]/F1n[0])*((F1n[0]/rhon[0])**2+(F3n[0]/F1n[0])**2)
        F1[0]=F1n[0]
        F2[0]=F2n[0]
        F3[0]=F3n[0]
        F4[0]=F4n[0]
        G1[0]=G1n[0]
        G2[0]=G2n[0]
        G3[0]=G3n[0]
        G4[0]=G4n[0]
        rho[0]=rhon[0]
        u[0]=F1[0]/rho[0]
        v[0]=F3[0]/F1[0]
        v_final=-u[0]*math.tan(theta)
        p[0]=F2[0]-(F1[0]*u[0])
        T[0]=(p[0])/(rho[0]*R)
        M[0]=((u[0]**2+v[0]**2)**0.5)/(Y*R*T[0])**0.5
        M_cal=M[0]

        si=math.atan(abs(v[0])/u[0])
        phi2=theta-si
        f_cal=((Y+1)/(Y-1))**0.5*math.atan(((Y-1)/(Y+1)*(M_cal**2-1))**0.5)-math.atan((M_cal**2-1)**0.5)
        f_act=f_cal+phi2
        M_act=inverse_prandtl_meyer(f_act,Y)
        p_act=p[0]*((1+((Y-1)/2)*M_cal**2)/(1+((Y-1)/2)*M_act**2))**(Y/(Y-1))
        T_act=T[0]*((1+((Y-1)/2)*M_cal**2)/(1+((Y-1)/2)*M_act**2))
        rho_act=p_act/(R*T_act)
        p[0]=p_act
        rho[0]=rho_act
        T[0]=T_act
        M[0]=M_act
        u_final=F1[0]/rho[0]
        u[0]=u_final
        v[0]=v_final
        F1[0]=rho[0]*u[0]
        F2[0]=rho[0]*u[0]**2+p[0]
        F3[0]=rho[0]*u[0]*v[0]
        F4[0]=(Y/(Y-1))*p[0]*u[0]+rho[0]*u[0]*((u[0]**2+v[0]**2)/2)
        G1[0]=rho[0]*v[0]
        G2[0]=rho[0]*u[0]*v[0]
        G3[0]=rho[0]*v[0]**2+p[0]
        G4[0]=(Y/(Y-1))*p[0]*v[0]+rho[0]*v[0]*((u[0]**2+v[0]**2)/2)

                
        #do for all interior points
        for j in range(1,n-1):
            dF1_dzeta[j]=((dndx[j])*((F1[j]-F1[j+1])/(dn)))+(1/h)*((G1[j]-G1[j+1])/(dn))
            dF2_dzeta[j]=((dndx[j])*((F2[j]-F2[j+1])/(dn)))+(1/h)*((G2[j]-G2[j+1])/(dn))
            dF3_dzeta[j]=((dndx[j])*((F3[j]-F3[j+1])/(dn)))+(1/h)*((G3[j]-G3[j+1])/(dn))
            dF4_dzeta[j]=((dndx[j])*((F4[j]-F4[j+1])/(dn)))+(1/h)*((G4[j]-G4[j+1])/(dn))
            F1p[j]=(dF1_dzeta[j]*dz)+F1[j]
            F2p[j]=(dF2_dzeta[j]*dz)+F2[j]
            F3p[j]=(dF3_dzeta[j]*dz)+F3[j]
            F4p[j]=(dF4_dzeta[j]*dz)+F4[j]
            A=(F3p[j]**2)/(2*F1p[j])-F4p[j]
            B=(Y/(Y-1))*F1p[j]*F2p[j]
            C=-((Y+1)/(2*(Y-1)))*(F1p[j]**3)
            rhop[j]=(-B+np.sqrt(B**2-4*A*C))/(2*A)
            G1p[j]=rhop[j]*(F3p[j]/F1p[j])
            G2p[j]=F3p[j]
            G3p[j]=rhop[j]*(F3p[j]/F1p[j])**2+F2p[j]-(F1p[j]**2)/rhop[j]
            G4p[j]=(Y/(Y-1))*(F2p[j]-(F1p[j]**2)/rhop[j])*(F3p[j]/F1p[j])+(rhop[j]/2)*(F3p[j]/F1p[j])*((F1p[j]/rhop[j])**2+(F3p[j]/F1p[j])**2)
            #corrector step
            dF1_dzeta1[j]=((dndx[j])*((F1p[j-1]-F1p[j])/(dn)))+(1/h)*((G1p[j-1]-G1p[j])/(dn))
            dF2_dzeta1[j]=((dndx[j])*((F2p[j-1]-F2p[j])/(dn)))+(1/h)*((G2p[j-1]-G2p[j])/(dn))
            dF3_dzeta1[j]=((dndx[j])*((F3p[j-1]-F3p[j])/(dn)))+(1/h)*((G3p[j-1]-G3p[j])/(dn))
            dF4_dzeta1[j]=((dndx[j])*((F4p[j-1]-F4p[j])/(dn)))+(1/h)*((G4p[j-1]-G4p[j])/(dn))
            dF1_dzetaav[j]=(dF1_dzeta[j]+dF1_dzeta1[j])/(2)
            dF2_dzetaav[j]=(dF2_dzeta[j]+dF2_dzeta1[j])/(2)
            dF3_dzetaav[j]=(dF3_dzeta[j]+dF3_dzeta1[j])/(2)
            dF4_dzetaav[j]=(dF4_dzeta[j]+dF4_dzeta1[j])/(2)
            F1n[j]=F1[j]+(dF1_dzetaav[j]*dz)
            F2n[j]=F2[j]+(dF2_dzetaav[j]*dz)
            F3n[j]=F3[j]+(dF3_dzetaav[j]*dz)
            F4n[j]=F4[j]+(dF4_dzetaav[j]*dz)
            An=(F3n[j]**2)/(2*F1n[j])-F4n[j]
            Bn=(Y/(Y-1))*F1n[j]*F2n[j]
            Cn=-((Y+1)/(2*(Y-1)))*(F1n[j]**3)
            rhon[j]=(-Bn+np.sqrt(Bn**2-4*An*Cn))/(2*An)
            G1n[j]=rhon[j]*(F3n[j]/F1n[j])
            G2n[j]=F3n[j]
            G3n[j]=rhon[j]*(F3n[j]/F1n[j])**2+F2n[j]-(F1n[j]**2)/rhon[j]
            G4n[j]=(Y/(Y-1))*(F2n[j]-(F1n[j]**2)/rhon[j])*(F3n[j]/F1n[j])+(rhon[j]/2)*(F3n[j]/F1n[j])*((F1n[j]/rhon[j])**2+(F3n[j]/F1n[j])**2)
            F1[j]=F1n[j]
            F2[j]=F2n[j]
            F3[j]=F3n[j]
            F4[j]=F4n[j]
            G1[j]=G1n[j]
            G2[j]=G2n[j]
            G3[j]=G3n[j]
            G4[j]=G4n[j]
            rho[j]=rhon[j]
            u[j]=F1[j]/rho[j]
            v[j]=F3[j]/F1[j]
            p[j]=F2[j]-(F1[j]*u[j])
            T[j]=(p[j])/(rho[j]*R)
            M[j]=((u[j]**2+v[j]**2)**0.5)/(Y*R*T[j])**0.5








        
    # Store data for animation
    # Store data for animation (modify your existing data storage)
    x_positions.append(x)
    u_data.append(u.copy())
    y_data.append(y.copy())
    v_data.append(v.copy())
    M_data.append(M.copy())
    rho_data.append(rho.copy())
    p_data.append(p.copy())  # ADD THIS LINE
    x+=dz
print(x,h)    
df=pd.DataFrame({'y':y,'neta':neta,'u':u,"v":v,'rho':rho,'p':p})
print(df)
# Animation function for u velocity vs y as flow marches in x
# Enhanced animation function for flow field visualization
def create_flow_field_animation():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Find global min/max for consistent scaling
    u_min = min([min(u_step) for u_step in u_data])
    u_max = max([max(u_step) for u_step in u_data])
    
    # Plot 1: Flow field with velocity contours
    ax1.set_xlim(0, L+3)
    ax1.set_ylim(-L*0.2, 42)
    ax1.set_xlabel('x position (m)')
    ax1.set_ylabel('y position (m)')
    ax1.set_title('Flow Field Evolution over Flat Plate with Ramp')
    ax1.grid(True, alpha=0.3)
    
    # Draw the geometry (flat plate + ramp)
    x_geom = [0, 10, L+3]
    y_geom = [0, 0, -(L+3-10)*np.tan(np.radians(5.352))]
    ax1.plot(x_geom, y_geom, 'k-', linewidth=3, label='Wall')
    ax1.fill_between(x_geom, y_geom, -L*0.3, color='gray', alpha=0.3)
    
    # Plot 2: Velocity profile at current x-station
    ax2.set_xlim(u_min * 0.95, u_max * 1.05)
    ax2.set_ylim(-L*0.2, 42)
    ax2.set_xlabel('u velocity (m/s)')
    ax2.set_ylabel('y position (m)')
    ax2.grid(True, alpha=0.3)
    
    # Initialize empty plots
    velocity_line, = ax2.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Flow field visualization
        ax1.set_xlim(0, L+3)
        ax1.set_ylim(-L*0.2, 42)
        ax1.set_xlabel('x position (m)')
        ax1.set_ylabel('y position (m)')
        ax1.set_title(f'Flow Field Evolution - x = {x_positions[frame]:.3f} m')
        ax1.grid(True, alpha=0.3)
        
        # Draw geometry
        x_geom = [0, 10, L+3]
        y_geom = [0, 0, -(L+3-10)*np.tan(np.radians(5.352))]
        ax1.plot(x_geom, y_geom, 'k-', linewidth=3, label='Wall')
        ax1.fill_between(x_geom, y_geom, -L*0.3, color='gray', alpha=0.3)
        
        # Show velocity vectors up to current position
        for i in range(min(frame+1, len(x_positions))):
            x_pos = x_positions[i]
            y_pos = y_data[i]
            u_vel = u_data[i]
            v_vel = v_data[i] if i < len(v_data) else np.zeros_like(u_vel)
            
            # Color by velocity magnitude
            velocity_mag = np.sqrt(u_vel**2 + v_vel**2)
            colors = plt.cm.viridis((velocity_mag - u_min) / (u_max - u_min))
            
            # Plot velocity vectors (scaled down for visibility)
            scale = 0.1
            ax1.quiver(np.full_like(y_pos, x_pos), y_pos, 
                      u_vel * scale, v_vel * scale, 
                      color=colors, alpha=0.7, width=0.003)
        
        # Highlight current position
        if frame < len(x_positions):
            ax1.axvline(x_positions[frame], color='red', linestyle='--', alpha=0.7, 
                       label=f'Current x = {x_positions[frame]:.3f} m')
            ax1.legend()
        
        # Plot 2: Current velocity profile
        ax2.set_xlim(u_min * 0.95, u_max * 1.05)
        ax2.set_ylim(-L*0.2, 42)
        ax2.set_xlabel('u velocity (m/s)')
        ax2.set_ylabel('y position (m)')
        ax2.set_title(f'Velocity Profile at x = {x_positions[frame]:.3f} m')
        ax2.grid(True, alpha=0.3)
        
        if frame < len(u_data):
            ax2.plot(u_data[frame], y_data[frame], 'b-', linewidth=2, marker='o', markersize=4)
            
            # Add boundary layer visualization
            # Find 99% of freestream velocity
            u_freestream = u_data[frame][-1]  # Top boundary
            u_99 = 0.99 * u_freestream
            
            # Find boundary layer thickness
            for j, u_val in enumerate(u_data[frame]):
                if u_val >= u_99:
                    delta_99 = y_data[frame][j]
                    ax2.axhline(delta_99, color='red', linestyle=':', alpha=0.7, 
                              label=f'δ₉₉ = {delta_99:.2f} m')
                    ax2.legend()
                    break
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(x_positions), 
                        interval=300, blit=False, repeat=True)
    
    plt.tight_layout()
    return anim

# Moving camera animation showing u velocity changes
def create_u_velocity_moving_camera(window_width=20):
    """
    Moving camera that clearly shows u velocity changes as flow marches
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Find global min/max for consistent scaling
    u_min = min([min(u_step) for u_step in u_data])
    u_max = max([max(u_step) for u_step in u_data])
    y_min_global = min([min(y_step) for y_step in y_data])
    y_max_global = max([max(y_step) for y_step in y_data])
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        current_x = x_positions[frame]
        
        # Moving window: center the view around current x position
        x_left = current_x - window_width/3
        x_right = current_x + 2*window_width/3
        
        # TOP PLOT: U velocity contour/field
        ax1.set_xlim(x_left, x_right)
        ax1.set_ylim(y_min_global - 2, y_max_global + 2)
        ax1.set_xlabel('x position (m)')
        ax1.set_ylabel('y position (m)')
        ax1.set_title(f'U Velocity Field Evolution - x = {current_x:.3f} m')
        ax1.grid(True, alpha=0.3)
        
        # Draw geometry within view
        _draw_geometry_in_view(ax1, x_left, x_right, y_min_global)
        
        # Show U velocity as colored contours/points for stations within view
        stations_in_view = []
        for i in range(frame + 1):
            if x_left <= x_positions[i] <= x_right:
                stations_in_view.append(i)
        
        # Show recent history
        history_length = min(15, len(stations_in_view))
        stations_to_show = stations_in_view[-history_length:]
        
        # Create mesh for u velocity visualization
        X_mesh = []
        Y_mesh = []
        U_mesh = []
        
        for i in stations_to_show:
            x_pos = x_positions[i]
            y_pos = y_data[i]
            u_vel = u_data[i]
            
            X_mesh.extend([x_pos] * len(y_pos))
            Y_mesh.extend(y_pos)
            U_mesh.extend(u_vel)
        
        # Plot u velocity as colored scatter points
        if X_mesh:
            scatter = ax1.scatter(X_mesh, Y_mesh, c=U_mesh, cmap='plasma', 
                                 s=25, alpha=0.8, vmin=u_min, vmax=u_max)
            
            # Add colorbar
            if frame == 0:
                cbar1 = plt.colorbar(scatter, ax=ax1)
                cbar1.set_label('U Velocity (m/s)', fontsize=12)
        
        # Highlight current position
        ax1.axvline(current_x, color='white', linestyle='--', alpha=0.9, 
                   linewidth=3, label=f'Current x = {current_x:.3f} m')
        
        # Add text showing u velocity statistics at current station
        if frame < len(u_data):
            u_current = u_data[frame]
            u_wall = u_current[0]  # Wall velocity
            u_freestream = u_current[-1]  # Freestream velocity
            u_mean = np.mean(u_current)
            
            stats_text = f'U Velocity Stats:\nWall: {u_wall:.1f} m/s\nFreestream: {u_freestream:.1f} m/s\nMean: {u_mean:.1f} m/s'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=10)
        
        ax1.legend(loc='upper right')
        
        # BOTTOM PLOT: U velocity evolution along x
        ax2.set_xlim(x_left, x_right)
        ax2.set_xlabel('x position (m)')
        ax2.set_ylabel('U Velocity (m/s)')
        ax2.set_title('U Velocity Evolution (Wall, Mean, Freestream)')
        ax2.grid(True, alpha=0.3)
        
        # Plot u velocity evolution for different y-locations
        if len(stations_to_show) > 1:
            x_stations = [x_positions[i] for i in stations_to_show]
            
            # Wall velocity (j=0)
            u_wall_evolution = [u_data[i][0] for i in stations_to_show]
            ax2.plot(x_stations, u_wall_evolution, 'r-', linewidth=2, 
                    marker='o', markersize=4, label='Wall (y≈0)')
            
            # Middle of boundary layer (j=n//4)
            u_mid_evolution = [u_data[i][len(u_data[i])//4] for i in stations_to_show]
            ax2.plot(x_stations, u_mid_evolution, 'g-', linewidth=2, 
                    marker='s', markersize=4, label='Mid Boundary Layer')
            
            # Freestream velocity (j=-1)
            u_free_evolution = [u_data[i][-1] for i in stations_to_show]
            ax2.plot(x_stations, u_free_evolution, 'b-', linewidth=2, 
                    marker='^', markersize=4, label='Freestream')
            
            # Mean velocity
            u_mean_evolution = [np.mean(u_data[i]) for i in stations_to_show]
            ax2.plot(x_stations, u_mean_evolution, 'k--', linewidth=2, 
                    marker='d', markersize=4, label='Domain Average')
        
        # Current position line
        ax2.axvline(current_x, color='red', linestyle=':', alpha=0.7, linewidth=2)
        
        # Add region indicators
        if current_x <= 10:
            ax2.axvspan(max(x_left, 0), min(x_right, 10), alpha=0.2, color='blue', label='Flat Plate')
        if x_right > 10:
            ax2.axvspan(max(x_left, 10), x_right, alpha=0.2, color='red', label='Diverging Ramp')
        
        ax2.legend(loc='best')
        ax2.set_ylim(u_min * 0.95, u_max * 1.05)
        
        return []
    
    def _draw_geometry_in_view(ax, x_left, x_right, y_min_global):
        """Helper function to draw geometry within view"""
        # Flat plate section
        if x_left <= 10:
            plate_start = max(0, x_left)
            plate_end = min(10, x_right)
            if plate_start < plate_end:
                ax.plot([plate_start, plate_end], [0, 0], 'k-', linewidth=4)
        
        # Ramp section
        if x_right >= 10:
            ramp_start = max(10, x_left)
            ramp_end = min(x_right, L+10)
            if ramp_start < ramp_end:
                ramp_start_y = -(ramp_start - 10) * np.tan(np.radians(5.352))
                ramp_end_y = -(ramp_end - 10) * np.tan(np.radians(5.352))
                ax.plot([ramp_start, ramp_end], [ramp_start_y, ramp_end_y], 'k-', linewidth=4)
        
        # Fill below wall
        x_wall_points = np.linspace(max(0, x_left), min(L+10, x_right), 50)
        wall_x = []
        wall_y = []
        for x_w in x_wall_points:
            wall_x.append(x_w)
            if x_w <= 10:
                wall_y.append(0)
            else:
                wall_y.append(-(x_w - 10) * np.tan(np.radians(5.352)))
        
        if wall_x:
            wall_x.extend([wall_x[-1], wall_x[0], wall_x[0]])
            wall_y.extend([y_min_global - 5, y_min_global - 5, wall_y[0]])
            ax.fill(wall_x, wall_y, color='gray', alpha=0.3)
    
    # Bind the helper method
    animate._draw_geometry_in_view = _draw_geometry_in_view
    
    anim = FuncAnimation(fig, animate, frames=len(x_positions), 
                        interval=250, blit=False, repeat=True)
    plt.tight_layout()
    return anim

# Side-by-side synchronized animation
def create_synchronized_animations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Find ranges for consistent scaling
    u_min = min([min(u_step) for u_step in u_data])
    u_max = max([max(u_step) for u_step in u_data])
    M_min = min([min(M_step) for M_step in M_data])
    M_max = max([max(M_step) for M_step in M_data])
    y_min = min([min(y_step) for y_step in y_data])
    y_max = max([max(y_step) for y_step in y_data])
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        current_x = x_positions[frame]
        
        # LEFT PLOT: Simple u velocity profile (u vs y)
        ax1.set_xlim(u_min * 0.9, u_max * 1.1)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('u velocity (m/s)')
        ax1.set_ylabel('y position (m)')
        ax1.set_title(f'U Velocity Profile - x = {current_x:.3f} m')
        ax1.grid(True, alpha=0.3)
        
        # Plot current u velocity profile
        if frame < len(u_data):
            ax1.plot(u_data[frame], y_data[frame], 'b-', linewidth=3, marker='o', markersize=4)
            
            # Add wall indicator at bottom
            ax1.axhline(y=y_data[frame][0], color='black', linewidth=4, label='Wall')
            ax1.legend()
        
        # Add position indicator
        ax1.text(0.02, 0.98, f'x = {current_x:.3f} m', transform=ax1.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.8), fontsize=12, fontweight='bold')
        
        # RIGHT PLOT: Moving camera with geometry, profiles, and expansion fan
        window_width = 25
        x_left = current_x - window_width/3
        x_right = current_x + 2*window_width/3
        
        # Calculate wall height at current position
        if current_x <= 10:
            wall_height = 0
        else:
            wall_height = -(current_x - 10) * np.tan(np.radians(5.352))
        
        ax2.set_xlim(x_left, x_right)
        ax2.set_ylim(wall_height - 1, wall_height + 20)
        ax2.set_xlabel('x position (m)')
        ax2.set_ylabel('y position (m)')
        ax2.set_title(f'Moving Camera with Profiles & Expansion Fan - x = {current_x:.3f} m')
        ax2.grid(True, alpha=0.3)
        
        # Draw wall geometry
        x_wall = np.linspace(max(0, x_left), min(L+10, x_right), 100)
        y_wall = []
        for x_w in x_wall:
            if x_w <= 10:
                y_wall.append(0)
            else:
                y_wall.append(-(x_w - 10) * np.tan(np.radians(5.352)))
        
        ax2.plot(x_wall, y_wall, 'k-', linewidth=4, label='Wall')
        ax2.fill_between(x_wall, y_wall, wall_height - 5, color='gray', alpha=0.3)
        
        # Add expansion fan lines if we're past the corner
        corner_x = 10
        corner_y = 0
        if current_x >= corner_x and x_right > corner_x:
            M_initial = 2.0
            M_final = 2.2
            
            mu_initial = np.arcsin(1/M_initial)
            mu_final = np.arcsin(1/M_final)
            
            x_start = max(corner_x, x_left)
            x_end = x_right
            
            # Leading edge line (steeper)
            y_lead_start = corner_y + (x_start - corner_x) * np.tan(mu_initial)
            y_lead_end = corner_y + (x_end - corner_x) * np.tan(mu_initial)
            ax2.plot([x_start, x_end], [y_lead_start, y_lead_end], 
                   'r-', linewidth=2, alpha=0.8, label='Leading edge')
            
            # Trailing edge line (shallower)  
            y_trail_start = corner_y + (x_start - corner_x) * np.tan(mu_final)
            y_trail_end = corner_y + (x_end - corner_x) * np.tan(mu_final)
            ax2.plot([x_start, x_end], [y_trail_start, y_trail_end], 
                   'r--', linewidth=2, alpha=0.8, label='Trailing edge')
        
        # Plot velocity and Mach profiles at current position
        if frame < len(u_data):
            # Scale velocity for display on the geometry plot
            u_scaled = u_data[frame] / u_max * (x_right - x_left) * 0.12
            profile_x = current_x + u_scaled
            
            ax2.plot(profile_x, y_data[frame], 'b-', linewidth=3, marker='o', markersize=3, label='U Velocity Profile')
            
            # Add velocity vectors and values
            skip = max(1, len(y_data[frame]) // 6)
            for j in range(0, len(y_data[frame]), skip):
                y_pos = y_data[frame][j]
                u_vel = u_data[frame][j]
                M_val = M_data[frame][j]
                
                # Draw horizontal arrow for u velocity
                arrow_length = u_vel / u_max * (x_right - x_left) * 0.08
                ax2.arrow(current_x, y_pos, arrow_length, 0, 
                        head_width=0.3, head_length=arrow_length*0.1, 
                        fc='red', ec='red', alpha=0.8, linewidth=1.5)
                
                # Add u velocity value
                ax2.text(current_x + arrow_length + 0.3, y_pos + 0.5, f'u:{u_vel:.0f}', 
                       fontsize=7, color='red', va='center', weight='bold')
                
                # Add Mach number value
                ax2.text(current_x + arrow_length + 0.3, y_pos - 0.8, f'M:{M_val:.2f}', 
                       fontsize=7, color='purple', va='center', weight='bold')
            
            # Add Mach profile line
            M_scaled = M_data[frame] / M_max * (x_right - x_left) * 0.08
            profile_x_M = current_x + M_scaled
            ax2.plot(profile_x_M, y_data[frame], 'purple', linewidth=2, marker='s', markersize=2, 
                   alpha=0.7, label='Mach Profile')
        
        ax2.legend(loc='upper right')
        
        return []
    
    anim = FuncAnimation(fig, animate, frames=len(x_positions), 
                        interval=200, blit=False, repeat=True)
    
    plt.tight_layout()
    return anim

# Create and run the synchronized side-by-side animation
sync_anim = create_synchronized_animations()
plt.show()
def create_mach_contour():
    """
    Creates Mach number contour plot
    """
    
    # Create mesh grid from all computed data
    X_mesh = []
    Y_mesh = []
    M_mesh = []
    rho_mesh = []
    p_mesh = []
    
    # Collect all data points from the simulation
    for i in range(len(x_positions)):
        x_pos = x_positions[i]
        y_pos = y_data[i]
        M_vals = M_data[i]
        rho_vals = rho_data[i]
        p_vals = p_data[i]
        
        for j in range(len(y_pos)):
            X_mesh.append(x_pos)
            Y_mesh.append(y_pos[j])
            M_mesh.append(M_vals[j])
            rho_mesh.append(rho_vals[j])
            p_mesh.append(p_vals[j])
    
    # Convert to numpy arrays
    X_mesh = np.array(X_mesh)
    Y_mesh = np.array(Y_mesh)
    M_mesh = np.array(M_mesh)
    rho_mesh = np.array(rho_mesh)
    p_mesh = np.array(p_mesh)
    
    # Create regular grid for contour plotting
    x_min, x_max = 0, max(x_positions)
    y_min_global = min([min(y_step) for y_step in y_data])
    y_max_global = max([max(y_step) for y_step in y_data])
    
    # Create grid
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min_global, y_max_global, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto regular grid
    from scipy.interpolate import griddata
    
    M_grid = griddata((X_mesh, Y_mesh), M_mesh, (Xi, Yi), method='linear')
    rho_grid = griddata((X_mesh, Y_mesh), rho_mesh, (Xi, Yi), method='linear')
    p_grid = griddata((X_mesh, Y_mesh), p_mesh, (Xi, Yi), method='linear')
    
    # Create wall mask to hide contours inside the wall
    wall_mask = np.zeros_like(Xi, dtype=bool)
    for i in range(len(xi)):
        for j in range(len(yi)):
            x_point = Xi[j, i]
            y_point = Yi[j, i]
            
            # Calculate wall height at this x position
            if x_point <= 10:
                wall_y = 0
            else:
                wall_y = -(x_point - 10) * np.tan(np.radians(5.352))
            
            # Mark points below wall as masked
            if y_point < wall_y:
                wall_mask[j, i] = True
    
    # Apply wall mask
    M_grid = np.ma.masked_where(wall_mask, M_grid)
    rho_grid = np.ma.masked_where(wall_mask, rho_grid)
    p_grid = np.ma.masked_where(wall_mask, p_grid)
    
    # Create Mach number plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    levels_M = np.linspace(np.nanmin(M_mesh), np.nanmax(M_mesh), 25)
    cs = ax.contourf(Xi, Yi, M_grid, levels=levels_M, cmap='jet', extend='both')
    ax.contour(Xi, Yi, M_grid, levels=levels_M, colors='black', linewidths=0.5, alpha=0.4)
    
    # Add wall geometry
    x_wall = np.linspace(0, max(x_positions), 1000)
    y_wall = []
    for x_w in x_wall:
        if x_w <= 10:
            y_wall.append(0)
        else:
            y_wall.append(-(x_w - 10) * np.tan(np.radians(5.352)))
    
    ax.fill_between(x_wall, y_wall, y_min_global - 5, color='gray', alpha=0.8, label='Wall')
    ax.plot(x_wall, y_wall, 'k-', linewidth=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_global, y_max_global)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Mach Number Contours', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label('Mach Number', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_density_contour():
    """
    Creates density contour plot
    """
    
    # Create mesh grid from all computed data
    X_mesh = []
    Y_mesh = []
    rho_mesh = []
    
    # Collect all data points from the simulation
    for i in range(len(x_positions)):
        x_pos = x_positions[i]
        y_pos = y_data[i]
        rho_vals = rho_data[i]
        
        for j in range(len(y_pos)):
            X_mesh.append(x_pos)
            Y_mesh.append(y_pos[j])
            rho_mesh.append(rho_vals[j])
    
    # Convert to numpy arrays
    X_mesh = np.array(X_mesh)
    Y_mesh = np.array(Y_mesh)
    rho_mesh = np.array(rho_mesh)
    
    # Create regular grid for contour plotting
    x_min, x_max = 0, max(x_positions)
    y_min_global = min([min(y_step) for y_step in y_data])
    y_max_global = max([max(y_step) for y_step in y_data])
    
    # Create grid
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min_global, y_max_global, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto regular grid
    from scipy.interpolate import griddata
    rho_grid = griddata((X_mesh, Y_mesh), rho_mesh, (Xi, Yi), method='linear')
    
    # Create wall mask to hide contours inside the wall
    wall_mask = np.zeros_like(Xi, dtype=bool)
    for i in range(len(xi)):
        for j in range(len(yi)):
            x_point = Xi[j, i]
            y_point = Yi[j, i]
            
            # Calculate wall height at this x position
            if x_point <= 10:
                wall_y = 0
            else:
                wall_y = -(x_point - 10) * np.tan(np.radians(5.352))
            
            # Mark points below wall as masked
            if y_point < wall_y:
                wall_mask[j, i] = True
    
    # Apply wall mask
    rho_grid = np.ma.masked_where(wall_mask, rho_grid)
    
    # Create density plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    levels_rho = np.linspace(np.nanmin(rho_mesh), np.nanmax(rho_mesh), 25)
    cs = ax.contourf(Xi, Yi, rho_grid, levels=levels_rho, cmap='plasma', extend='both')
    ax.contour(Xi, Yi, rho_grid, levels=levels_rho, colors='black', linewidths=0.5, alpha=0.4)
    
    # Add wall geometry
    x_wall = np.linspace(0, max(x_positions), 1000)
    y_wall = []
    for x_w in x_wall:
        if x_w <= 10:
            y_wall.append(0)
        else:
            y_wall.append(-(x_w - 10) * np.tan(np.radians(5.352)))
    
    ax.fill_between(x_wall, y_wall, y_min_global - 5, color='gray', alpha=0.8, label='Wall')
    ax.plot(x_wall, y_wall, 'k-', linewidth=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_global, y_max_global)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Density Contours', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label('Density (kg/m³)', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_pressure_contour():
    """
    Creates pressure contour plot
    """
    
    # Create mesh grid from all computed data
    X_mesh = []
    Y_mesh = []
    p_mesh = []
    
    # Collect all data points from the simulation
    for i in range(len(x_positions)):
        x_pos = x_positions[i]
        y_pos = y_data[i]
        p_vals = p_data[i]
        
        for j in range(len(y_pos)):
            X_mesh.append(x_pos)
            Y_mesh.append(y_pos[j])
            p_mesh.append(p_vals[j])
    
    # Convert to numpy arrays
    X_mesh = np.array(X_mesh)
    Y_mesh = np.array(Y_mesh)
    p_mesh = np.array(p_mesh)
    
    # Create regular grid for contour plotting
    x_min, x_max = 0, max(x_positions)
    y_min_global = min([min(y_step) for y_step in y_data])
    y_max_global = max([max(y_step) for y_step in y_data])
    
    # Create grid
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min_global, y_max_global, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto regular grid
    from scipy.interpolate import griddata
    p_grid = griddata((X_mesh, Y_mesh), p_mesh, (Xi, Yi), method='linear')
    
    # Create wall mask to hide contours inside the wall
    wall_mask = np.zeros_like(Xi, dtype=bool)
    for i in range(len(xi)):
        for j in range(len(yi)):
            x_point = Xi[j, i]
            y_point = Yi[j, i]
            
            # Calculate wall height at this x position
            if x_point <= 10:
                wall_y = 0
            else:
                wall_y = -(x_point - 10) * np.tan(np.radians(5.352))
            
            # Mark points below wall as masked
            if y_point < wall_y:
                wall_mask[j, i] = True
    
    # Apply wall mask
    p_grid = np.ma.masked_where(wall_mask, p_grid)
    
    # Create pressure plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    levels_p = np.linspace(np.nanmin(p_mesh), np.nanmax(p_mesh), 25)
    cs = ax.contourf(Xi, Yi, p_grid, levels=levels_p, cmap='viridis', extend='both')
    ax.contour(Xi, Yi, p_grid, levels=levels_p, colors='black', linewidths=0.5, alpha=0.4)
    
    # Add wall geometry
    x_wall = np.linspace(0, max(x_positions), 1000)
    y_wall = []
    for x_w in x_wall:
        if x_w <= 10:
            y_wall.append(0)
        else:
            y_wall.append(-(x_w - 10) * np.tan(np.radians(5.352)))
    
    ax.fill_between(x_wall, y_wall, y_min_global - 5, color='gray', alpha=0.8, label='Wall')
    ax.plot(x_wall, y_wall, 'k-', linewidth=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_global, y_max_global)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Pressure Contours', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label('Pressure (Pa)', fontsize=12)
    
    plt.tight_layout()
    return fig

contour=create_pressure_contour()
plt.show()
contour1=create_mach_contour()
plt.show()
contour2=create_density_contour()
plt.show()

