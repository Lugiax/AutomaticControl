# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:07:23 2015

Compilación de métodos numéricos

@author: carlos
"""

def newton(f,x0,it=20,dstep=0.001):
    x=x0
    for i in range(it):
        df=(f(x+dstep)-f(x-dstep))/dstep
        if df==0:
            break
        x=x-f(x)/df
    return(x)
    
def RK4(f, x=0,t=0, step=0.1, inter=[0,10]):
    if isinstance(f, (tuple, list)):
        print('Functions')
        tlist=[z*step for z in range(inter[0],int(inter[1]/step))]
        xlist=[]
        for function in f:
            xlist.append(RK4(function)[1])
        
        return(tlist,xlist)
            
    else:
        print('function')
        xlist=[]
        tlist=[]
        print('Range:',inter[0],' a' ,inter[1]/step)
        for i in range(inter[0], int(inter[1]/step)):
            print('t:',t)
            k1=step*f(x,t)
            k2=step*f(x+0.5*k1,t+0.5*step)
            k3=step*f(x+0.5*k2,t+0.5*step)
            k4=step*f(x+k3,t+step)
            
            x+=1/6*(k1+2*k2+2*k3+k4)
            t+=step
            xlist.append(x)
            tlist.append(t)
            
        return(tlist,xlist)
        
def RK4s(f, x=0, t=0, step=0.1):
    k1=step*f(x,t)
    k2=step*f(x+0.5*k1,t+0.5*step)
    k3=step*f(x+0.5*k2,t+0.5*step)
    k4=step*f(x+k3,t+step)
    x+=1/6*(k1+2*k2+2*k3+k4)
    return(x)
    
def EulerInt(f, x=0,t=0, step=0.1, inter=[0,10]):
    if isinstance(f, (tuple, list)):
        print('Functions')
    else:
        print('function')
        xlist=[]
        tlist=[]
        
        for i in range(inter[0], int(inter[1]/step)):           
            x+=f(x,t)*step
            t=step
            xlist.append(x)
            tlist.append(t)
            
        return(tlist,xlist)

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    step=0.0001   
    V=0
    t=0
    f0= lambda t: 1-np.sin(t)/1
    f= lambda t: 0.9*(1+np.cos(t)/1)
    rho0=1.5
    rho= lambda V: rho0+0.01*V
    fp= lambda V, t: f0(t)*rho0-f(t)*rho(V)
    
    fp2= lambda V, t: f0(t)*rho0-2-f(t)*rho(V)+1
    
    x,y=RK4((fp, fp2), step=step)
    
    plt.Figure()
    plt.subplot(121)
    plt.plot(x,y[0])
    plt.subplot(122)
    plt.plot(x,y[1])
    plt.show()
    
    '''
    x,y=EulerInt(fp, step=step)
    
    plt.subplot(122)
    plt.plot(x,y)
    plt.show()
    '''
    
    
   
