# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:07:23 2015

Compilación de métodos numéricos

@author: carlos
"""
from __future__ import division

def newton(f,x0,it=20,dstep=0.0001):
    x=x0
    for i in range(it):
        df=(f(x+dstep)-f(x-dstep))/dstep
        if df==0:
            break
        x=x-f(x)/df
        #print 'Nuevo valor:',x
    return(x)

def FalsaPosicion(f,x1,x2,xr=0,it=40,tol=1e-5):
    f1=f(x1)
    f2=f(x2)
    repeticiones1=0;repeticiones2=0
    for i in range(it):
        xr_viejo=xr
        xr=x2-(f2*(x1-x2))/(f1-f2)
        error=abs((xr_viejo-xr)/xr)
        fr=f(xr)
        comprobacion=f1*fr
        if comprobacion<0:
            x2=xr;f2=fr
            repeticiones2=0
            repeticiones1+=1
            if repeticiones2==2: f1=f1/2
        elif comprobacion>0:
            x1=xr;f1=fr
            repeticiones1=0
            repeticiones2+=1
            if repeticiones1==2: f2=f2/2
        else:
            error=0
        
        if error<=tol:
            break
    
    return xr
        
        
        
def Euler(f, x=0,t=0, step=0.1, inter=[0,10]):
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
    pass
