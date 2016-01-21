# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:53:04 2015

@author: carlos

Modulo que contiene las diferentes clases que componen a una 
columna de destilación
"""
from numericos import newton
import numpy as np
import matplotlib.pyplot as plt
'''
Rehervidor
Su única alimentación es el líquido que baja del primer plato.
Se añade además una cantidad de calor y se extraen dos corrientes:
una de fondos y otra de vapor.
La temperatura de referencia es 273.15K=0ºC
'''

class reboiler():
    def __init__(self,deb=False):
        self.deb=deb
        
        '''Entradas'''
        self.L=0;self.xl=0;self.hl=0;self.Pt=0
        '''Salidas'''
        self.V=0;self.H=0
        self.B=0;self.x=0;self.h=0
        
        self.M=0;self.T=0

        '''Propiedades de mezcla y substancias'''
        self.lamvapsubs=(854,2260)
        self.cpsubs=(2.42,4.18)
        self.tono=((8.20417,1642.89,230.3),(8.0713,1730.63,233.426))
        self.alpha=2.54
                      
        '''Constantes de Control'''
        self.kcb=0.7;self.tdb=0.9;self.Bref=9 ##Para fondos
        self.kcq=0.5;self.tdq=0.5 ##Para reboiler
        self.Mref=30##Para la masa del interior del reboiler
              
        '''Ecuaciones diferenciales'''
        self.dhdt=lambda M,B,h,Q,x:(self.L*self.hl - (self.Qvap*Q/self.lamvap_f(x)*(h+self.lamvap_f(x))+B*h) + Q - h * (self.L-(Q/self.lamvap_f(x)+B)))/M
        self.dMdt=lambda B,Q,x:self.L-(self.Qvap*Q/self.lamvap_f(x)+B)
        self.dQdt=lambda M,B,Q,x:-self.kcq*self.lamvap_f(x)*(self.Mref-M-self.tdq*(self.L-(self.Qvap*Q/self.lamvap_f(x)+B)))
        self.dBdt=lambda B:self.kcb*(self.Bref-B)/(1+self.tdb*self.kcb)
        self.dxdt=lambda M,B,Q,x:(self.L*self.xl-(Q/self.lamvap_f(x)*self.equil(x)+B*x)-x*(self.L-(self.Qvap*Q/self.lamvap_f(x)+B)))/M

        '''Ecuaciones no diferenciales'''
        self.lamvap_f=lambda x:(self.lamvapsubs[0]-self.lamvapsubs[1])*x+self.lamvapsubs[1]
        self.cp=lambda x:(self.cpsubs[0]-self.cpsubs[1])*x+self.cpsubs[1]
        self.equil=lambda x:self.alpha*x/(1+(self.alpha-1)*x)

    def Teb(self):
        def tonof(indice,T):
            A,B,C=self.tono[indice]
            return(10**(A-B/(C+T)))
        def fobj(T):
            Pa=tonof(0,T)
            Pb=tonof(1,T)
            x= (self.Pt-Pb)/(Pa-Pb)-self.x
            return(x)
        def teb(T):
            return(tonof(0,T)-self.Pt)
        return(newton(fobj,newton(teb,0)))
        
    def condini(self):
        ## Condiciones iniciales:
        self.L=20;self.xl=0.8
        self.M=30.;self.Q=1.1e4;self.B=9
        ## Estado Estacionario: self.M=30;self.Q=1.08e4;self.B=9;self.V=11
        self.x=self.xl;self.y=self.equil(self.x);lamvap=self.lamvap_f(self.x)
        self.hl=220.;self.h=self.hl;self.H=self.h+lamvap
        self.T=self.h/self.cp(self.x);self.V=self.Q/self.lamvap_f(self.x)
        self.Pt=760.##mmHg
        
    def actualizar(self,t,paso=0.1):
        Teb=self.Teb()
        M=self.M;B=self.B;h=self.h;x=self.x;Q=self.Q
        
        if self.T>=Teb:
            self.Qvap=1
        else:
            self.Qvap=0
            
        k11=self.dhdt(M,B,h,Q,x)
        k12=self.dMdt(B,Q,x)
        k13=self.dQdt(M,B,Q,x)
        k14=self.dBdt(B)
        k15=self.dxdt(M,B,Q,x)
        k21=self.dhdt(M+paso*k12/2,B+paso*k14/2,h+paso*k11/2,Q+paso*k13/2,x+paso*k15/2)
        k22=self.dMdt(B+paso*k14/2,Q+paso*k13/2,x+paso*k15/2)
        k23=self.dQdt(M+paso*k12/2,B+paso*k14/2,Q+paso*k13/2,x+paso*k15/2)
        k24=self.dBdt(B+paso*k14/2)
        k25=self.dxdt(M+paso*k12/2,B+paso*k14/2,Q+paso*k13/2,x+paso*k15/2)
        k31=self.dhdt(M+paso*k22/2,B+paso*k24/2,h+paso*k21/2,Q+paso*k23/2,x+paso*k25/2)
        k32=self.dMdt(B+paso*k24/2,Q+paso*k23/2,x+paso*k25/2)
        k33=self.dQdt(M+paso*k22/2,B+paso*k24/2,Q+paso*k23/2,x+paso*k25/2)
        k34=self.dBdt(B+paso*k24/2)
        k35=self.dxdt(M+paso*k22/2,B+paso*k24/2,Q+paso*k23/2,x+paso*k25/2)
        k41=self.dhdt(M+paso*k32,B+paso*k34,h+paso*k31,Q+paso*k33,x+paso*k35)
        k42=self.dMdt(B+paso*k34,Q+paso*k33,x+paso*k35/2)
        k43=self.dQdt(M+paso*k32,B+paso*k34,Q+paso*k33,x+paso*k35)
        k44=self.dBdt(B+paso*k34)
        k45=self.dxdt(M+paso*k32,B+paso*k34,Q+paso*k33,x+paso*k35)

        if M>0:
            dx=paso*(k15+2*k25+2*k35+k45)/6
            dM=paso*(k12+2*k22+2*k32+k42)/6
        else:
            dx=0;dM=0
        self.x+=dx
        self.M+=dM
        self.Q+=paso*(k13+2*k23+2*k33+k43)/6
        self.h+=paso*(k11+2*k21+2*k31+k41)/6
        self.B+=paso*(k14+2*k24+2*k34+k44)/6
        
        if self.Q<0:
            self.Q=0
        self.V=self.Qvap*self.Q/self.lamvap_f(self.x)
        self.T=self.h/self.cp(self.x)
        
        
    def simular(self,controladores,Lvar,dt=0.01,plot=0):
        self.kcb,self.tdb,self.kcq,self.tdq=controladores
        self.condini()
        t=0
        Ml,Bl,Vl,tl,Tl,xli=[self.M],[self.B],[self.V],[t],[self.T],[self.x]
        for i in range(len(Lvar)):
            t+=dt
            self.L=float(Lvar[i])
            self.actualizar(t,dt)
            Ml.append(self.M),Bl.append(self.B),Vl.append(self.V),Tl.append(self.T),tl.append(t),xli.append(self.x)

        if plot==1:
            print('Graficando la simulacion')
            plt.figure(figsize=(16,10))
               
            plt.subplot(2,2,1);plt.grid(True)
            plt.plot(tl,Ml,'b.',label='Acumulacion')
            plt.plot(tl,Bl,'g.',label='Fondos')
            plt.plot(tl,Vl,'r.',label='Vapor')
            plt.xlabel('tiempo');plt.ylabel('kg/min')
            plt.legend(loc=4)
            
            plt.subplot(2,2,2);plt.grid(True)
            plt.plot(tl,Tl,'b.')
            plt.xlabel('tiempo');plt.ylabel('Temperatura')
            
            plt.subplot(2,2,3);plt.grid(True)
            plt.title('Perturbacion')
            plt.plot(Lvar)
            plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')
            
            plt.subplot(2,2,4);plt.grid(True)
            plt.title('FraccionMolar')
            plt.plot(tl,xli,'b.')
            plt.xlabel('tiempo');plt.ylabel('X')
            
            plt.show()
        
        return(np.trapz(np.abs(np.array(Ml[-25:])-20)))
        
if __name__=='__main__':
    """
    Cambia de Script
    """