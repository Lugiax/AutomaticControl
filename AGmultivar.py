# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:25:04 2015

@author: carlos

Algorítmo Genético con variables múltiples
"""

from math import log
import random as rnd
from binstr import b_bin_to_gray, b_gray_to_bin


class AG(object):
    
    def __init__(self,deb=False,seed=None):
        
        self.deb=deb ## Para depurar el programa  
        if seed:
            rnd.seed(seed)
        
        self.dxmax=0.01 # Presición
        self.Nind=50 # Número de individuos por default
        self.Ngen=100 # Número de generaciones por default
        self.prop_cruz=0.3 # Proporción de cruzamiento entre parejas por default
        self.prob_mut=0.05 # Probabilidad de mutación de un individuo por default
        self.elit=1 # Probabilidad de ser elitista (1 es elitismo total)
        
        self.hist_mej=[[],[],[]] ## Variable para guardar la historia del mejor individuo [Generación, Fitness, x]
            
        if deb:print('\nSe ha iniciado con el algoritmo, favor de introducir los parámetros')
            
        
        ## Funciones de codificación y decodificación
        self.decod= lambda n, a, b: (int(b_gray_to_bin(n),2)*int((b-a)/self.dxmax)/self.dmax+int(a/self.dxmax))*self.dxmax
        self.cod= lambda n: b_bin_to_gray(("{:0>"+str(self.l)+"}").format(bin(n).lstrip("0b")))

    def decodificado(self, genoma):
    	dec=list()
    	for i in range(self.nvars):
    		if self.comun:
    			a=self.a
    			b=self.b
    		else:
    			a=self.vars[i][1]
    			b=self.vars[i][2]
    		#print(genoma[i*self.l:(i+1)*self.l])
    		#print(int(genoma[i*self.l:(i+1)*self.l],2),int((self.b_a)/self.dxmax)/self.dmax,int(a/self.dxmax))
    		dec.append(self.decod(genoma[i*self.l:(i+1)*self.l],a,b))
    	return(dec)
        
        
    def parametros(self, pres=None, Nind=None, Ngen=None, prop_cruz=None, prob_mut=None, elit=1, optim=1, tipo_cruz='2p'):
        if pres:
            self.dxmax=pres
        if Nind:
            self.Nind=Nind+Nind%2
        if Ngen:
            self.Ngen=Ngen
        if prop_cruz:
            self.prop_cruz=prop_cruz
        if prob_mut:
            self.prob_mut=prob_mut
        if optim==1:
            self.max=True
            if self.deb:print('Máximo')
        else:
            self.max=False
            if self.deb:print('Mínimo')
        
        self.tipo_cruz=tipo_cruz
        self.elit=elit
        
    def Fobj(self, f, datos=None): ## Introducir la función a evaluar
        self.f_obj=f
        self.datos=datos
        
        if self.deb:print('Se ha introducido correctamente la función objetivo')
    
    def variables(self, variables=None, comun=None):
    	##Si se ingresa la variable comun, deberá tener la siguiente forma:
    	##  comun=[#deVariables, lim_inferior, lim_superior]
    	if comun:
    		dabmax=comun[2]-comun[1]
    		self.nvars=comun[0]
    		self.comun=True
    		self.a=comun[1]
    		self.b=comun[2]
    	elif variables:
    		dabmax=0
    		self.vars=variables
    		for var in variables:
    		    dab=var[2]-var[1]
    		    if dab>dabmax: dabmax=dab
    		self.nvars=len(variables)
    		self.comun=False
         
    	self.b_a=dabmax	
    	self.l=int(log(((dabmax)/self.dxmax)+1,2))+1
    	self.dmax=2**self.l-1
    	if self.deb==True:print('Ingreso de variables Exitoso ',dabmax,self.l,self.dmax)
            
            

    def fitnes(self, pob):
        result=[list(), list()]
        for ind in pob:
            #print(f(decod(ind)))
            if self.datos:
                fit_ind=self.f_obj(self.decodificado(ind),self.datos)
            else:
                fit_ind=self.f_obj(self.decodificado(ind))
            #print(ind, fit_ind, decod(ind)*dxmax)
            #print('Fitnes',pob.index(ind),fit_ind)
            result[1].append(fit_ind)
            result[1]=sorted(result[1], reverse=self.max)
            indice=result[1].index(fit_ind)
            result[0].insert(indice, ind)

            
        return(result)

    def crearPob(self,N_ind=None, d_max=None):
        pob=[]
        
        if not N_ind:
            N_ind=self.Nind
        if not d_max:
            d_max=self.dmax
            
        for i in range(N_ind):
            genoma=''
            for j in range(self.nvars):
                rand=rnd.randrange(0,d_max)
                genoma+=self.cod(rand)
            #print('Lgenoma',len(genoma))
            pob.append(genoma)
        
        return(pob)


    def cruzamiento(self, fit,tipo='2p'):
    ## El cruzamiento se hará de tipo Vasconcelos
        pob1=list()
        
        #print('Cruzamiento Nind:',int(self.Nind/2))
        for i in range(int(self.Nind/2)):
            #print('\nSeleccion',i+1)
            ind1_0=fit[0][i]
            ind2_0=fit[0][self.Nind-i-1]
            #print(ind1_0, ind2_0)
            ind1=''
            ind2=''
            if tipo=='uniforme':
                for bit in range(len(ind1_0)):
                    nbit1=ind1_0[bit]
                    nbit2=ind2_0[bit]
                    dados=rnd.random()
                    #print (dados)
                    if dados<self.prop_cruz:
                        temp=nbit1
                        nbit1=nbit2
                        nbit2=temp
                    
                    ind1+=nbit1
                    ind2+=nbit2
                    
                #print(ind1_0,ind2_0)  
                #print(ind1, ind2)
            elif tipo=='2p':
                l=self.l*self.nvars
                rs=(rnd.randrange(l),rnd.randrange(l))
                r1=min(rs)
                r2=max(rs)
                if r1==r2 or (r1==0 and r2==l-1):
                    ind1=ind1_0
                    ind2=ind2_0
                else:
                    temp=ind1_0[r1:r2]
                    ind1=ind1_0[:r1]+ind2_0[r1:r2]+ind1_0[r2:]
                    ind2=ind2_0[:r1]+temp+ind2_0[r2:]
#                    print(temp,r1,r2,l)
#                    print(ind1_0,ind2_0,(len(ind1_0),len(ind2_0)))
#                    print(ind1,ind2,(len(ind1),len(ind2)))

            pob1.append(ind1)
            pob1.append(ind2)
        
        #print('Cruzamiento lpob:',len(pob1))
        return(pob1)


    def mutacion(self, pob):
                #print('\Mutación')
        invert={'0':'1','1':'0'}
        
        pob_mut=list()
        for ind in pob:
            ind_mut=''
            for bit in ind:
                nbit=bit
                if rnd.random()<self.prob_mut:
                    nbit=invert.get(bit)
                ind_mut+=nbit
                
            pob_mut.append(ind_mut)
            
        return(pob_mut)
    
    def elitismo(self, fit, fit1):
        fit_max=fit[1][0]
        #print('Elitismo:',len(fit1[0]),len(fit1[1]))
        for ind in range(self.Nind):
            if self.max:
                if fit1[1][ind]>fit_max or rnd.random()>self.elit:
                    continue
                if fit1[1][ind]<fit[1][ind]:
                    fit1[1][ind]=fit[1][ind]
                    fit1[0][ind]=fit[0][ind]
                    
            elif not self.max:
                if fit1[1][ind]<fit_max or rnd.random()>self.elit:
                    continue
                if fit1[1][ind]>fit[1][ind]:
                    fit1[1][ind]=fit[1][ind]
                    fit1[0][ind]=fit[0][ind]
        
        return(fit1)
    

    def start(self):
        
        ##Creación de individuos!!!!!!!!!!!!!!!!!!!!!
        if self.deb:print('\nCreación de individuos')
        self.pob=self.crearPob()

        
        ##Acomodo de acuerdo al desempeño!!!!!!!!!!!!
        if self.deb:print('\nAcomodo de individuos por habilidad')
        fit=self.fitnes(self.pob)
         
        for gen in range(self.Ngen):    
            if self.deb:print('\nGeneracion:',gen+1)
            #print(fit)
            
            #print("\nCruzamiento")
            
            ## Cruzamiento!!!!!!!!!!!!!!!!!!!!
            pob1=self.cruzamiento(fit,tipo=self.tipo_cruz)
                        
            
            ##Mutación!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
            pob_mut=self.mutacion(pob1)
            ##se vuelve a probar la nueva población
            fit1=self.fitnes(pob_mut)
            
            ### Elitismo !!!!!!!!!!!!!!!!!!!!!
            fit=self.elitismo(fit, fit1)

            self.hist_mej[0].append(gen)
            self.hist_mej[1].append(fit[1][0])
            self.hist_mej[2].append(fit[0][0])
            mejor=(self.decodificado(fit[0][0]),fit[1][0])
            if self.deb:print('Mejor Individuo:',fit[0][0],' vars=',mejor[0],' f=',mejor[1] )
        
        return(mejor)
            
if __name__=='__main__':
    
    print('Hola!!!')