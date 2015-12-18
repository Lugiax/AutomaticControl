# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:40:50 2015

@author: carlos
"""

import numpy as np


class RN(object):
    
    def __init__(self,deb=False):
        self.deb=deb
        ## Parámetros de red:
        self.bias=-1
        
        ##Matrices extra
#        self.xelim=[]
#        self.xenorm=[]
#        self.yelim=[]
#        self.yenorm=[]
        self.nxe=None;self.nye=None
        
        self.sig= lambda mat: 1/(1+np.exp(-mat))
        
    def parametros(self,arq=None,W=None,arq_io=None):
        self.arq=arq
        self.W=W
        if arq_io:
            self.arq.insert(0,arq_io[0])
            self.arq.append(arq_io[1])
            self.N=len(self.arq)
            


    def datosEnt(self,datos):
        self.xe=datos[0];self.ye=datos[1]
        #Se verifica que el numero de variables en cada conjunto de datos sea el mismo y se almacena
        self.nxe,mx=verificarDatosEnt(self.xe);self.nye,my=verificarDatosEnt(self.ye)
        if mx!=my:
            raise ValueError('Diferente numero de muestras en x-y')
        self.m=mx#Se almacena el número de muestras de los datos
#        almacenar(self.xe,self.xenorm,self.xelim)
#        almacenar(self.ye,self.yenorm,self.yelim)
        #Se almacena el número de variables por observación
#        self.nxe=len(self.xe[0])
#        self.nye=len(self.ye[0])
        if self.deb:print('No. DatEnt:',self.xe,'-',self.ye)
        self.arq.insert(0,self.nxe)
        self.arq.append(self.nye)
        self.N=len(self.arq)
        
    def generarPesos(self,seed=None):
        ## Generación de los pesos:
        np.random.seed(seed)
        self.W=dict()
        for capa in range(1,self.N):
            nxcapa=self.arq[capa]
            nxcapa_anterior=self.arq[capa-1]  
            #print nxcapa, nxcapa_anterior
            self.W[str(capa)]=np.random.random([nxcapa,nxcapa_anterior+1])
            #print w[str(capa)]
    
        
    def FP(self,W=None,xi=None):
        Ys=list()#Lista para almacenar las salidas de cada capa
        yi=list()
        self.error=0
        if not W:
            W=self.W
        if not isinstance(xi,np.ndarray):
            xi=self.xe
        ## Propagación hacia adelante
        for m in range(len(xi)): #Por cada serie de datos de entrenamiento
            if self.deb:print('\nMuestra%d'%(m+1))
            Ys.append(list())#Se agrega una nueva lista donde almacenar las salidas
            for i in range(1,self.N):#Por cada capa posterior a la primera
                           
                if self.deb:print('\nFP:',i)
                if i==1:#si es la segunda capa se reciben los datos de entrada en xe
                    x=xi[m]
#                    x=list()#Se crea la lista de datos de entrada
#                    for j in range(self.nxe):
#                        x.append(self.xe[m][j])    
                else:
                    x=Ys[m][i-2]
 
                x=np.append(x,[self.sig(self.bias)])
                v=np.dot(W[str(i)],np.array(x))
                y=self.sig(v)[0]
                
                Ys[m].append(y)
                if self.deb:print('x{}{}: {}'.format(m,i,x))
                if self.deb:print('Pesos\n',W[str(i)])
                if self.deb:print('y{}{}: {}'.format(m,i,Ys[m][i-1]))
            
            if self.nye:            
                for y in range(self.nye):
                    yred=Ys[m][-1][y]
                    ye=self.ye[m][y]
                    er=0.5*(yred-ye)**2
                    self.error+=er
                    if self.deb:print('Error{}-> red:{:.4f} - ent:{:.4f} - error:{:.4f}'.format(y,yred,ye,er))
            
            yi.append(Ys[m][-1])

        if self.deb:print('Error Tot',self.error)
        return(yi,self.error)
            
     
        
#def almacenar(datos,norm,lim,cont=0):
#    print('Datos',datos,type(datos[0]))
#    if isinstance(datos[cont],(np.float32,np.float64,np.int32,np.int64,)):        
#        lim.append([min(datos),max(datos)])
#        norm.append((datos-lim[cont][0])/(lim[cont][1]-lim[cont][0]))
#    elif isinstance(datos[0],np.ndarray):
#        for i in range(len(datos)):
#            almacenar(datos[i],norm,lim,cont=i)
 
def verificarDatosEnt(datos):
    l=None
    m=0
    for i in datos:
        li=len(i)
        if not l:
            l=li
        else:
            if li!=l:
                raise ValueError('La configuración de los datos no es la misma')
                break
        m+=1
    return(l,m)
    
def dimensionarMatricesPesos(arq):
    matrices=[]
    nPesos=0
    for i in range(1,len(arq)):
        matrices.append((arq[i],arq [i-1]+1))
        nPesos+=arq[i]*(arq[i-1]+1)
    
    return(matrices,nPesos)
       
def redimensionarPesos(pesos,estructura):
    W=dict()
    lant=0
    cont=1
    for i in estructura:
        W[str(cont)]=list()
        l=i[0]*i[1]
        W[str(cont)].append(np.reshape(pesos[lant:lant+l],i))
        lant+=l;cont+=1
    
    return(W)
        

    
if __name__=='__main__':
    print('Hola :D')

    xe=np.array([[0.1],
                 [0.9]])
    ye=np.array([[0.05],[0.1]])
    red=RN(True)
    red.parametros(arq=[2,2])
    red.datosEnt((xe,ye))
    red.generarPesos(1)
    print('Arquitectura:',red.arq)
    y,err=red.FP()
    print(err)


