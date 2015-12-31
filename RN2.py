# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:49:02 2015

@author: carlos
Se creará la clase de Red Neuronal
"""
from __future__ import division
import numpy as np
import numpy.random as rnd

class RedNeuronal(object):
    def __init__(self,estructura,datos_de_entrenamiento=None,neurodos_entrada_salida=None,deb=False,bias=1,seed=None):
        ## La estructura de la red deberá ser configurada como una tupla con
        ## los neurodos por capa que se desee que tenga la red, por ejemplo:
        ##    Para una red con 3 capas ocultas y 5 neurodos por capa se deberá
        ##    ingresar una tupla como la siguiente: (5,5,5)
        self.estructura=estructura[::]
        self.datos_de_entrenamiento=datos_de_entrenamiento
        self.neurodos_entrada_salida=neurodos_entrada_salida
        
        ## Variables
        self.pesos=None
        self.bias=bias
        self.deb=deb
        self.seed=seed
    
    def FP(self,pesos=None,xi=None,seed=None,estructura=None):
        if not estructura:
            estructura=self.estructura
        if not isinstance(pesos,(np.ndarray,list)):
            pesos=self.pesos
            
        ##Ya que los pesos y los datos son correctos, se hace la propagación
        ##hacia adelante
        activaciones=[xi]
        for capa in range(len(estructura)-1):
            x=activaciones[capa]
            x=np.append(x,[[self.bias]])
            z=np.dot(pesos[capa],x)
            activaciones.append(self.sigmoide(z))

        return(activaciones)

    
    def Entrenar(self,xi,yi,pesos=None,alpha=0.1,max_iter=100000,seed=None):
        if not seed: seed=self.seed
        ##Se confirma que los datos ingresados sean correctos, además de obtner
        ##el número de datos de entrada y de salida
        nx,mx=verificarDatosEnt(xi);ny,my=verificarDatosEnt(yi)
        if mx!=my:
            raise ValueError('Las dimensiones de muestras de x y y no concuerdan')
        ##Se copia la estructura predeterminada de la red y se añade
        ##la cantidad de neurodos en la entrada y a la salida
        nueva_estructura=self.estructura[::]
        nueva_estructura.insert(0,nx);nueva_estructura.append(ny)
        self.estructura=nueva_estructura
        matrices,npesos=dimensionarMatricesDePesos(nueva_estructura)
        if self.deb:print 'Estructura/Matrices:',nueva_estructura,matrices
        ##Si no se ingresan los pesos, se crean aleatoriamente unos
        if not isinstance(pesos,np.ndarray):
            pesos=self.AsignarPesos(nueva_estructura,seed)
            if self.deb:print 'Pesos asignados'
        else:
            ##Se verifican los pesos para que puedan ser utilizados
            verificarPesos(nueva_estructura,pesos)
            if self.deb:print 'Pesos verificados'

        ## Entrenamiento
        if self.deb:print '\nInicio de entrenamiento :D\n'
        for ent in range(max_iter):
            ##Seleccion de datos de entrenamiento
            num_entrenamiento=rnd.randint(0,mx)
            x=xi[num_entrenamiento];y=yi[num_entrenamiento]
            ##Propagacion hacia adelante
            activaciones=self.FP(pesos=pesos,xi=x,seed=seed)
            y_red=activaciones[-1]
            error=y-y_red
            ##Se calculan y almacenan las deltas de cada capa
            ##Para la capa final:
            d_final=(np.atleast_2d(error*self.sig_prim(y_red))).T
            deltas=[d_final]
            ##Para las demás capas:
            for i in range(len(activaciones)-2,0,-1):
                filas=range(len(pesos[i].T)-1)
                wi=(pesos[i].T)[filas,:]
                act=np.array([activaciones[i]]).T
                delta=np.dot(wi,deltas[-1])*self.sig_prim(act)
                deltas.append(delta)
                
            ##Se invierten para facilidad de uso posterior
            deltas.reverse()
            
            ##Actualizacion de pesos en linea
            for i in range(len(pesos)):
                act=np.atleast_2d(np.append(activaciones[i],[self.bias]))
                delta_pesos=np.dot(deltas[i],act)
                pesos[i]+=alpha*delta_pesos
                
            if ent%10000==0 and self.deb: print '\nIteracion:',ent,' error:',.5*np.sum(error)**2,'\n',x,y_red,'->',y
        
        print 'Fin del entrenamiento'
            
        self.pesos=pesos
        return(pesos)

    
    def AsignarPesos(self,estructura,seed=None):
        rnd.seed(seed)
        ##Si no hay pesos asignados se crea la matriz de pesos aleatorios
        est,num_total_pesos=dimensionarMatricesDePesos(estructura)
        pesos_sin_formato=2*rnd.rand(num_total_pesos)-1
        pesos=redimensionarPesos(pesos_sin_formato,est)
        return(pesos)
            
    def sigmoide(self,a):
        return(1.0/(1.0+np.exp(-a)))
    
    def sig_prim(self,a):
        return(a*(1.-a))
        
        
    def __str__(self):
        return('Red Neuronal')

'''
-------------------------------------------------------------------------------
'''
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

def verificarPesos(est,pesos):
    matriz=dimensionarMatricesDePesos(est)[0]
    for i in range(len(pesos)):
        if matriz[i]!=pesos[i].shape:
            raise ValueError('La matriz de pesos no concuerda don la estructura')
        
def dimensionarMatricesDePesos(arq):
    matrices=[]
    nPesos=0
    for i in range(1,len(arq)):
        ##Se le suma el uno para contar el neurodo bias
        matrices.append((arq[i],arq [i-1]+1))
        nPesos+=arq[i]*(arq[i-1]+1)
    
    return(matrices,nPesos)
       
def redimensionarPesos(pesos,estructura):
    W=list()
    lant=0
    cont=0
    for i in estructura:
        l=i[0]*i[1]
        W.append(np.reshape(pesos[lant:lant+l],i))
        lant+=l;cont+=1
    
    return(W)

'''
-------------------------------------------------------------------------------
'''


if __name__=='__main__':
    est=[2]#real es [2,2,2,1]
    nn=RedNeuronal(est,deb=True,bias=1)
    xi=np.array([[0,0],
                 [1,0],
                 [0,1],
                 [1,1]])
    yi=np.array([[0],
                 [1],
                 [1],
                 [0]])
#    xi=np.array([[.05,.1]])
#    yi=np.array([[.01,.99]])
    nuevos_pesos=nn.Entrenar(xi=xi,yi=yi,max_iter=100001,alpha=0.1)
    res=nn.FP(xi=[xi[0]],pesos=nuevos_pesos)
    print '\nPredicciones'
    print xi[0],np.round(res[-1])
    res=nn.FP(xi=[xi[1]],pesos=nuevos_pesos)
    print xi[1],np.round(res[-1])
    res=nn.FP(xi=[xi[2]],pesos=nuevos_pesos)
    print xi[2],np.round(res[-1])
    res=nn.FP(xi=[xi[3]],pesos=nuevos_pesos)
    print xi[3],np.round(res[-1])
    
