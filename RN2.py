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
    def __init__(self,estructura,datos_de_entrenamiento=(None,None),deb=False):
        ## La estructura de la red deberá ser configurada como una tupla con
        ## los neurodos por capa que se desee que tenga la red, por ejemplo:
        ##    Para una red con 3 capas ocultas y 5 neurodos por capa se deberá
        ##    ingresar una tupla como la siguiente: (5,5,5)
        self.estructura=estructura[::]
        self.n_capas_ocultas=len(estructura)
        self.pesos_asignados=False
        self.capas=[Capa() for x in range(len(estructura)+2)] ##Se añade la capa de salida y entrada
        self.x_ent,self.y_ent=datos_de_entrenamiento if datos_de_entrenamiento else (None,None)
        
        ## Variables
        self.error_total=None
    
    def FP(self,pesos=None,xi=None,yi=None):
        if not pesos and not self.pesos_asignados:
            est,num_total_pesos=dimensionarMatricesDePesos(self.estructura)
            pesos_sin_formato=[rnd.random() for x in range(num_total_pesos)]
            pesos=redimensionarPesos(pesos_sin_formato,est)
            print pesos
            
            '''
            HAY QUE HACER QUE LA ESTRUCTURA DE LA RED SEA CREADA ANTES DE LA FP
            '''
    
    def CalcErrorTotal(self):
        pass
    
    def Entrenar(self):
        pass
    
    def IdentificarEstructura(self):
        pass
    
    def AsignarPesos(self,pesos):
        contador=1
        for conjunto in pesos:
            self.capas[contador].pesos=conjunto
            
            
    def __str__(self):
        return('Red Neuronal')
        
        
        

class Capa(object):
    def __init__(self):
        self.num_capa=None
        self.pesos=None
        self.num_neurodos=None
        self.respuestas=list()
    
    def CalcularSalidas(self):
        pass
    
    def ConfigurarPesos(self,pesos_iniciales=None):
        if not pesos_iniciales:
            pass
    

class Neurodo(object):
    def __init__(self):
        self.num_neurodo=None
        self.num_capa=None
        self.respuesta=None
        self.pesos=list()
    
    def FuncionRespuesta(self):
        pass
    
    def Respuesta(self):
        pass

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
    
def dimensionarMatricesDePesos(arq):
    matrices=[]
    nPesos=0
    for i in range(1,len(arq)):
        matrices.append((arq[i],arq [i-1]+1))
        nPesos+=arq[i]*(arq[i-1]+1)
    
    return(matrices,nPesos)
       
def redimensionarPesos(pesos,estructura):
    W=list()
    lant=0
    cont=0
    for i in estructura:
        W.append(list())
        l=i[0]*i[1]
        W[cont].append(np.reshape(pesos[lant:lant+l],i))
        lant+=l;cont+=1
    
    return(W)

if __name__=='__main__':
    est=[2,2,3]
    nn=RedNeuronal(est)
    print dimensionarMatricesDePesos(est)
    print nn.estructura
    print nn.n_capas_ocultas
    print nn.x_ent