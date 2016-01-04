# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:49:02 2015

@author: carlos
Se creará la clase de Red Neuronal
"""
from __future__ import division
import numpy as np
import numpy.random as rnd
from AGmultivar import AG


class RedNeuronal(object):
    def __init__(self,estructura,datos_de_entrenamiento=None,neurodos_entrada_salida=None,deb=False,bias=-1,seed=None):
        ## La estructura de la red deberá ser configurada como una tupla con
        ## los neurodos por capa que se desee que tenga la red, por ejemplo:
        ##    Para una red con 3 capas ocultas y 5 neurodos por capa se deberá
        ##    ingresar una tupla como la siguiente: (5,5,5)
        self.estructura=estructura[::]
        self.datos_de_entrenamiento=datos_de_entrenamiento
        if isinstance(neurodos_entrada_salida,(list,tuple)):
            entrada,salida=neurodos_entrada_salida
            self.estructura.insert(0,entrada);self.estructura.append(salida)
        
        if isinstance(datos_de_entrenamiento,(list,tuple)):
            nx=len(datos_de_entrenamiento[0]);ny=len(datos_de_entrenamiento[1])
            nueva_estructura=self.estructura[::]
            nueva_estructura.insert(0,nx);nueva_estructura.append(ny)
            self.estructura=nueva_estructura
        
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

    
    def Entrenar(self,datos_ent=None,pesos=None,alpha=0.3,max_iter=500000,seed=None,tipo_entrenamiento='EL',parametrosAG=None,pruebasAG=3):
        ##Parametros AG:(Nind, Ngen)    ;    agpruebas-> Numero de pruebas para asegurar convergencia
        ##El tipo de entrenamiento por defecto es  Fuera de Linea (FL), también
        ##puede seleccionarse el tipo En Linea (EL)
        if not seed: seed=self.seed
        ##Se confirma que los datos ingresados sean correctos, además de obtner
        ##el número de datos de entrada y de salida
        if not isinstance(datos_ent,(np.ndarray,list,tuple)):
            xi,yi=self.datos_de_entrenamiento
            nx,mx=verificarDatosEnt(xi);ny,my=verificarDatosEnt(yi)
        else:
            xi,yi=datos_ent
            nx,mx=verificarDatosEnt(xi);ny,my=verificarDatosEnt(yi)
            print 'Datos leidos correctamente\n'
            nueva_estructura=self.estructura[::]
            nueva_estructura.insert(0,nx);nueva_estructura.append(ny)
            self.estructura=nueva_estructura
            
        
        if mx!=my:
            raise ValueError('Las dimensiones de muestras de x y y no concuerdan')
        ##Se copia la estructura predeterminada de la red y se añade
        ##la cantidad de neurodos en la entrada y a la salida
        
        matrices,npesos=dimensionarMatricesDePesos(self.estructura)
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
        if self.deb:print '\nInicio de entrenamiento ,tipo: {}'.format(tipo_entrenamiento)
            
        if tipo_entrenamiento=='EL':
            '''
            Entrenamiento tipo En Linea, por cada muestra se actualizan los pesos
            '''
            for ent in range(max_iter):
                ##Seleccion de datos de entrenamiento
                num_entrenamiento=rnd.randint(0,mx)
                x=xi[num_entrenamiento];y=yi[num_entrenamiento]
                ##Propagacion hacia adelante
                activaciones=self.FP(pesos=pesos,xi=x,seed=seed)
                y_red=activaciones[-1]
                error=y-y_red#-y*np.log(y_red)+(1-y)*np.log(1-y_red)
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
                    
                if ent%int(max_iter*.1)==0 and self.deb: print '\nIteracion:',ent,' error:',.5*np.sum(error)**2,'\n',np.round(y_red),'->',y
        
        elif tipo_entrenamiento=='FL':
            '''
            Entrenamiento fuera de linea, se prueba con todas las muestras y se actualizan después los pesos
            '''
            for ent in range(max_iter):
                
                deltas_pesos=list()
                for m in range(mx):
#                    print '\nMuestra',m+1,'-'*10
                    ##Seleccion de datos de entrenamiento
                    x=xi[m];y=yi[m][0]
                    ##Propagacion hacia adelante, se almacenan todas las activaciones
                    ##de los neurodos
                    activaciones=self.FP(pesos=pesos,xi=x,seed=seed)
                    y_red=activaciones[-1]
                    error=y-y_red#-y*np.log(y_red)+(1-y)*np.log(1-y_red)
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
                    
                    ##Actualizacion de pesos con el error total
                    for i in range(len(pesos)):
#                        print '\nPesos',i+1
                        act=np.atleast_2d(np.append(activaciones[i],[self.bias]))
                        delta_pesos=np.dot(deltas[i],act)
                        try:
#                            print 'Anteriores',deltas_pesos[i]
#                            print 'A añadir',delta_pesos
                            deltas_pesos[i]+=delta_pesos/mx
                            
                            #print 'Se suma la matriz de delta_pesos'
                        except:
#                            print 'Añadidos',delta_pesos
                            deltas_pesos.append(delta_pesos/mx)
                            #print 'Se agrega la matriz de delta_pesos'
                    
                for i in range(len(deltas_pesos)):
                    pesos[i]+=alpha*delta_pesos
                    
                if ent%int(max_iter*.1)==0 and self.deb: print '\nIteracion:{:^8} -  error:{:^5.4e}'.format(ent,.5*np.sum(error)**2)
        
        elif tipo_entrenamiento=='AG':
            '''
            Entrenamiento por medio de algoritmo genético!!! :D
            '''
            if not parametrosAG: ##Revisando los parámetros
                raise ValueError('No hay parametros para el Algoritmo genético')
            else:
                num_individuos,num_generaciones=parametrosAG
                
            ag=AG(deb=self.deb)##Se inicia el AG y se asignan parámetros
            ag.parametros(optim=0,Nind=num_individuos,Ngen=num_generaciones,pruebas=pruebasAG)
            ##Se define la función objetivo, la que deberá optimizarse
            def fobj(pesos,est):
                error_tot=0
                for m in range(mx):#mx):
                    x=xi[m]
                    W=redimensionarPesos(pesos,est)
                    y_red=self.FP(pesos=W,xi=x)[-1]
                    y=yi[m]
                    error_par=(y-y_red)**2#-y*np.log(a)-(1-y)*np.log(1-a)
                    error_tot+=np.array([np.sum(error_par)])
                
                return(error_tot/(m+1))       

            ag.variables(comun=[npesos,-10,10])
            ag.Fobj(fobj,matrices)
            Went,error=ag.start()
            pesos=redimensionarPesos(Went,matrices)
   
            print 'Error min: {:.4e}'.format(error)


            
        if self.deb:print 'Fin del entrenamiento'
            
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
    est=[2,2]#real es [2,2,2,1]
    nn=RedNeuronal(est,deb=True)
    xi=np.array([[0,0],
                 [1,0],
                 [0,1],
                 [1,1]])
    yi=[[0],[1],[1],[0]]
    yi=np.array(yi)

#    xi=np.array([[.05,.1]])
#    yi=np.array([[.01,.99]])
    nuevos_pesos=nn.Entrenar(datos_ent=(xi,yi),tipo_entrenamiento='AG',parametrosAG=(25,500))
    res=nn.FP(xi=[xi[0]],pesos=nuevos_pesos)
    print '\nPredicciones'
    print xi[0],res[-1]
    res=nn.FP(xi=[xi[1]],pesos=nuevos_pesos)
    print xi[1],res[-1]
    res=nn.FP(xi=[xi[2]],pesos=nuevos_pesos)
    print xi[2],res[-1]
    res=nn.FP(xi=[xi[3]],pesos=nuevos_pesos)
    print xi[3],res[-1]
    
