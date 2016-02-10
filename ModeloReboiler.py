# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:08:01 2016

@author: root
"""
from __future__ import division
import numpy
import numericos
import matplotlib
#import matplotlib.pyplot as plt

def Reboiler(Controladores,Lvar,
             mezcla=('C8','C10'),dt=0.01,plot=0,delay=0):
    ##Base de datos de los compuestos a utilizar. datos obtenidos de:
    ## Perry R. (2010) "Manual del Ingeniero Quimico" 7Ed. McGraw-Hill:España
    ## Datos de Alcanos y Agua
    bd_subs={
    'C8':{'Pvap':(96.084,-7900.2,-11.003,7.1802e-6,2), #Perry 2-55
          'Dens':(0.53731,0.26115,568.7,0.28034), #Perry 2-101
          'Lamvap':(5.518e7,0.38467,0,0), #Perry 2-163
          'Cp':(2.2483e5,-1.8663e2,9.5891e-1,0,0), #Perry 2-177
          'Tc':568.7,#K  #Perry 2-55
          'MM':114.231 ##kg/kmol
    },
    'C9':{'Pvap':(109.35,-9030.4,-12.882,7.8544e-6,2),
          'Dens':(0.48387,0.26147,594.6,0.28281),
          'Lamvap':(6.037e7,0.38522,0,0),
          'Cp':(3.8308e5,-1.1398e3,2.7101,0,0),
          'Tc':594.6,#K
          'MM':128.258 ##kg/kmol
    },
    'C10':{'Pvap':(112.73,-9749.6,-13.245,7.1266e-6,2),
          'Dens':(0.42831,0.25795,617.7,0.28912),
          'Lamvap':(6.6126e7,0.39797,0,0),
          'Cp':(2.7862e5,-1.9791e2,1.0737,0,0),
          'Tc':617.7,#K
          'MM':142.285 ##kg/kmol
    },
    'H2O':{'Pvap':(73.649,-7258.2,-7.3037,4.1653e-6,2),
           'Lamvap':(5.2053e7,0.3199,-0.212,0.25795),
           'Cp':(2.7637e5,-2.0901e3,8.125,-1.4116e-2,9.3701e-6),
           'Tc':647.13,#K
          'MM':18.015 ##kg/kmol
    },
    'EtOH':{'Pvap':(74.475,-7164.3,-7.327,3.134e-6,2),
           'Lamvap':(5.69e7,0.3359,0,0),
           'Cp':(1.0264e5,-1.3963e2,-3.0341e-2,2.0386e-3,0),
           'Tc':513.92,#K
          'MM':46.069 ##kg/kmol
    }
    } 
    
    ## La temperatura esta dada en Kelvin
    
    def Densidad(substancia,T):## Función para calcular la densidad
        C1,C2,C3,C4=bd_subs[substancia]['Dens']
        return C1/C2**(1+(1-T/C3)**C4) ## kmol/m3
    
    def Cp(substancia,T):## Función para calcular el Cp
        C1,C2,C3,C4,C5=bd_subs[substancia]['Cp']
        return C1+C2*T+C3*T**2+C4*T**3+C5*T**4 ## kJ/(kmol K)
    
    def Lamvap(substancia,T):## Función para calcular el calor de vaporización
        C1,C2,C3,C4=bd_subs[substancia]['Lamvap']
        Tc=bd_subs[substancia]['Tc']; Tr=T/Tc
        return (C1*(1-Tr)**(C2+C3*Tr+C4*Tr**2))#/1000 ## kJ/kmol
    
    def Entalpia(substancia,T): ## Funcion para calcular la entalpia
        C1,C2,C3,C4,C5=bd_subs[substancia]['Cp']
        return C1*T+C2*T**2/2.+C3*T**3/3.+C4*T**4/4.+C5*T**5/5. ## kJ/kmol
        
    def Pvap(substancia,T): ## Funcion para calcular la presión de vapor
        C1,C2,C3,C4,C5=bd_subs[substancia]['Pvap']
        return numpy.exp(C1+C2/T+C3*numpy.log(T)+C4*T**C5) ## Pa
    
    def F_alpha(mezcla,T): ## Calculo de la volatilidad relativa
        Pvap1=Pvap(mezcla[0],T);Pvap2=Pvap(mezcla[1],T)
        return Pvap1/Pvap2
        
    def Ponderacion(Prop1,Prop2,x1): ## Pondera para obtener propiedades de mezcla
        return (Prop1-Prop2)*x1+Prop2
    
    ## Funcionas propias para la destilación
    def TempEbullicion(subs,x1=1,T0=300.,Pt=101325.): ##La presion debe ser dada en Pa
        def Funcion_objetivo(T):## Funcion objetivo, será pasada como argumento al
                                ## buscador de ceros 
            if isinstance(subs, tuple): ##Si hay 2 compuestos
                Pvap1=Pvap(subs[0],T)
                Pvap2=Pvap(subs[1],T)
                return (Pt-Pvap2)/(Pvap1-Pvap2)-x1 ## Condicion de equilibrio
            else: ## Si solo hay un compuesto
                return Pvap(subs,T)-Pt##Condicion de equilibrio para un compuesto
        ## Uso del metodo numerico de la Falsa Posicion, se usa este ya que falla
        ## menos que el Newton-Raphson al encontrar la raíz y se especifica desde
        ## un inicio el intervalo en el que se está operando-> T0,T0+100K
        if isinstance(subs,tuple):
            ## Se comienza con una aproximacion de la temperatura de ebullicion
            ## del componente más volátil
            T0_metodo=TempEbullicion(subs[0],T0=T0,Pt=Pt)
            return numericos.FalsaPosicion(Funcion_objetivo,T0_metodo,T0_metodo+100)
        else:
            return numericos.FalsaPosicion(Funcion_objetivo,T0,T0+100)
    
    def Equilibrio(x,alpha): ##Equilibrio líquido-vapor de mezcla
        return alpha*x/(1+(alpha-1)*x)    
    
    ###############################################################################
    def Er_rel(a,b): ##Función de error relativo
        return(2*abs(a-b)/(abs(a)+abs(b)))
        
    def RK4(vals,paso):
        M0,x0,h0,Q0,intM0=vals
        #Calculo con Euler
        M_E=M0+paso*dMdt(*vals)
        x_E=x0+paso*dxdt(*vals)
        h_E=h0+paso*dhdt(*vals)
        Q_E=Q0+paso*dQdt(*vals)
        #Cálculo de todas las k1    
        k11=dMdt(*vals);k12=dxdt(*vals);k13=dhdt(*vals);k14=dQdt(*vals)
        #Se calculan los valores de los trapecios
        M_1_2=(Mref-M0)+paso*k11/2.;intM1=(M0+M_1_2)*paso/4.
        #Cálculo de todas las k2
        vals1=(M0+paso*k11/2.,x0+paso*k12/2.,h0+paso*k13/2.,Q0+paso*k14/2.,intM1)
        k21=dMdt(*vals1);k22=dxdt(*vals1);k23=dhdt(*vals1);k24=dQdt(*vals1)
        #Cálculo de todas las k3
        vals2=M0+paso*k21/2.,x0+paso*k22/2.,h0+paso*k23/2.,Q0+paso*k24/2.,intM1
        k31=dMdt(*vals2);k32=dxdt(*vals2);k33=dhdt(*vals2);k34=dQdt(*vals2)
        #Se calculan los valores de los trapecios de la segunda parte
        M1=M_1_2+paso*k11/2.;intM2=(M1+M_1_2)*paso/4.;n_intM=intM1+intM2   
        #Cálculo de todas las k4
        vals3=M0+paso*k31/2.,x0+paso*k32/2,h0+paso*k33/2.,Q0+paso*k34/2.,n_intM
        k41=dMdt(*vals3);k42=dxdt(*vals3);k43=dhdt(*vals3);k44=dQdt(*vals3)
        
        dM=paso*(k11+2*k21+2*k31+k41)/6.
        dx=paso*(k12+2*k22+2*k32+k42)/6.
        dh=paso*(k13+2*k23+2*k33+k43)/6.
        dQ=paso*(k14+2*k24+2*k34+k44)/6.
    
        M_RK4=M0+dM;x_RK4=x0+dx;h_RK4=h0+dh;Q_RK4=Q0+dQ
        
        errores=(Er_rel(h_E,h_RK4),
                 Er_rel(M_E,M_RK4),
                 Er_rel(Q_E,Q_RK4),
                 Er_rel(x_E,x_RK4))
        error_max=max(errores)
        #print 'Errores',errores
        return(error_max,M_RK4,x_RK4,h_RK4,Q_RK4,n_intM)
        
    def Recur(vals,paso,recursion=0,tol=.1):
        if recursion>=50: ## muúmero máximo de recursiones
            return(None)
        if recursion: ## Si ya es una recursion se repite 2 veces
            repeticiones=2
        else: ## Sino sólo se repite 1
            repeticiones=1
            
        for dummy_i in range(repeticiones):
            resRK4=RK4(vals,paso)
            if resRK4[0]<=tol: ## Si el error es menor a la tolerancia
                vals=resRK4[1:] ## Actualización de los valores
            else: ## Sino se hace una recursion partiendo a la mitad el paso
                vals=Recur(vals,paso/2,recursion+1,tol)
                #print resRK4[0]
        
        return(vals)
    
    ###############################################################################
    ## Ecuaciones diferenciales para modelado del sistema
    def dMdt(M,x,h,Q,intM):
        return L-(Evap*Q/LamvapMezcla+B)
    
    def dxdt(M,x,h,Q,intM):
        return (L*(x_L-x)+Evap*Q/LamvapMezcla*(x-y))/M
    
    def dhdt(M,x,h,Q,intM):
        return (L*(h_L-h)+Evap*Q/LamvapMezcla*(h-H)+Q)/M
    
    def dQdt(M,x,h,Q,intM):
        return -LamvapH2O*kc*(Mref-M\
               -tau_D*(L-(Evap*Q/LamvapMezcla+B))\
               +tau_I*intM)
        
    ###############################################################################
    #######################INICIO DE LA SIMULACION#################################
    ###############################################################################
    
    substancia1=mezcla[0]
    substancia2=mezcla[1]
    Pt=101325 # Pa
    ## Condiciones iniciales:
    x_L=0.6 #kmol/min
    T_L=412 # K
    ## Si se considera que la capacidad calorifica permanece constante, entonces
    ## podemos tomar el valor C1 de las constantes para el cálculo del Cp
    Cp1=bd_subs[substancia1]['Cp'][0];Cp2=bd_subs[substancia2]['Cp'][0]
    Cp_L=Ponderacion(Cp1,Cp2,x_L)    
    ## De esta manera entonces podemos calcular la entalpia
    h_L=Cp_L*T_L
    #print h_L
    '''
    Método anterior para calcular entalpías y Cps...
    h_L1=Entalpia(substancia1,T_L);h_L2=Entalpia(substancia2,T_L)
    h_L=Ponderacion(h_L1,h_L2,x_L) #kJ/kmol
    print h_L
    
    Cp1=Cp(substancia1,T_L);Cp2=Cp(substancia2,T_L);CpMez=Ponderacion(Cp1,Cp2,x_L)
    CpMez=h_L/T_L
    '''
    ## Datos iniciales para el vapor de calentamiento
    Q=4.5e8;LamvapH2O=Lamvap('H2O',TempEbullicion('H2O',Pt=2*Pt)) # kJ/kmol
    
    Teb=TempEbullicion(mezcla,x_L,Pt=Pt) ## Temperatura de ebullicion de la mezcla    
    M=30;B=10 #kmol/min
    x=0.445
    T=Teb## Para que la mezcla se encuentre en su punto de ebullicion
    ## Si se considera que la capacidad calorifica permanece constante, entonces
    ## podemos tomar el valor C1 de las constantes para el cálculo del Cp
    CpMez=Ponderacion(Cp1,Cp2,x)
    h=CpMez*T
    y=Equilibrio(x,F_alpha(mezcla,T))
    
    ## Cálculo de la entalpia de vaporizacion 
    LamvapM1=Lamvap(substancia1,T);LamvapM2=Lamvap(substancia2,T)
    LamvapMezcla=Ponderacion(LamvapM1,LamvapM2,x) # kJ/kmol
    #print '{:.4e} - {:.4e} - T:{}'.format(h_L1,h_L2,h_L/CpMez)
    #print 'Teb Volatil:',TempEbullicion(substancia1,Pt=Pt),' Teb Mezcla',Teb
    #print LamvapMezcla,LamvapH2O,h
    ## Cálculo de la entalpia del vapor. Se le suma la entalpia de vaporizacion
    ## a la entalpia del líquido
    H=h+LamvapMezcla  # kJ/kmol
    
    Evap=1 # Bandera que indica si hay evaporación
    
    intM=0 # Valor de la integral de (Mref-M)
    
    ## Constantes de control
    kc,tau_D,tau_I=1,.5,.5 ; parametros_iniciales=True
    Mref=30
    
    
    ## Desarrollo de la simulación
    t=0
    t_l,M_l,B_l,V_l,x_l,T_l,Teb_l,Q_l,LVM_l,y_l=[t],[M],[B],[Q/LamvapMezcla],[x],[T],[Teb],[Q],[LamvapMezcla],[y]
    
    #T_p=np.linspace(500,600,100)
    #plt.plot(T_p,Lamvap(substancia1,T_p))
    #plt.show()
    
    
    #print '{:^5}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}'.format('t','M','h','Q','V','T','LVM')
    
    for i in range(len(Lvar)):
        if i>len(Lvar)*delay and parametros_iniciales:
            kc,tau_D,tau_I=Controladores; parametros_iniciales=False
            
        L=Lvar[i]
        if T>Teb:
            Evap=1
        else:
            Evap=0
            
    #    Cp1=Cp(substancia1,T);Cp2=Cp(substancia2,T);CpMez=Ponderacion(Cp1,Cp2,x)
    #    print 'CpMez 1:{} ; 2:{}'.format(Cp1,Cp2)    
    #    print 'LamvapMez 1:{} ; 2:{}'.format(LamvapM1,LamvapM2)
    #    print 'Temperatura:',T
            
        #if i%int(len(Lvar)*.1)==0:print '{:^5}|{:^8.2f}|{:^6.2e}|{:^6.2e}|{:^6.2e}|{:^8.2f}|{:^6.2e}'.format(t,M,h,Q,Evap*Q/LamvapMezcla,T,LamvapMezcla)
        
        M,x,h,Q,intM=Recur((M,x,h,Q,intM),paso=dt,tol=0.001)
        
        if M>=50:
            M=50
        elif M<=0:
            M=0.0001
            
        if Q<=0:#Seguro por si Q es negativo
            Q=0
        elif Q/LamvapMezcla>=20:#Y por si el flujo de vapor pasa del máximo
            Q=20*LamvapMezcla #No podrá ingresar mayor cantidad de calor al sistema
            
        t+=dt
        t_l.append(t);M_l.append(M);B_l.append(B);x_l.append(x)
        V_l.append(Evap*Q/LamvapMezcla);Teb_l.append(Teb);Q_l.append(Q)
        LVM_l.append(LamvapMezcla);y_l.append(y)
        if T==0:
            T_l.append(T_l[-1])
        else:
            T_l.append(T)
        
        ## Actualizacion de los valores iniciales
        CpMez=Ponderacion(Cp1,Cp2,x)
        T=h/CpMez
        #if i%int(tf/dt*.1)==0:print (L*(h_L-h)+Evap*Q/LamvapMezcla*(h-H)+Q)/M
        Teb=TempEbullicion(mezcla,x_L,Pt=Pt) 
        y=Equilibrio(x,F_alpha(mezcla,T))
    #    LamvapM1=Lamvap(substancia1,T);LamvapM2=Lamvap(substancia2,T)
    #    LamvapMezcla=Ponderacion(LamvapM1,LamvapM2,x)
        H=h+LamvapMezcla
    
    
    if plot:
        plt=matplotlib.pyplot
        plt.figure(figsize=(100,100))
        plt.subplot(2,2,1);plt.grid();plt.title('Flujos Molares')
        plt.plot(t_l,M_l,t_l,B_l,t_l,V_l,t_l[:-1],Lvar)#,t_l,numpy.array(V_l)+B_l)
        plt.subplot(2,2,2);plt.grid();plt.title('Fraccion molar')
        plt.plot(t_l,x_l,t_l,y_l)
        plt.subplot(2,2,3);plt.grid();plt.title('Temperaturas')
        plt.plot(t_l,T_l,t_l,Teb_l)
        plt.subplot(2,2,4);plt.grid();plt.title('Calores')
        plt.plot(t_l,Q_l,t_l,LVM_l)
        plt.show()
    
    return numpy.trapz(numpy.abs(numpy.array(M_l)-Mref))+\
           numpy.log10(numpy.sum(numpy.abs(numpy.diff(M_l))+\
           numpy.abs(numpy.diff(V_l))+numpy.abs(numpy.diff(Q_l))))+\
           sum((kc,tau_D,tau_I))

if __name__=='__main__':
    dt=0.01
    tf=10
    Lvar=numpy.random.random(int(tf/dt))*2-1+15
    controladores=(2,.5,.5)
    reb=Reboiler(controladores,Lvar,dt=dt,plot=1)
    print reb
    