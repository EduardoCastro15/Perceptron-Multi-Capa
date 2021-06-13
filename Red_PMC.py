import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt

class MLP():
    # constructor
    def __init__(self,xi,d,w_a,w_b,w_c,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida):
        # Variables de inicialización 
        self.xi = np.transpose(xi)
        self.d = d
        self.wa = w_a
        self.wb = w_b
        self.wc = w_c
        self.us = us
        self.uoc = uoc
        self.precision = precision
        self.epocas = epocas
        self.fac_ap = fac_ap
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salida = n_salida
        
        # Variables de aprendizaje
        self.di = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Ew = 0 # Error cuadratico medio
        self.Error_prev = 0 # Error anterior
        self.Errores = []
        self.Error_actual = np.zeros((len(d))) # Errores acumulados en un ciclo de muestras
        self.Entradas = np.zeros((1,n_entradas))
        self.un = np.zeros((n_ocultas,1)) # Potencial de activacion en neuronas ocultas
        self.gu = np.zeros((n_ocultas,1)) # Funcion de activacion de neuronas ocultas
        self.Y = 0.0 # Potencial de activacion en neurona de salida
        self.y = 0.0 # Funcion de activacion en neurona de salida
        self.epochs = 0
        
        # Variables de retropropagacion
        self.error_real = 0
        self.ds = 0.0 # delta de salida
        self.docu = np.zeros((n_ocultas,1)) # Deltas en neuronas ocultas
        
    def Operacion(self):
        respuesta = np.zeros((len(self.d),1))
        for p in range(len(self.d)):
            self.Entradas = self.xi[:,p]
            self.Propagar()
            respuesta[p,:] = self.y
        return respuesta.tolist()
    
    def Aprendizaje(self):
        Errores = [] # Almacenar los errores de la red en un ciclo
        while(np.abs(self.error_red) > self.precision):
            self.Error_prev = self.Ew
            for i in range(len(d)):
                self.Entradas = self.xi[:,i] # Senales de entrada por iteracion
                self.di = self.d[i]
                self.Propagar()
                self.Backpropagation()
                self.Propagar()
                self.Error_actual[i] = (0.5)*((self.di - self.y)**2)
            # error global de la red
            self.Error()
            Errores.append(self.error_red)
            self.epochs +=1
            # Si se alcanza un mayor numero de epocas
            if self.epochs > self.epocas:
                break
        # Regresar 
        return self.epochs,self.wa,self.wb,self.wc,self.us,self.uoc,Errores
                
    
    def Propagar(self):
        # Operaciones en la primer capa
        for a in range(self.n_ocultas):
            self.un[a,:] = np.dot(self.w1[a,:], self.Entradas) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for o in range(self.n_ocultas):
            self.gu[o,:] = tanh(self.un[o,:])
        
        # Calcular Y potencial de activacion de la neuronas de salida
        self.Y = (np.dot(self.w2,self.gu) + self.us)
        # Calcular la salida de la neurona de salida
        self.y = tanh(self.Y)
    
    def Backpropagation(self):
        # Calcular el error
        self.error_real = (self.di - self.y)
        # Calcular ds
        self.ds = (dtanh(self.Y) * self.error_real)
        # Ajustar w2
        self.w2 = self.w2 + (np.transpose(self.gu) * self.fac_ap * self.ds)
        # Ajustar umbral us
        self.us = self.us + (self.fac_ap * self.ds)
        # Calcular docu
        self.docu = dtanh(self.un) * np.transpose(self.w2) * self.ds
        # Ajustar los pesos w1
        for j in range(self.n_ocultas):
            self.w1[j,:] = self.w1[j,:] + ((self.docu[j,:]) * self.Entradas * self.fac_ap)
        
        # Ajustar el umbral en las neuronas ocultas
        for g in range(self.n_ocultas):
            self.uoc[g,:] = self.uoc[g,:] + (self.fac_ap * self.docu[g,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Ew = ((1/len(d)) * (sum(self.Error_actual)))
        self.error_red = (self.Ew - self.Error_prev)

# Funcion para obtener la tanh
def tanh(x):
    return np.tanh(x)

# Funcion para obtener la derivada de tanh x
def dtanh(x):
    return 1.0 - np.tanh(x)**2

# Funcion sigmoide de x
def sigmoide(x):
    return 1/(1+np.exp(-x))

# Funcion para obtener la derivada de de la funcion sigmoide
def dsigmoide(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)

# Propagama principal
if "__main__"==__name__:
    # Carga de los datos
    datos_cielo = pd.read_csv('cieloRGB.csv') # Leer archivo csv
    datos_boscoso = pd.read_csv('boscosoRGB.csv') # Leer archivo csv
    datos_suelo = pd.read_csv('sueloRGB.csv') # Leer archivo csv
        
    # Crear vector de entradas xi
    c1 = np.array(datos_cielo)
    c2 = np.array(datos_boscoso)
    c3 = np.array(datos_suelo)

    d1 = c1[:, 3]
    d2 = c1[:, 3]
    d3 = c1[:, 3]
    
    # Vector de validación
    xj = np.array([[209, 169, 131],
                   [ 89, 133,  60],
                   [152, 140, 111]])
    
    # Parametros de la red
    f, c = xi.shape
    fac_ap = 0.5 #Factor de aprendizaje
    precision = 0.1 #Precision inicial
    epocas = 484 #Numero maximo de epocas (1.2e^6) = 484.1145
    epochs = 0 #Contador de epocas utilizadas
    
    # Arquitectura de la red
    n_entradas = c # Numero de entradas
    cap_ocultas = 2 # Una capa oculta
    n_ocultas = 3 # Neuronas en la capa oculta 1
    n_ocultas2 = 2 # Neuronas en la capa oculta 2
    n_salida = 1 # Neuronas en la capa de salida
    
    # Valor de umbral o bia
    us = 1.0 # umbral en neurona de salida
    uoc = np.ones((n_ocultas,1),float) # umbral en las neuronas ocultas
    
    # Matriz de pesos sinapticos
    #random.seed(0)
    #w_1 = random.rand(n_ocultas,n_entradas)
    #w_2 = random.rand(n_salida,n_ocultas)
    w_a = np.array([[-2.89,  22.02, -24.07],
                    [12.34, -13.82, -16.50],
                    [ 9.31, -37.61,  -5.19]])
    w_b = np.array([[ 5.26,  4.66, -0.63],
                    [-1.08, -4.34, -5.46]])
    w_c = np.array([[-3.86, -8.90],
                    [ 6.45, -7.24],
                    [-9.03,  3.53]])
    
    #Inicializar la red PMC
    red = MLP(xi,d,w_a,w_b,w_c,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida)
    epochs,wa_a,wb_a,wc_a,us_a,uoc_a,E = red.Aprendizaje()
    
    # graficar el error
    plt.grid()
    plt.ylabel("Error de la red",fontsize=12)
    plt.xlabel("Épocas",fontsize=12)
    plt.title("Perceptrón Multicapa",fontsize=14)
    x = np.arange(epochs)
    plt.plot(x,E,'b',label="Error global")
    plt.legend(loc='upper right')
    plt.show
    
    # validacion
    red = MLP(xi,d,wa_a,wb_a,wc_a,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida)
    salidas = red.Operacion()
    print("Salidas: ",salidas)
    print("Epochs: ", epochs)
    