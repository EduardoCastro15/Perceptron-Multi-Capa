import numpy as np
import pandas as pd
from numpy import random
import matplotlib.pyplot as plt
import csv

class MLP():
    # constructor
    def __init__(self,xi,d,w_a,w_b,w_c,us,uoc,precision,epocas,fac_ap,n_ocultas,n_ocultas2,n_entradas,n_salida):
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
        self.n_ocultas2 = n_ocultas2
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
        self.un2 = np.zeros((n_ocultas2,1)) # Potencial de activacion en neuronas ocultas2
        self.gu = np.zeros((n_ocultas,1)) # Funcion de activacion de neuronas ocultas
        self.gu2 = np.zeros((n_ocultas2,1)) # Funcion de activacion de neuronas ocultas
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
            self.un[a,:] = np.dot(self.wa[a,:], self.Entradas) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta 1
        for o in range(self.n_ocultas):
            self.gu[o,:] = tanh(self.un[o,:])

        # Operaciones en la capa oculta 2
        for a in range(self.n_ocultas2):
            self.un2[a,:] = np.dot(self.wb[a,:], self.Entradas) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta 2
        for o in range(self.n_ocultas2):
            self.gu2[o,:] = tanh(self.un2[o,:])            

        # Calcular Y potencial de activacion de la neuronas de salida
        self.Y = (np.dot(self.wc,self.gu2) + self.us)
        # Calcular la salida de la neurona de salida
        self.y = tanh(self.Y)
    
    def Backpropagation(self):
        # Calcular el error
        self.error_real = (self.di - self.y)
        # Calcular ds
        self.ds = (dtanh(self.Y) * self.error_real)
        # Ajustar wc
        self.wc = self.wc + (np.transpose(self.gu2) * self.fac_ap * self.ds)
        # Ajustar umbral us
        self.us = self.us + (self.fac_ap * self.ds)
        # Calcular docu
        print(self.un2)
        print('wc')
        print(self.wc)
        print('wc transposed')
        print(np.transpose(self.wc))
        print('ds')
        print(self.ds)
        self.docu = dtanh(self.un2) * np.transpose(self.wc) * self.ds
        # Ajustar los pesos w1
        for j in range(self.n_ocultas2):
            self.wb[j,:] = self.wb[j,:] + ((self.docu[j,:]) * self.Entradas * self.fac_ap)
        
        # Ajustar el umbral en las neuronas ocultas
        for g in range(self.n_ocultas2):
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

def obtener_vector_validacion_bosque_y_rango01():
    with open('boscosoRGB_completo.csv',newline='') as fp:
        data = list(csv.reader(fp))
        
        r,g,b = 0,0,0
        for row in data[1:]:    # Skip the header row and convert first values to integers
            row[0] = int(row[0])
            r += row[0]
            row[0] = float(row[0])
            row[0] = row[0]/255
            row[1] = int(row[1])
            g += row[1]
            row[1] = float(row[1])
            row[1] = row[1]/255
            row[2] = int(row[2])
            b += row[2]
            row[2] = float(row[2])
            row[2] = row[2]/255
            row[3] = int(row[3])
        
        r = r/len(data[1:])
        g = g/len(data[1:])
        b = b/len(data[1:])
        
        print(f'Vector de validacion Bosque R: {r} G: {g} B: {b}')
        r = r/255
        g = g/255
        b = b/255
        validation_vector = [r,g,b,2]

        for row in data[1:]:
            row.extend(validation_vector)

        with open("boscoso_rango01.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)

def obtener_vector_validacion_cielo_y_rango01():
    with open('cieloRGB_completo.csv',newline='') as fp:
        data = list(csv.reader(fp))
        
        r,g,b = 0,0,0
        for row in data[1:]:    # Skip the header row and convert first values to integers
            row[0] = int(row[0])
            r += row[0]
            row[0] = float(row[0])
            row[0] = row[0]/255
            row[1] = int(row[1])
            g += row[1]
            row[1] = float(row[1])
            row[1] = row[1]/255
            row[2] = int(row[2])
            b += row[2]
            row[2] = float(row[2])
            row[2] = row[2]/255
            row[3] = int(row[3])
        
        r = r/len(data[1:])
        g = g/len(data[1:])
        b = b/len(data[1:])
        
        print(f'Vector de validacion Cielo R: {r} G: {g} B: {b}')
        r = r/255
        g = g/255
        b = b/255
        validation_vector = [r,g,b,1]

        for row in data[1:]:
            row.extend(validation_vector)

        with open("cielo_rango01.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)      



def obtener_vector_validacion_suelo_y_rango01():
    with open('sueloRGB_completo.csv',newline='') as fp:
        data = list(csv.reader(fp))
        
        r,g,b = 0,0,0
        for row in data[1:]:    # Skip the header row and convert first values to integers
            row[0] = int(row[0])
            r += row[0]
            row[0] = float(row[0])
            row[0] = row[0]/255
            row[1] = int(row[1])
            g += row[1]
            row[1] = float(row[1])
            row[1] = row[1]/255
            row[2] = int(row[2])
            b += row[2]
            row[2] = float(row[2])
            row[2] = row[2]/255
            row[3] = int(row[3])
        
        r = r/len(data[1:])
        g = g/len(data[1:])
        b = b/len(data[1:])
        
        print(f'Vector de validacion suelo R: {r} G: {g} B: {b}')
        r = r/255
        g = g/255
        b = b/255
        validation_vector = [r,g,b,3]

        for row in data[1:]:
            row.extend(validation_vector)

        with open("suelo_rango01.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)      

def Datos_entrenamiento(matriz,x1,xn):
    xin = matriz[:,x1:xn+1]
    return xin
def Datos_validacion(matriz,xji,xjn):
    xjn = matriz[:,xji:xjn+1]
    return xjn
    

# Programa principal
if "__main__"==__name__:
    # Carga de los datos
    obtener_vector_validacion_bosque_y_rango01()
    obtener_vector_validacion_cielo_y_rango01()
    obtener_vector_validacion_suelo_y_rango01()
    
    #Carga de los 3 archivos en uno
    data = data2 = data3 = ""
  
    # Reading data from boscoso
    with open('boscoso_rango01.csv') as fp:
        data = fp.read()
    
    # Reading data from cielo 
    with open('cielo_rango01.csv') as fp:
        data2 = fp.read()
        data2 = data2[1:] #Eliminando la primer línea
    # Reading data from suelo
    with open('suelo_rango01.csv') as fp:
        data3 = fp.read()
        data3 = data3[1:] #Eliminando la primer línea
    # Merging 2 files
    # To add the data of file2
    # from next line 
    data += data2
    data += data3
    
    with open ('rgb_bcs_juntos.csv', 'w') as fp:
        fp.write(data)
    
    datos = pd.read_csv('rgb_bcs_juntos.csv',low_memory=False)
    matrix_data = np.array(datos)

    #Datos de entrada
    x_inicio = 0
    x_n = 2
    #Datos de entrada validación
    xj_inicio = 4
    xj_n = 6

    # Crear vector de entradas xi
    xi = (Datos_entrenamiento(matrix_data,x_inicio,x_n))
    d = matrix_data[:,x_n+1]
    # Vector de validación
    xj = (Datos_validacion(matrix_data,xj_inicio,xj_n))
    
    # Parametros de la red
    f, c = xi.shape
    fac_ap = 0.5 #Factor de aprendizaje
    precision = 0.1 #Precision inicial
    epocas = 484 #Numero maximo de epocas (1.2e^6) = 484.1145
    epochs = 0 #Contador de epocas utilizadas
    
    # # Arquitectura de la red
    n_entradas = c # Numero de entradas
    cap_ocultas = 2 # Dos capa oculta
    n_ocultas = 3 # Neuronas en la capa oculta 1
    n_ocultas2 = 2 # Neuronas en la capa oculta 2
    n_salida = 3 # Neuronas en la capa de salida
    
    # # Valor de umbral o bia
    us = 1.0 # umbral en neurona de salida
    uoc = np.ones((n_ocultas,1),float) # umbral en las neuronas ocultas
    
    # # Matriz de pesos sinapticos
    random.seed(0)
    w_a = random.rand(n_ocultas,n_entradas)
    w_b = random.rand(n_ocultas2,n_ocultas)
    w_c = random.rand(n_salida,n_ocultas2)
    
    #Inicializar la red PMC
    red = MLP(xi,d,w_a,w_b,w_c,us,uoc,precision,epocas,fac_ap,n_ocultas,n_ocultas2,n_entradas,n_salida)
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
    red = MLP(xi,d,wa_a,wb_a,wc_a,us,uoc,precision,epocas,fac_ap,n_ocultas,n_ocultas2,n_entradas,n_salida)
    salidas = red.Operacion()
    print("Salidas: ",salidas)
    print("Epochs: ", epochs)
    