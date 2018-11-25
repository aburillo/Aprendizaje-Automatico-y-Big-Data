#!/usr/bin/env python
# coding: utf-8

# # Práctica 1

#     El objetivo de esta práctica es comprobar como con el descenso de gradiente se realiza unas predicciones mas ajustadas a los datos de   entrada, tomando como ejemplos de entrenamiento los datos existentes en los archivos ex1data1.csv y ex1data2.csv.

#  
#  La hipotesis del modelo lineal: $  h_{\theta} = \theta_{0} + \theta_{1}x $  y la función de coste : 
#  $J(\theta) =  \frac{1}{2m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)})^{2} $ las hemos implementado de forma vectorizada, para realizar los calculos de la manera mas eficiente posible
#   $$ \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)})x_{j}^{i} $$ 
# 

# In[25]:


from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

def readCsv(filename):
    valores= read_csv(filename,header=None).values
    return valores.astype(float)

def h0(tt0, tt1, z):
    return np.add(tt0,np.multiply(tt1,z))
def J(tt0,tt1,x,y):
    return np.sum(np.power(np.subtract(h0(tt0,tt1,x),y),2))/(2*len(x))


#     La función funGDA calcula las thetas minímas que definen la recta que más se ajustan a los datos de entrenamiento. Cuenta con un factor de aprendizaje (factorAp) que agudiza las thetas según su tamaño, el del factor. 
#      hkj $$ \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)})(x_{j}^{(i)}) $$ 

# In[26]:


def funGDA(x, y, fun, factorAp, iterac, tm):
    tta0 = 0
    tta1 = 0
    for i in range(0, iterac):        
        temp0 = tta0 - factorAp*derivada0(x, y, tta0, tta1, tm, fun)
        temp1 = tta1 - factorAp*derivada1(x, y, tta0, tta1, tm, fun) 
        tta0 = temp0
        tta1 = temp1
    return (tta0, tta1)


#     Para el cálculo de la derivada de la función de coste, dividimos el mismo en dos funciones: derivada0, que se encarga de calcular la derivada para theta0; y derivada1, que calcula la derivada de theta1.
#     
#     $$ \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)})(x_{j}^{(i)}) $$ 

# Para el cálculo de la derivada de la función de coste, dividimos el mismo en dos funciones: derivada0, que se encarga de calcular la derivada para theta0; y derivada1, que calcula la derivada de theta1. 
# 
# Derivada0: $$ \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)}) $$
# Derivada1: $$ \theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{0}(x^{(i)})-y^{(i)})(x_{j}^{(i)}) $$

# In[27]:


def derivada0(x, y, t0, t1, tm, fun):
    suma = 0
    for i in range(0, tm):
        suma += (fun(t0, t1, x[i]) - y[i])
    suma = suma/tm
    return suma

def derivada1(x, y, t0, t1, tm, fun):
    suma = 0
    for i in range(0, tm):
        suma += ((fun(t0, t1, x[i]) - y[i])*x[i])
    suma = suma/tm
    return suma


#     Para representar los datos en gráficas, hemos realizado tres funciones distintas, y en cada una de las funciones, 
#     reflejamos los datos de una manera distinta. Lo hemos representado en un dibujo 2D ( dibujaFun ), en otro 3D (dibuja3D),
#     y el último es una representación donde se ha utilizado una escala logarítmica para el eje z (dibujoLogaritmico).

# In[28]:


def dibujaFun(tt0, tt1,x, y):
    plt.plot(x, y, 'r+')
    y = h0(tt0,tt1,x)
    plt.plot(x, y, linewidth=0.1) 


# In[29]:


def dibuja3D(x,y,m):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

# Make data.
    tt0 = np.arange(-10, 10, 0.25)
    tt1 = np.arange(-1, 4, 0.25)
    tt0, tt1 = np.meshgrid(tt0, tt1)
    # Plot the surface.
    filas,columnas=tt0.shape
    matriz= np.empty_like(tt0)
    for i in range(0,filas):
        for j in range(0,columnas):
            matriz[i,j]=J(tt0[i,j],tt1[i,j],x,y)
    
    surf = ax.plot_surface(tt0, tt1, matriz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    #surf = ax.plot_surface(tt0, tt1, J(x,y,tt0,tt1), cmap=cm.coolwarm,
     #                  linewidth=0, antialiased=False)

# Customize the z axis.
    ax.set_zlim(0,800)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return matriz


# In[30]:


def dibujoLogaritmico(matriz,x,y):
    fig = plt.figure()

# Make data.
    tt0 = np.arange(-10, 10, 0.25)
    tt1 = np.arange(-1, 4, 0.25)
    tt0, tt1 = np.meshgrid(tt0, tt1)
    # Plot the surface.
    
    surf = plt.contour(tt0, tt1, matriz, np.logspace(-2,3,20))
    minM= np.unravel_index(np.argmin(matriz),matriz.shape)
    plt.plot(tt0[minM[0]][minM[1]],tt1[minM[0]][minM[1]],'x')
    
# Customize the z axis.
    #ax.set_zlim(0,800)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()  


# In[31]:


def dibujaGrafica():
    matriz = readCsv("ex1data1.csv")
    x = np.array(matriz[:,0])
    y = np.array(matriz[:,1])
    tm = x.size
    plt.figure()
    #ax = plt.gca()
    #ax.axis([int(np.amin(x)), int(np.amax(x)), int(np.amin(y)), int(np.amax(y))])
    tta0 = 0
    tta1 = 0
    tta0, tta1 = funGDA(x, y, h0, 0.01, 1500, tm)
    dibujaFun(tta0, tta1,x, y)
    matriz = dibuja3D(x,y,tm)
    dibujoLogaritmico(matriz,x,y)


# In[32]:


dibujaGrafica()


# Parte 2: regresion con múltiples variables

#     Para esta segunda parte, hemos cambiado la forma de calcular la hipótesis del modelo (h0) y la función de coste (J). En  esta segunda parte mantenemos el calculo de ambas funciones vectorizado.

# Para esta segunda parte, hemos cambiado la forma de calcular la hipótesis del modelo(h0) y la función de coste (J). En  esta segunda parte mantenemos el calculo de ambas funciones vectorizado.
#  $$ h_{\theta} = \theta^{T}x $$
#  $$J(\theta) =  \frac{1}{2m} (X\theta - \vec{y})^{T} (X\theta - \vec{y}) $$ 

# In[33]:


def J2(matrizN, tt):
    x = np.array([np.ones(matrizN.shape[0]), matrizN[:,0], matrizN[:,1]])
    x = np.transpose(x)
    y = np.array(matrizN[:,2])
    aux = np.subtract(np.matmul(x, np.transpose(tt)), y)
    auxi = np.matmul(np.transpose(aux), aux)
    
    return np.divide(auxi, (2*len(y)))

def h02(x, tts):
    return np.dot(tts, x)


#    Al igual que antes, el cálculo de descenso de gradiente ha cambiado su algoritmo. En este caso, en vez de dividir el    cálculo en el número de thetas que se necesite, vectorizamos el cálculo y lo generalizamos, haciendo más óptimo y limpio el cálculo. Recordar que para hallar la derivada de la función de coste, debemos haber normalizado los datos de entrenamiento de "matrizN".

# In[34]:


def funGDA2(matrizN, factorAp, iterac, tm):
    tts = np.zeros(matrizN.shape[1]);
    for i in range(0, iterac):
        tts = tts - factorAp*derivada(matrizN, tts, tm)
    return tts

def derivada(matrizN, tts, tm):
    x = np.array([np.ones(matrizN.shape[0]), matrizN[:,0], matrizN[:,1]])
    y = np.array(matrizN[:,2])
    deriv = np.matmul(np.subtract(h02(x, tts), y), np.transpose(x))
    return np.divide(deriv, tm)


# Como el rango de los distintos atributos es muy diferente (unidades en el caso del número
# de habitaciones y miles en el caso de la superficie) para acelerar la convergencia al aplicar el
# método de descenso de gradiente, es necesario normalizar los atributos, sustituyendo cada
# valor por el cociente entre su diferencia con la media y la desviación estándar de ese atributo en
# los ejemplos de entrenamiento.
# 
# Y al igual que normalizamos los datos antes de aplicar el descenso de gradiente, al obtener la salida de éste, 
# tenemos que desnormalizar los datos, aplicando el paso inverso.
# 

# In[35]:


def normaliza(matriz, medias, desviaciones):
    return np.divide(np.subtract(matriz, medias), desviaciones)

def desnormaliza(matriz, medias, desviaciones):
    return np.add(np.multiply(matriz, desviaciones), medias)


#     Para comprobar que el descenso de gradiente realiza bien sus predicciones llamamos a la ecuacion normal(funRapid), que obtiene el valor óptimo del vector theta.

#  Para comprobar que el descenso de gradiente realiza bien sus predicciones llamamos a la ecuacion normal(funRapid), que obtiene el valor óptimo del vector theta.
#  $$ {\theta} = (X^{T} X)^{-1} X^{T} \vec{y} $$

# In[36]:


def funRapid(matriz):
    x = np.array([np.ones(matriz.shape[0]), matriz[:,0], matriz[:,1]])
    x = np.transpose(x)
    y = np.array(matriz[:,2])
    inv = np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.transpose(x))    
    return np.matmul(inv, y)


# In[37]:


def dibujaGrafica2():
    matriz = readCsv("ex1data2.csv")
    superficie = np.array(matriz[:,0])
    uds = np.array(matriz[:,1])
    precio = np.array(matriz[:,2])
    ttsRapid = funRapid(matriz)
    medias = np.array([np.mean(superficie), np.mean(uds), np.mean(precio)])
    desviaciones = np.array([np.std(superficie), np.std(uds), np.std(precio)])
    matrizN = normaliza(matriz, medias, desviaciones)
    tts = PruebasTasaAprendizaje(matrizN)
    print(tts)
    #matrizDN = desnormaliza(matrizN, medias, desviaciones)
    #print(normaliza([2,4,6],[np.mean[2],np.mean[4],np.mean[6]],[np.std[2],np.std[4],np.std[6]])
    #coste1 = J2(matrizDN, tts)
    coste2 = J2(matriz, ttsRapid)
    #dato de entrada ej
    ej = np.array([1, 1650, 3])
    #normalizamos [1650,3]  
    ejN = normaliza(np.array([1650, 3]),medias[0:len(medias)-1],desviaciones[0:len(desviaciones)-1])
    #le anadimos un 1 delante del array normalizado para poder operar con las tts
    ejN1=np.hstack([np.ones([1]),ejN])
    #obtenermos el valor resultado todavia normalizado
    ejH1=h02(ejN1, tts)
    print("La prediccion mediante Descenso de Gradiente es: ")
    print(desnormaliza(ejH1,medias[-1],desviaciones[-1]))# desnormlaizamos el valor resultado con la media y desviacion de y
    print("La prediccion mediante la ecuacion normal es: ")
    print(h02(ej, ttsRapid))


# In[38]:


dibujaGrafica2()


# In[39]:


def PruebasTasaAprendizaje(matrizN):
    print("Para una tasa de aprendizaje de 0.3 las tts obtenidas son :")
    print(funGDA2(matrizN, 0.3, 15000, matrizN.shape[0]))
    print("Para una tasa de aprendizaje de 0.03 las tts obtenidas son :")
    print(funGDA2(matrizN, 0.03, 15000, matrizN.shape[0]))
    print("Para una tasa de aprendizaje de 0.01 las tts obtenidas son :")
    print(funGDA2(matrizN, 0.01, 15000, matrizN.shape[0]))
    print("Para una tasa de aprendizaje de 0.3 las tts obtenidas son :")
    return funGDA2(matrizN, 0.1, 15000, matrizN.shape[0])
    


# In[ ]:




