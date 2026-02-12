import numpy as np

# Funcion sigmoid
def sigmoid(x):
    e = np.exp(-x)
    denom = 1 + e
    resultado = 1 / denom
    
    # Funcion por si alguien manda un escalar
    if isinstance(resultado, (int, float)):
        return float(resultado)
    
    return resultado

def sigmoid_derivative(x):
    s = sigmoid(x)
    uno_menos_s = 1 - s
    deriv = s * uno_menos_s
    
    return deriv
# Funcion Relu
def relu(x):
    cero = 0
    salida = np.maximum(cero, x)
    
    return salida


def relu_derivative(x):
    mask = x > 0
    deriv = np.where(mask, 1, 0)
    
    return deriv

# Funcion Tanh
def tanh(x):
    valor = np.tanh(x)
    return valor


def tanh_derivative(x):
    t = np.tanh(x)
    cuadrado = t ** 2
    deriv = 1 - cuadrado
    
    return deriv
# Funcion Lineal
def linear(x):
    salida = x
    return salida


def linear_derivative(x):
    shape = np.shape(x)
    unos = np.ones_like(x)
    
    return unos

# Funcion Gelu
def gelu(x):
    c = np.sqrt(2 / np.pi)
    
    x_cubo = np.power(x, 3)
    inside = x + 0.044715 * x_cubo
    
    tanh_part = np.tanh(c * inside)
    
    resultado = 0.5 * x * (1 + tanh_part)
    
    return resultado


def gelu_derivative(x):  
    c = np.sqrt(2 / np.pi)
    
    x2 = x ** 2
    x3 = x ** 3
    
    inside = x + 0.044715 * x3
    tanh_part = np.tanh(c * inside)

    sech2 = 1 - tanh_part ** 2
    
    parte1 = 0.5 * (1 + tanh_part)
    
    deriv_inside = c * (1 + 3 * 0.044715 * x2)
    
    parte2 = 0.5 * x * sech2 * deriv_inside
    
    deriv = parte1 + parte2
    
    return deriv
