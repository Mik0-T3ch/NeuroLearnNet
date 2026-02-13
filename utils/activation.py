import numpy as np


class Activation:
    def __init__(self, name="base"):
        self.name = name

    def forward(self, x):
        #aqui va el calculo principal de la activacion
        raise NotImplementedError("forward no implementado")

    def derivative(self, x):
        #esto sirve para el aprendizaje del modelo
        raise NotImplementedError("derivative no implementado")


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x):
        #convierte valores en un rango entre 0 y 1
        exp_part = np.exp(-x)
        denom = 1 + exp_part
        result = 1 / denom
        return result

    def derivative(self, x):
        s = self.forward(x)
        uno_menos = 1 - s
        deriv = s * uno_menos
        return deriv


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        #todo lo negativo se vuelve 0
        cero = 0
        salida = np.maximum(cero, x)
        return salida

    def derivative(self, x):
        #solo deja pasar lo positivo
        mask = x > 0
        deriv = mask.astype(float)
        return deriv


class Tanh(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        #similar a sigmoid pero centrada en 0
        valor = np.tanh(x)
        return valor

    def derivative(self, x):
        #cambio de la salida
        t = np.tanh(x)
        cuadrado = t * t
        deriv = 1 - cuadrado
        return deriv


class Linear(Activation):
    def __init__(self):
        super().__init__("linear")

    def forward(self, x):
        salida = x
        return salida

    def derivative(self, x):
        #siempre es 1
        unos = np.ones_like(x)
        return unos


class GELU(Activation):
    def __init__(self):
        super().__init__("gelu")

    def forward(self, x):
        #version mas suave que relu
        c = np.sqrt(2 / np.pi)
        x3 = x ** 3
        inside = x + 0.044715 * x3
        tanh_part = np.tanh(c * inside)
        resultado = 0.5 * x * (1 + tanh_part)
        return resultado

    def derivative(self, x):
        #cambio de salida
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


#mapa rapido para pedir activaciones por nombre
ACTIVATIONS = {
    "sigmoid": Sigmoid(),
    "relu": ReLU(),
    "tanh": Tanh(),
    "linear": Linear(),
    "gelu": GELU(),
    "lin": Linear(),  
}


def get_activation(nombre):
    #devuelve la activacion segun el nombre
    nombre = str(nombre).lower().strip()

    if nombre in ACTIVATIONS:
        return ACTIVATIONS[nombre]
    else:
        print(f"[warning] activacion '{nombre}' no existe, usando linear xd")
        return ACTIVATIONS["linear"]
