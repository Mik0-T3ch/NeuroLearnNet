import numpy as np

class Loss:
    def __init__(self, name="base"):
        self.name = name

    def forward(self, y_true, y_pred):
        raise NotImplementedError("forward no implementado")

    def derivative(self, y_true, y_pred):
        raise NotImplementedError("derivative no implementado")

class MSE(Loss):
    def __init__(self):
        super().__init__("mse")

    def forward(self, y_true, y_pred):
        diff = y_true - y_pred
        cuadrado = diff ** 2
        loss = np.mean(cuadrado)
        return loss

    def derivative(self, y_true, y_pred):
        n = y_true.shape[0]
        diff = y_pred - y_true
        deriv = (2 / n) * diff
        return deriv

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__("bce")

    def forward(self, y_true, y_pred):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        parte1 = y_true * np.log(y_pred_clipped)
        parte2 = (1 - y_true) * np.log(1 - y_pred_clipped)

        loss = -np.mean(parte1 + parte2)
        return loss

    def derivative(self, y_true, y_pred):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        arriba = y_pred_clipped - y_true
        abajo = y_pred_clipped * (1 - y_pred_clipped)

        deriv = arriba / abajo
        return deriv

LOSSES = {
    "mse": MSE(),
    "bce": BinaryCrossEntropy(),
    "binary_crossentropy": BinaryCrossEntropy(),
}

def get_loss(nombre):
    nombre = str(nombre).lower().strip()

    if nombre in LOSSES:
        return LOSSES[nombre]
    else:
        print(f"[warning] loss '{nombre}' no existe, usando mse por defecto xd")
        return LOSSES["mse"]
