import numpy as np

class Perceptron:

    def __init__(self, lr=0.01, epochs=100, bias=True, seed=None, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.bias = bias
        self.seed = seed
        self.verbose = verbose

        self.w = None
        self.errors_ = []

    def _asegurar_arrays(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if y.ndim != 1:
            y = y.reshape(-1)

        return X, y

    def _poner_bias(self, X):
        if not self.bias:
            return X

        unos = np.ones((X.shape[0], 1))
        Xb = np.hstack([unos, X])
        return Xb

    def _step(self, z):
        if isinstance(z, np.ndarray):
            return (z >= 0).astype(int)
        else:
            return 1 if z >= 0 else 0
        
    def fit(self, X, y):
        X, y = self._asegurar_arrays(X, y)
        Xb = self._poner_bias(X)

        if set(np.unique(y)) == {-1, 1}:
            y = (y == 1).astype(int)

        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, Xb.shape[1])

        self.errors_ = []

        for epoca in range(self.epochs):
            errores = 0

            for xi, yi in zip(Xb, y):
                z = np.dot(xi, self.w)
                y_hat = self._step(z)

                error = yi - y_hat

                if error != 0:
                    ajuste = self.lr * error * xi
                    self.w = self.w + ajuste
                    errores += 1

            self.errors_.append(errores)

            if self.verbose:
                print(f"epoca {epoca+1}/{self.epochs} | errores: {errores}")

            if errores == 0:
                break

        return self

    def net_input(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xb = self._poner_bias(X)
        z = Xb @ self.w
        return z

    def predict(self, X):
        z = self.net_input(X)
        return self._step(z)

    def score(self, X, y):
        X, y = self._asegurar_arrays(X, y)
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return acc
