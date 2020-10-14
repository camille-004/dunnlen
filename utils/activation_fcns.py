import numpy as np

'''
Identity Function
'''


def identity(x):
    return x


def identity_deriv(x):
    return 1


'''
Heaviside Step Function
'''


def heaviside_step(x):
    return 0 if x < 0 else 1


def heaviside_step_deriv(x):
    return 0


'''
Sigmoid Function
'''


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


'''
Tanh (Hyperbolic Tangent) Function
'''


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


'''
Inverse Tangent Function
'''


def inv_tan(x):
    return np.arctan(x)


def inv_tan_deriv(x):
    return 1 / (1 + x ** 2)


'''
ElliotSig/SoftSign Function
'''


def elliot_sig(x):
    return x / (1 + np.abs(x))


def elliot_sig_deriv(x):
    return 1 / ((1 + np.abs(x)) ** 2)


'''
Inverse Square Root Unit (ISRU) Function
'''


def isru(x, alpha=0.01):
    return x / (np.sqrt(1 + alpha * x ** 2))


def isru_deriv(x, alpha=0.01):
    return (1 / (np.sqrt(1 + alpha * x ** 2))) ** 3


'''
Inverse Square Root Linear Unit (ISRLU) Function
'''


def isrlu(x, alpha=0.01):
    if x < 0:
        return x / (np.sqrt(1 + alpha * x ** 2))
    else:
        return x


def isrlu_deriv(x, alpha=0.01):
    if x < 0:
        return (1 / (np.sqrt(1 + alpha * x ** 2))) ** 3
    else:
        return 1


'''
Square Nonlinearity (SQNL) Function
'''


def sqnl(x):
    if x < 2.0:
        return 1
    elif 2.0 >= x >= 0:
        return x - ((x ** 2) / 4)
    elif 0 > x >= -2.0:
        return x + ((x ** 2) / 4)
    else:
        return -1


def sqnl_deriv(x):
    return 1 - (x / 2), 1 + (x / 2)


'''
Rectified Linear Unit (ReLu) Function
'''


def relu(x):
    return 0 if x < 0 else x


def relu_deriv(x):
    return 0 if x < 0 else 1


'''
Leaky Rectified Linear Unit (Leaky ReLu) Function
'''


def leaky_relu(x):
    return 0.01 * x if x < 0 else x


def leaky_relu_deriv(x):
    return 0.01 if x < 0 else 1


'''
Parametric Rectified Linear Unit (PReLu) Function
'''


def prelu(x, alpha=0.01):
    return alpha * x if x < 0 else x


def prelu_deriv(x, alpha=0.01):
    return alpha if x < 0 else 1


'''
SoftPlus Function
'''


def softplus(x):
    return np.log(1 + np.e ** x)


def softplus_deriv(x):
    return 1 / (1 - np.e ** x)


'''
Bent Identity Function
'''


def bent(x):
    return ((np.sqrt(x ** 2 - 1) - 1) / 2) + x


def bent_deriv(x):
    return x / (2 * np.sqrt(x ** 2 + 1)) + 1


'''
SoftExponential Function
'''


def soft_exp(x, alpha=0.01):
    if alpha < 0:
        return -(np.log(1 - alpha * (x + alpha)) / alpha)
    elif alpha == 0:
        return x
    else:
        return ((np.e ** (alpha * x) - 1) / alpha) + alpha


def soft_exp_deriv(x, alpha=0.01):
    if alpha < 0:
        return 1 / (1 - alpha * (alpha + x))
    else:
        return np.e ** (alpha * x)


'''
SoftClipping Function
'''


def soft_clip(x, alpha=0.01):
    return (1 / alpha) * np.log(
        (1 + np.e ** (alpha ** x)) / (1 + np.e ** (alpha ** (x - 1))))


def soft_clip_deriv(x, p=1):
    def sech(arg):
        return np.cosh(arg) ** (-1)

    return 0.5 * np.sinh(p / 2) * sech(p * x / 2) * sech((p / 2) * (1 - x))


'''
Sinusoid Function
'''


def sinusoid(x):
    return np.sin(x)


def sinusoid_deriv(x):
    return np.cos(x)


'''
Sinc Function
'''


def sinc(x):
    return 1 if x == 0 else np.sin(x) / x


def sinc_deriv(x):
    return 0 if x == 0 else (np.cos(x) / x) - (np.sin(x) / (x ** 2))


'''
Gaussian Function
'''


def gaussian(x):
    return np.e ** -(x ** 2)


def gaussian_deriv(x):
    return -2 * x * np.e ** -(x ** 2)
