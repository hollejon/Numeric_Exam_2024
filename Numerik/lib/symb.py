import sympy as sp
import numpy as np


# Definiere die Symbole
x0, x1, x2, x3, x4, x5, x6, x7, t, y, y0, R, phi1, phi2 = sp.symbols('x0, x1 x2 x3 x4 x5 x6 x7 t y y0 R phi1 phi2')

# Definiere die Funktionen
#f1 = sp.sin(phi1) * x1 + sp.sin(phi1+phi2) * y
#f2 = sp.cos(phi1) * x1 + sp.cos(phi1+phi2) * y
#f3 = x1 * sp.exp(-x2 * t) * sp.sin(x3 * t + x4)
#
#g1 = x1 / (1 + ((t - x5) / (x7*0.5))**2)
#g2 = x2
#g3 = x3*t
#g4 = x4*t**2
#g5 = x5*t**3
f1 = ((0 - x0)**2) + ((5 - y0)**2) - R**2
f2 = ((4 - x0)**2) + ((-1 - y0)**2) - R**2
f3 = ((-2 - x0)**2) + ((-3 - y0)**2) - R**2


# Erstelle eine Liste der Funktionen
functions = [f1, f2, f3]

def jacobi(f, x):
    # Berechne die Jacobimatrix
    J = sp.Matrix([[sp.diff(f, var) for var in x] for f in functions])
    return J


print("Jacobi:\n",jacobi(functions, [x0, y0, R]))

#jacobi = jacobi(functions, [x0, y0, R])
#print(len(jacobi))
#for i in range(len(jacobi)):
#    print(jacobi[i])

print(np.eye(2))