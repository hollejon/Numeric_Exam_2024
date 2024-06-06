"""
                                        _____            _                                                                  
                                       |  __ \          | |                                                                 
  _ __    _   _   _ __ ___       __ _  | |  | |   __ _  | |  _ __                                                           
 | '_ \  | | | | | '_ ` _ \     / _` | | |  | |  / _` | | | | '_ \                                                          
 | | | | | |_| | | | | | | |   | (_| | | |__| | | (_| | | | | | | |                                                         
 |_| |_|  \__,_| |_| |_| |_|    \__, | |_____/   \__, | |_| |_| |_|                                                         
                                 __/ |            __/ |                                                                     
                                |___/            |___/                    
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Uebersicht:

def euler_forward(x0, X, N, f):    
    x0: Startwert x, y
    X: Endwert
    N: Anzahl Schritte
    f: Funktion f(x, y) = y'

def euler_backward(x0, X, N, f, df, tol):
    x0: Startwert x, y
    X: Endwert
    N: Anzahl Schritte
    f: Funktion f(x, y) = y'
    df: Jacobi-Matrix der Funktion f
    tol: Toleranz

def calc_error(x_data, y_data, model_analytisch):
    x_data: x-Werte - Numerisch
    y_data: y-Werte - Numerisch
    model_analytisch: analytische Funktion

"""   



# Euler-vorwaerts / RK explizit
def euler_forward(x0, X, N, f):
    h = (X-x0[0])/N
    N = N+1
    #print("EXP: Schrittweite: ", h)

    x = np.zeros((N))
    y = np.zeros((N))
    x[0] = x0[0]
    y[0] = x0[1]

    for i in range(1, N):
        x[i] = x[i-1] + h
        k = f(x[i-1], y[i-1])
        y[i] = y[i-1] + h*k

    return x, y


"""
# =============== Example Euler-Vorwaerts / RK-Explizit ===============
def model_analytisch(x):
    return np.exp(-4 * x)

def model1(x, y):
    return -4 * y

x0 = [0, 1] # Startwert x, y
res_x, res_y = euler_forward(x0, 2, 20, model1)

x_an = np.linspace(0,2,100)
plt.plot(x_an, model_analytisch(x_an), '-', label='y(x) Analytisch')
plt.plot(res_x, res_y, 'o-', label='Euler Vorwaerts')
plt.ylabel('y-Achse')
plt.xlabel('x-Achse')
plt.legend()
plt.grid(True)
plt.show()

"""


# Euler-rueckwaerts / RK implizit
def euler_backward(x0, X, N, f, df, tol):
    # x0: Startpunkt
    # X: Endpunkt
    # f: Funktion
    # df: Jacobi Matrix der Funktion f
    # N: Anzahl der Schritte
    # tol: Toleranz
    max_iter=20

    h= (X-x0[0])/N
    N = N+1
    #print( "IMP: Schrittweite: ", h)
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0[0]
    y[0] = x0[1]

    #Anfangspunkt berechnen
    k = f(x[0],y[0])

    for i in range(1,N):
        step = 0
        x[i] = x[i-1] + h
        r = k - f(x[i-1] + h, y[i-1] + (h*k))

        while np.abs(r) > tol and step < max_iter:
            j = df(x[i-1] + h, y[i-1] + (h*k))
            delta_k = -r / (1 - (h*j))
            k = k + delta_k
            r = k - f(x[i-1] + h, y[i-1] + (h*k))
            step += 1
        y[i] = y[i-1] + (h*k)
    return x,y

"""
# =============== Example: Euler-Rueckwaerts / RK-Implizit ===============
def model2(x,y):
    return -4*y

def model2_df(x,y):
    return -4

def model_analytisch(x):
    return np.exp(-4 * x)

x0 = (0, 1)
x_imp,y_imp = euler_backward(x0, 2, 10, model2, model2_df ,1e-3)

x_an = np.linspace(0,2,100)
plt.plot(x_imp, y_imp, 'o-', label='Implizites Euler-Verfahren')
plt.plot(x_an, model_analytisch(x_an), '-', label='y(x) Analytisch')
plt.ylabel('y-Achse')
plt.xlabel('x-Achse')
plt.title('Schrittweite: 0.2')
plt.legend()
plt.grid(True)
plt.show()
"""

# Berechnet den Fehler zwischen numerischer und analytischer Lösung

def calc_error(x_data, y_data, model_analytisch):
    r = []
    for i in range(len(x_data)):
        print(x_data[i])
        x_anal = model_analytisch(x_data[i])
        r.append(abs(y_data[i] - x_anal))
    return r

#plt.semilogy(x_num, calc_error(x_num, y_num), 'o-', label='Fehler')







