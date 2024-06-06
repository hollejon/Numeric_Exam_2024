import numpy as np
import matplotlib.pyplot as plt

# Parameter
f =     10**5       # Frequenz in HZ
R1 =    800.0         # Primärwiderstand in Ohm
Lp =    0.00005   # Primärinduktivität in H    
Lps =   150.0*10**-6  # Streuinduktivität in H
Lsp =   150.0*10**-6  # Streuinduktivität in H
R2 =    6.0           # Sekundärwiderstand in Ohm
Ls =    500*10**-6  # Sekundärinduktivität in H
U0 =    4.0           # Leerlaufspannung in V

print("f:", f, " R1:", R1, " Lp:", Lp, " Lps:", Lps, " Lsp:", Lsp, " R2:", R2, " Ls:", Ls, " U0:", U0)

# DGL System Ax =b
#A = np.array([[(R1*1/Lp)/(1-Lps*Lsp*1/Ls),(R2*1/Ls*Lps)/(1-Lps*Lsp*1/Ls)],
#              [(R2*(1/Ls)*Lsp*Lps)/(1+U0*(1/Lp)*Lsp*Lps),(R2*(1/Ls)*Lsp)/(1+U0*(1/Lp)*Lsp*Lps*(1/Ls))]])
#
#b = np.array([(U0*(1/Ls))/(1-Lps*Lsp*(1/Ls)),0])

def runga_kutta_RK4_2Systeme(x0, tend, h, f):
    # x0: Anfangswerte
    # y0: Anfangswerte
    # tend: Zeit bis zum Ende
    # h: Schrittweite
    # f: Funktion mit n Gleichungen
    N = int(tend / h) +1
    # Vektor mit dx-daten und x-daten
    x = np.array([np.zeros((N)), np.zeros((N))])
    # Zeitvektor
    t = np.zeros((N))
    
    x[0][0] = x0[0]
    x[1][0] = x0[1]

    for i in range(1, N):
        t[i] = t[i-1] + h

        k1 = f([t[i], x[0][i-1],               x[1][i-1]])
        k2 = f([t[i], x[0][i-1] + 0.5*h*k1[1], x[1][i-1] + 0.5*h*k1[0]])
        k3 = f([t[i], x[0][i-1] + 0.5*h*k2[1], x[1][i-1] + 0.5*h*k2[0]])
        k4 = f([t[i], x[0][i-1] + h*k3[1],     x[1][i-1] + h*k3[0]])

        x[0][i] = x[0][i-1] + h*((1/6)*k1[1] + (1/3)*k2[1] + (1/3)*k3[1] + (1/6)*k4[1]) #dx
        x[1][i] = x[1][i-1] + h*((1/6)*k1[0] + (1/3)*k2[0] + (1/3)*k3[0] + (1/6)*k4[0]) #x

    return x, t


def RK4(t0,x0,h,tend,f):
    N = int((tend - t0) / h)
    t = np.zeros(N+1)
    x = np.zeros((N+1,np.shape(x0)[0]))
    
    t[0] = t0
    x[0,:] = x0
    for i in range(1, N+1):
        t[i] = t[i-1] + h

        k1 = f(t[i-1], x[i-1,:])
        k2 = f(t[i-1] + 0.5*h, x[i-1,:] + 0.5*h*k1)
        k3 = f(t[i-1] + 0.5*h, x[i-1,:] + 0.5*h*k2)
        k4 = f(t[i-1] + h, x[i-1,:] + h*k3)
        x[i,:] = x[i-1,:] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return t, x

#Versuch 1
def model1(t, x):
    #t = x[0]
    #x1 = x[1] # I1
    #x0 = x[2] # I2
    t = t
    x0 = x[0]
    x1 = x[1]

    U0 = 4*np.sin(2*np.pi*f*t)
    return np.array([(Lsp*Lps*(1/Ls)*R1*x0*(1/Lp) + R2*x1*(1/Ls)*Lsp) / (1+U0*(1/Lp)*Lsp*Lps*(1/Ls)),
                      (U0*(1/Ls) + R2*x1*(1/Ls)*Lps - R1*x0*(1/Lp))/(1-Lps*Lsp*(1/Ls))])

#Versuch 2
def model2(t, x):
    #t = x[0]
    #x0 = x[1] # I1
    #x1 = x[2] # I2

    t = t
    x1 = x[0]
    x0 = x[1]

    U0 = 4*np.sin(2*np.pi*f*t)
    return np.array([(R2*x0*Lps*Ls - R1*x1*(1/Lp)-U0) / (1 - (Lsp*Lps*Ls)),
                      ((R1*x1)-U0)/((Lp/Lsp)-Lps)])

x0 = (0.0, 0.0)
tend = 0.00005
h =    0.000000001


#I_x, t = runga_kutta_RK4_2Systeme(x0, tend, h, model2)
t2, I_x2 = RK4(0,x0,h,tend, model2)



# =================== Plot erstellen ===================
plt.figure(1)
#plt.plot(t, I_x[0],  '.-', label='v0=0')
plt.plot(t2, I_x2[:,0],  '.-', label='I1')
plt.ylabel('I1[A]')
plt.xlabel('t')
plt.title('RK4 mit Schrittweite h=0.1ns')
plt.legend()
plt.grid(True)

plt.figure(2)
#plt.plot(t, I_x[1],  '.-', label='v0=0')
plt.plot(t2, I_x2[:,1],  '.-', label='v0=0')
plt.ylabel('I2[A]')
plt.xlabel('t')
plt.title('RK4 mit Schrittweite h=0.1ns')
plt.legend()
plt.grid(True)
plt.show()


# ================================ Eueler Vorwaerts ================================

def eulerforward_2Systeme(x0, h, tend, f):
    # x0: Anfangswerte
    # y0: Anfangswerte
    # tend: Zeit bis zum Ende
    # h: Schrittweite
    # f: Funktion mit n Gleichungen
    N = int(tend / h) +1

    # Vektor mit dx-daten und x-daten
    x = np.zeros((N, np.shape(x0)[0]))

    # Zeitvektor
    t = np.zeros((N))
    
    x[0][0] = x0[0]
    x[0][1] = x0[1]

    for i in range(1, N):
        t[i] = t[i-1] + h

        k1 = f(t[i], x[i-1,:])
        x[i, 0] = x[i-1, 0] + h*k1[1] #dx
        x[i, 1] = x[i-1, 1] + h*k1[0] #x

    return x, t

# ================================ Implizite Trapezregel ================================
def implizit_trapezverfahren(x0, X, f, df, N, tol):
    # x0: Startpunkt
    # X: Endpunkt
    # f: Funktion
    # df: Jacobi Matrix der Funktion f
    # N: Anzahl der Schritte
    # tol: Toleranz
    max_iter=20

    h = (X-x0[0])/(N-1)
    print( "IMP: Schrittweite: ", h)
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0[0]
    y[0] = x0[1]
    s = 0

    for i in range(1,N):
        #print("K: ", k)
        step = 0
        
        x[i] = x[i-1] + h
        k_1 = f(x[i-1], y[i-1])
        k_2 = k_1
        r = k_2 - f(x[i-1]+h, y[i-1] + h*(k_1*0 + k_2*0))

        J = -df(t[i-1] + h*0.5, x[i-1] + (h*k*0.5))
        # Einheitsmatrix:
        I2 = np.eye(2)
        M = I2 - h * 0.5 * J
        
        while np.abs(r) > tol and step < max_iter:
            j = -df(x[i-1] + h*1, y[i-1] + h*(k_1*0 + k_2*0))
            delta_k = -r / (1-h*j)
            k_2 += delta_k
            r = k_2 - f(x[i-1] + h*1, y[i-1] + h*(k_1*0 + k_2*0))

            b = -r
            ATA = np.dot(M.T, M)
            ATb = np.dot(M.T, b)
            delta_k = np.linalg.solve(ATA, ATb)
            k = k + delta_k


            step += 1
        
        #print("k's: ", kl[0], kl[1])
        y[i] = y[i-1] + h*(k_1*0.5 + k_2*0.5)
    return x, y


# Lösung mit Euler implizit
def f(x,y):
     
     t = x[0]
     x0 = x[1] # I1
     x1 = x[2] # I2
     return np.array([(R2*x0*Lps*Ls - R1*x1*(1/Lp)-U0) / (1 - (Lsp*Lps*Ls)),
                      ((R1*x1)-U0)/((Lp/Lsp)-Lps)])


def df(x,y):

    return np.array([[-(R1*1/Lp)/(1-Lps*Lsp*1/Ls),-(R2*1/Ls*Lps)/(1-Lps*Lsp*1/Ls)],[(R1/(Lp/Lsp)-Lps),0]])



        

# Funktion Euler Implizit Eindimensional -> umschreiben auf 2D Stimmt no hine und vorne nöd !!
def RK_implizit(x0, X, N, f, df, tol):
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



x0 = (0.0, 0.0)
tend = 0.00005
h =    0.000000001
tol = 0.0001


I_x, t = RK_implizit(x0, tend, 1000, f, df, tol)
#t2, I_x2 = RK4(0,x0,h,tend, model2)



# =================== Plot erstellen ===================
plt.figure(4)
plt.plot(t, I_x[0],  '.-', label='v0=0')
#plt.plot(t2, I_x2[:,0],  '.-', label='I1')
plt.ylabel('I1[A]')
plt.xlabel('t')
plt.title('RK Implizit')
plt.legend()
plt.grid(True)


plt.figure(5)
plt.plot(t, I_x[1],  '.-', label='v0=0')
#plt.plot(t2, I_x2[:,1],  '.-', label='v0=0')
plt.ylabel('I2[A]')
plt.xlabel('t')
plt.title('RK4 mit Schrittweite h=0.1ns')
plt.legend()
plt.grid(True)
plt.show()