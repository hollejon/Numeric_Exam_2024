"""
 _   _                           _           _   _____                 _   _                 
| \ | |_   _ _ __ ___   ___ _ __(_) ___ __ _| | |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
|  \| | | | | '_ ` _ \ / _ \ '__| |/ __/ _` | | | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
| |\  | |_| | | | | | |  __/ |  | | (_| (_| | | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
|_| \_|\__,_|_| |_| |_|\___|_|  |_|\___\__,_|_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
"""
"""
LUT(M)..........................................LU decomposition for tridiagonal matrix
LU(A............................................LU decomposition for general matrix A
fbSubsT(LU, b)..................................Solve Ax = b for tridiagonal matrix LU
linsolve(A, b)..................................Solve Ax = b for general matrix A
fbSubs(LR, b)...................................Solve Ax = b for general matrix A
mysign(x).......................................Signum Funktion
e(n)............................................Einheitsvektor
HouseholderTransformation(w)....................Householder transformation
Newton(x0, F, df, tol, K).......................Newton Iteration
rndOrtho(n).....................................random orthogonal matrix
gauss_newton(function, jacobi, x0, xdata, ydata, tol, max_iter,
              damped, maxDampingIter)...........Gauss-Newton Algorithmus für nicht lineare Gleichungen
"""

# To use this library, you need to add the path to the library to your python script.
"""
import sys
sys.path.append("Your absolute path to the library")
import numeric_functions as nf
"""
import matplotlib.pyplot as plt
from matplotlib.pylab import norm
import numpy as np
import scipy as sp
from sympy import solve_triangulated


"""

LU decomposition for tridiagonal matrix
in: a  =  [[0,      a_{21}, ..., a_{n-1,n-2}, a_{n,n-1}],
           [a_{11}, a_{22}, ..., a_{n-1,n-1}, a_{nn}],
           [a_{12}, a_{23}, ..., a_{n-1,n},   0]]

out: LU = [[0,      l_{21}, ..., l_{n-1,n-2}, l_{n,n-1}],
           [r_{11}, r_{22}, ..., r_{n-1,n-1}, r_{nn}],
           [r_{12}, r_{23}, ..., r_{n-1,n},   0]]
           
"""

# LU decomposition for tridiagonal matrix
def LUT(M): 
    n = M.shape[1] 
    for k in range(1,n): 
        for i in range(0,2): 
            if i == 0: 
                M[i][k] = M[i][k]/M[i+1][k-1] 
            else: 
                M[i][k] = M[i][k]-M[i-1][k]*M[i+1][k-1]
    return M



# LU decomposition for general matrix A
def LU(A):
    m = A.shape[0]
    idx = np.array(range(m))    
    #print("M", m, "idx: ", idx, "A: ", A)

    for k in range(0,m):


        for i in range(k+1,m):
            if(A[k][k] == 0):
                #Pivot ist gleich 0 -> Vertausche Zeilen
                # Aktuelle zeile Speichern
                row_act = A[k].copy()
                
                # Zeile überschreiben
                A[k] = A[k+1]
                A[k+1] = row_act

                # P Vektor anpassen
                idx[k] = k+1
                idx[k+1] = k

            A[i][k]= A[i][k]/A[k][k]
 
            for j in range(k+1,m):
                A[i][j] = A[i][j] - A[i][k]*A[k][j]

    return A, idx



"""
in: LU (output from LUT), vector b
out: vector x s.t. L@U@x == b
"""  
# Solve Ax = b for tridiagonal matrix LU
def fbSubsT(LU, b):

    n = len(b)          #calculate length of b
    y = np.zeros(n)     #array with size of n
    x = np.zeros(n)     #array with size of n
    
    #vorwärts
    y[0] = b[0]         #The first element of y is set equal to the first element of vector b
    for i in range(1,n):
        y[i] = b[i] - LU[0, i] * y[i-1]   #For each element of y (starting from the second one), it subtracts the dot prodt of the corresponding row of LU with the elements of y from b[i]
                    #This line computes the last element of x by dividing the last element of y by the last diagonal element of LU    
    #rückwärts
    x[-1]= y[-1] / LU[1,-1]
    for i in range(n-2,-1,-1):
        x[i] = (y[i] - LU[2,i] * x[i+1]) / LU[1,i]  #This loop computes the remaining elements of x iteratively starting from the second last one. It subtracts the dot product of the corresponding row of LU with the elements of x from the corresponding element of y and then divides it by the diagonal element of LU
    
    return x


# lineares Gleichungssystem A*x = b lösen.
def linsolve(A, b):
    M_LU, idx = LU(A)
    #Fuer L*R*x = P*b muessen wir P*b berechnen
    rows_b = len(b)
    P_b = np.zeros(rows_b)
    
    for i, val in enumerate(idx):
        # i := 0,1,2,...
        # val := index der Zeile von P
        P_b[i] = b[val]

    res = fbSubs(M_LU, P_b)
    return res




# Solve Ax = b for general matrix A
def fbSubs(LR, b):
    # code
    n = len(b)   
   
     # Vorwärtseinsetzen
    y = np.zeros(n)
    y[0]=b[0]
    for i in range(1,n):
        y[i] = b[i] - np.dot(LR[i][:i], y[:i])
   
 
    # Rückwärtseinsetzen
    x = np.zeros(n)
    x[n-1]= y[n-1]/LR[n-1][n-1]
    for j in range(n-2, -1, -1):  
        x[j] = (y[j] - np.dot(LR[j][j+1:], x[j+1:]))/LR[j][j]
 
    return x


# Signum Funktion
def mysign(x): # numpy sign liefert 0 für 0
    if x >= 0:
        return 1
    else:
        return -1

# Einheitsvektor
def e(n):
    r = np.array([1]+[0 for k in range(n-1)])
    return r


# Householder transformation
def HouseholderTransformation(w):
    Im = np.eye((len(w)))
    x = w[:,0]
    x_norm = norm(x)
    sgn = mysign(w[0,0])
    e_1 = e(len(w))

    v = x + (sgn * x_norm) * e_1

    #print("Erster Vektor: ", x, x_norm, "\tSGN: ",sgn, w[0,0], "e1: ", e_1, v)

    v_t_v = v.T @ v
    v_v_t = np.outer(v,v) # v mal v.T rechnen
    #print("V: ",np.absolute(w))

    H = Im - ((2* v_v_t) /  v_t_v)
    #print("Im: ", Im)
    #print(v_t_v, v_v_t)
    #print("H:", H)
    return H




def gauss_newton(function, jacobi, x0, xdata, ydata, tol=1e-6, max_iter=100, damped=False, maxDampingIter=100):
    """
    Gauss-Newton Algorithmus für nicht lineare Gleichungen

    Eingabe
    function:   Funktion welche gefitet werden soll
    jacobi:     Jakobi-Matrix der Funktion welche gefittet werden soll: Jacobi = Df(x)
    x0:         Erster Schätzwert der Parameter. Müssen in der nähe der Lösung liegen damit der Algorithmus konvergiert.
    ydata:      Vektor mit Messdaten
    xdata:      Vektor mit Zeitdaten
    max_iter:   Maximale Interationsschritte (default: 100)
    tol:        Toleranz der Kovnergenz (default: 1e-6)

    Rückgabe:
    parameter:  Lösungsvektor, Zahlen der Parameter, damit die Lösung den kleinsten fehler Quadrate der Messdaten entspricht
    """

    k = 0 #Anzahl Interationen
    parameter = x0.copy() #muss in der nähe der Lösung liegen damit es konvergiert
    y = function(xdata, parameter) - ydata
    j = jacobi(xdata, parameter)

    r = np.array([np.linalg.norm(j.T@y)])
    print("k", "x1\t", "x2\t", "x3\t", "x4\t", "r\t\t", "s")
    
    while r > tol and k < max_iter:
        j = jacobi(xdata, parameter)
        y = function(xdata, parameter) - ydata
       
        # Normalengleichung lösen
        #deltax = np.linalg.solve(j.T@j, -j.T@y)

        # Mit QR-Zerlegung
        Q,R = np.linalg.qr(j)
        deltax = solve_triangulated(R, -Q.T@y, lower=False)
        # Falls Dämpfung aktiviert ist
        if damped == True:
            delta_k = 1
            n = 1
            while  np.linalg.norm(parameter + deltax*delta_k) > np.linalg.norm(parameter) and n < maxDampingIter:
                print("Dämpfung", n)
                delta_k = delta_k / 2
                n = n + 1
            deltax = deltax*delta_k

        parameter = parameter + deltax
        k += 1

        r = np.linalg.norm(j.T@y)
        s = np.sum((y-function(xdata, parameter))**2)
        print(k, round(parameter[0], 3), 
               round(parameter[1], 3), "\t",
               round(parameter[2], 3), "\t",
               round(parameter[3], 3),"\t",
               "{:.3e}\t".format(r),
               "{:.3e}".format(s))

    return parameter


# Newton Iteration
def Newton(x0, F, df, tol = 1e-5, K = 1000):
    k=0
    x = x0
    r = 1
    res_k = np.array([])

    while(r > tol and k < K):
        k = k+1
        J = df(x)

        # y berechnen
        y = F(x)

        # Gleichungssystem lösen
        ATA = np.dot(J.T,J)
        ATB = np.dot(J.T,y) 

        dx = np.linalg.solve(ATA,ATB)
        #print("Resultat: ", x, dx)
        x = x - dx
        
        r = np.linalg.norm(F(x))  
        res_k = np.append(res_k,[r],axis= 0)
        #print("Residuum: ", r, "\tAktueller Wert: ", x, "\tParameter: ", dx, "\tIteration: ", k)

    return x, res_k

# ============ Beispiel für Newton Iteration: ============
# Nullstellenform der Modellfunktion
#
#def model(x):
#    p = np.array([((0 - x[0])**2) + ((5 - x[1])**2) - x[2]**2,
#                  ((4 - x[0])**2) + ((-1 - x[1])**2) - x[2]**2,
#                   ((-2 - x[0])**2) + ((-3 - x[1])**2) - x[2]**2])
#    return p
#
#def modeldf(x):
#    return np.array([[2*x[0], 2*x[1] - 10, -2*x[2]], 
#                     [2*x[0] - 8, 2*x[1] + 2, -2*x[2]], 
#                     [2*x[0] + 4, 2*x[1] + 6, -2*x[2]]])
#
#
#
#x0 = [1,1,1]
#print("Ergebnis: ", Newton(x0, model, modeldf))





# Erzeugen von speziellen Matrizen

# random orthogonal matrix
def rndOrtho(n):
    S = np.random.rand(n,n)
    S = S - S.T
    O = np.linalg.solve(S - np.identity(n), S + np.identity(n))
    return O


# random matrix with specified condition number 
# n =: Zeilenzahl 
# cond =: geforderte Kondition
def rndCond(n, cond):    
    d = np.logspace(-np.log10(cond)/2, np.log10(cond)/2,n);
    A = np.diag(d)
    U,V = rndOrtho(n), rndOrtho(n)
    return U@A@V.T



"""
https://docs.scipy.org/doc/scipy/reference/integrate.html
def DGLs(y_vec,t):
    y_1, y_2 = y_vec
    dy_1_dt = 3*y_1 + 2*y_2 - (2*t**2 + 1)*np.exp(2*t)
    dy_2_dt = 4*y_1 + y_2 + (t**2 +2*t - 4)*np.exp(2*t)
    return np.array([dy_1_dt,dy_2_dt])


def dgl2(y_vec,t):
    y_1, y_2 = y_vec
    dy_1_dt = (t**2)/y_1
    dy_2_dt = 0
    return np.array([dy_1_dt,dy_2_dt])

N=100
tval = np.linspace(0, 1, N+1)
y_1_0 = 0
y_2_0 = -4
initialval = np.array([y_1_0,y_2_0])
Loes = sp.integrate.odeint(dgl2, initialval, tval)



plt.xlabel(r"$\rm t$")
plt.ylabel(r"$\rm y_1,y_2$")
plt.plot(tval, Loes[:, 0],c="blue", label=r"$\rm y_1(t)$")
plt.plot(tval, Loes[:, 1],c="red", label=r"$\rm y_2(t)$")
plt.legend(loc='upper center',fontsize=16)
plt.grid(True)
plt.show()

"""