import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from numpy.linalg import cholesky

data = np.genfromtxt('Praktikum4/data.txt')
t = data[:,0]         
y_m = data[:,1]                  

# Abtastrate berechnen
sampling_rate = 1 / (t[1] - t[0])
print(f"Abtastrate: {sampling_rate} Hz")

# Grundfrequenz schätzen
T = t[-1] - t[0]                    # Periode schätzen
f0 = 1 / T                          # Grundfrequenz schätzen
print(f"Grundfrequenz: {f0} Hz")

# Systemmatrix für f5(t) berechnen
w0 = 1
n = 5
A = np.zeros((2*n + 1, len(t)))

for i in range(0, n+1):
    A[2*i] = np.cos(i * w0 * t)                 
    if i > 0:                            
        A[2*i-1] = np.sin(i * w0 * t)

print(f"Dimension der Systemmatrix: {A.shape}")

# Normalgleichung berechnen
AtA = A @ A.T
print(f"Dimension der Normalgleichung: {AtA.shape}")

# Lösen Sie mit Hilfe der Cholesky-Zerlegung
L = cholesky(AtA)
c = np.linalg.solve(L, A @ y_m)

# Cholesky-Zerlegung der Matrix AtA
L = cholesky(AtA)

# Lösen des linearen Systems
y = solve_triangular(L, A @ y_m, lower=True)
c = solve_triangular(L.T, y, lower=False)

# Berechnen von fn(t)
fn_t = np.zeros_like(t)
for i in range(0, n+1):
    fn_t += c[2*i] * np.cos(i * w0 * t)
    if i > 0:
        fn_t += c[2*i-1] * np.sin(i * w0 * t)

print(f"fn(t) = {fn_t}")

# Visualisieren Sie die Lösung
plt.figure(figsize=(10, 5))
plt.plot(t, y_m, 'o', label='Messwerte', markersize=2)
plt.plot(t, fn_t, label='Approximierende Funktion')
plt.title('Messwerte und Approximierende Funktion')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.show()

# Quadratische Fehlersumme berechnen
error = np.sum((y_m - fn_t)**2)
print(f"Quadratische Fehlersumme: {error}")




# class myFit:
#   def __init__(self, data):
#       self.setData(data)
#       self.c = None       # Klassen Variable für die Modell Koeffizienten
#       self.omega = omega

#   def setData(self, data):
#       self.ti = data[:,0] # Zeitstempel
#       self.ui = data[:,1] # Messwerte

#   def computeCoefs(self, n=5):
#       self.n = n

#     #   <<snipp>>

#     #   self.c = <<snipp>>

#   def compute(self,t):
#       if not type(mf.c) == np.ndarray:
#           self.computeCoefs()

#     #   y = <<snipp>>

#       return y
  
# # Objekt instanzieren
# mf = myFit(data)
# # Koeffizienten berechnen
# mf.computeCoefs(5)
# # Visualisieren
# plt.plot(t,mf.compute(t))
# plt.show()