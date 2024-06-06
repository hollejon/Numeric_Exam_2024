import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Definiere die Funktionen für das Differentialgleichungssystem
def equations(t, y, R1, R2, Ls, Lp, Lps, Lsp):
    I1, I2 = y
    U0 = -4*np.sin(2*np.pi*f*t)
    dI1dt = (-U0/Ls + R2*I2*Lps/Ls - R1*I1/Lp) / (1 - Lps*Lsp/Ls)
    dI2dt = (-U0 + R1*I1) / (Lp/Lsp - Lps)
    return [dI1dt, dI2dt]

# Parameter
f =     10**5         # Frequenz in HZ
R1 =    800.0         # Primärwiderstand in Ohm
Lp =    0.000050      # Primärinduktivität in H    
Lps =   150.0*10**-6  # Streuinduktivität in H
Lsp =   150.0*10**-6  # Streuinduktivität in H
R2 =    6.0           # Sekundärwiderstand in Ohm
Ls =    500*10**-6  # Sekundärinduktivität in H
U0 =    4.0           # Leerlaufspannung in V

# Setze die Anfangswerte für I1 und I2
y0 = [0, 0]

# Definiere die Zeitpunkte, an denen die Lösung berechnet werden soll
t_span = (0, 0.00005)  # Start- und Endzeit
t_eval = np.linspace(0, 0.00005, 10000000)  # Zeitpunkte für die Auswertung
print(t_eval)

# Löse das Differentialgleichungssystem
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, args=(R1, R2, Ls, Lp, Lps, Lsp))

# Plote die Lösung
plt.plot(sol.t, sol.y[0], label='I1(t)')
plt.plot(sol.t, sol.y[1], label='I2(t)')
plt.xlabel('Zeit')
plt.ylabel('Strom')
plt.title('Lösung des Differentialgleichungssystems')
plt.legend()
plt.grid(True)
plt.show()