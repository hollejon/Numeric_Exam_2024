"""
Diese Lib enthält einige Python-Beispiele für numerische Funktionen.

"""
import numpy as np
import matplotlib.pyplot as plt


# =================== Zeitvektor aufstellen ===================
# np.linspace(von, bis, anzahl Schritte)
tdata_model = np.linspace(0, 100, 1000)



# =================== Plot erstellen ===================
xdata = np.linspace(0, 10, 100)
ydata = np.sin(xdata)
plt.figure(1)
plt.plot(xdata, ydata, ',', label='Title Graph')
plt.ylabel('y-Achse')
plt.xlabel('x-Achse')
plt.title('Titel Plot')
plt.legend()
plt.grid(True)
plt.show()



# Erstelle  Subplot (1 Reihe, 2 Spalten)
xdata = np.linspace(0, 10, 100)
ydata = np.sin(xdata)
plt.figure(2)
plt.subplot(121)
plt.plot(xdata, ydata, 'o-', label='Title Graph')
plt.ylabel('y-Achse')
plt.xlabel('x-Achse')
plt.title('Titel Plot')
plt.legend()
plt.grid(True)
plt.subplot(122)
plt.plot(xdata, ydata, 'o-', label='Title Graph')
plt.ylabel('y-Achse')
plt.xlabel('x-Achse')
plt.title('Titel Plot')
plt.legend()
plt.grid(True)
plt.show()
