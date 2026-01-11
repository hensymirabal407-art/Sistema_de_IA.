import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Sistema de IA simple - Predicción de gasto con Árbol de Decisión

dia = int(input("Ingrese el día de la semana (1-7): "))

x = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([120, 220, 320, 420, 520, 1000, 2000])

modelo = DecisionTreeRegressor(random_state=0)
modelo.fit(x, y)

prediccion = modelo.predict([[dia]])

print("Su gasto estimado es:", round(prediccion[0], 2), "dólares")

if prediccion[0] < 200:
    print("Cliente de bajo consumo")
elif prediccion[0] < 400:
    print("Cliente de consumo medio")
else:
    print("Cliente de alto consumo")

plt.scatter(x, y)
plt.plot(x, modelo.predict(x))
plt.scatter(dia, prediccion, marker='x', s=100)
plt.xlabel("Día de la semana")
plt.ylabel("Gasto ($)")
plt.title("Predicción de gasto del cliente (Árbol de Decisión)")
plt.show()
