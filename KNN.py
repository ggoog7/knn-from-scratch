import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)
# Признаки: [температура, осадки]
X = np.array([
    [20, 0],
    [22, 1],
    [18, 0],
    [25, 5],
    [5, 0],
    [0, 10],
    [15, 2],
    [28, 0],
    [-5, 0],
    [10, 8]
])

y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])

X_pos=X[y==1]
X_neg=X[y==0]

plt.scatter(X_pos[:,0],X_pos[:,1], color="green",label="Пойду гулять", s=100)
plt.scatter(X_neg[:,0],X_neg[:,1], color="red",label="Не пойду гулять", s=100)

plt.xlabel('Температура (°C)')
plt.ylabel('Осадки (мм)')
plt.legend()
plt.grid(True) #Сетка
plt.title('Данные для KNN (прогулка или нет)')
plt.show()

def knn_pred(X_train,y_train,x_new,k=3):
    dist=[]

    for i in range(len(X_train)):
        # Евклидово расстояние: sqrt((x1 - x2)^2 + (y1 - y2)^2)
        dist_n= np.sqrt(np.sum((X_train[i] - x_new) ** 2))
        dist.append((dist_n, y_train[i]))

    dist.sort()
    k_neighbors = dist[:k]

    votes = [neighbor[1] for neighbor in k_neighbors]


    most_common = Counter(votes).most_common(1)[0][0]

    return most_common, k_neighbors

#test
new_point = np.array([15, 0])

prediction, neighbors = knn_pred(X, y, new_point, k=3)

print(f"\nНовая точка, температура: {new_point[0]}, осадки: {new_point[1]}")
print(f"Предсказание: {'Пойду гулять!' if prediction == 1 else 'Не пойду гулять'}")

