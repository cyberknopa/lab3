import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1)
data, target = mnist.data, mnist.target

# Преобразование меток в числовой формат
target = target.astype(np.int)

# Используем только 10% данных
data, _, target, _ = train_test_split(data, target, test_size=0.9, random_state=42)

# Нормализация данных
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Создание модели логистической регрессии
model = LogisticRegression(max_iter=1000)

# Обучение модели
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
