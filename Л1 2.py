# КОД: классификация по 9 регионам, образованным 4-мя перпендикулярными прямыми
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Установки воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

# Параметры датасета
n_samples = 1000  # общее число точек
X = np.random.rand(n_samples, 2).astype(np.float32)  # точки в [0,1]x[0,1]

# Определяем 4 линии: x = a1, x = a2, y = b1, y = b2
a1, a2 = 0.25, 0.75  # вертикали
b1, b2 = 0.25, 0.75  # горизонтали

# Присвоение класса: 3x3 сетка -> 9 классов (индексация: row*3 + col)
# где col = 0 (x < a1), 1 (a1 <= x <= a2), 2 (x > a2)
# аналогично row по y: 0 (y < b1), 1 (b1 <= y <= b2), 2 (y > b2)
cols = np.where(X[:,0] < a1, 0, np.where(X[:,0] > a2, 2, 1))
rows = np.where(X[:,1] < b1, 0, np.where(X[:,1] > b2, 2, 1))
y_int = rows * 3 + cols  # метки 0..8

# One-hot
y_cat = to_categorical(y_int, num_classes=9)

# Посмотрим распределение по классам
(unique, counts) = np.unique(y_int, return_counts=True)
print("Counts per class (0..8):", dict(zip(unique.tolist(), counts.tolist())))

# Создаём модель (MLP)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем (с валидацией)
history = model.fit(X, y_cat, epochs=120, batch_size=32, validation_split=0.2, verbose=0)

# Графики обучения
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
plt.tight_layout()
plt.show()

# Визуализация решающего пространства
grid_n = 400
xx, yy = np.meshgrid(np.linspace(0,1,grid_n), np.linspace(0,1,grid_n))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
preds = np.argmax(model.predict(grid, verbose=0), axis=-1).reshape(xx.shape)

plt.figure(figsize=(7,7))
plt.contourf(xx, yy, preds, alpha=0.5, levels=np.arange(-0.5,9.5,1), cmap='tab10')
# Нанесём границы линий (чёткие линии для a1,a2,b1,b2)
plt.axvline(a1, color='k', linewidth=1.2)
plt.axvline(a2, color='k', linewidth=1.2)
plt.axhline(b1, color='k', linewidth=1.2)
plt.axhline(b2, color='k', linewidth=1.2)

# Наносим обучающие точки
scatter = plt.scatter(X[:,0], X[:,1], c=y_int, edgecolors='k', cmap='tab10', s=25)
plt.title('Решающее пространство (9 классов) и обучающая выборка')
plt.xlim(0,1); plt.ylim(0,1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Отдельно — легенда и пример разметки 3x3
plt.figure(figsize=(5,5))
for cls in range(9):
    # для примера возьмём точку-метку в центре соответствующего блока
    row = cls // 3
    col = cls % 3
    cx = ( [0, a1, a2, 1][col] + [0, a1, a2, 1][col+1] ) / 2 if False else ( (0.125, 0.5, 0.875)[col] )
    cy = ( (0.125, 0.5, 0.875)[row] )
    plt.text(cx, cy, str(cls), ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
plt.axvline(a1, color='k'); plt.axvline(a2, color='k'); plt.axhline(b1, color='k'); plt.axhline(b2, color='k')
plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal')
plt.title('Нумерация классов 0..8 (3x3 сетка)')
plt.show()

print("Train accuracy (last):", history.history['accuracy'][-1])
print("Val accuracy (last):", history.history['val_accuracy'][-1])
