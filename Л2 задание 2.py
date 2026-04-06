import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("=== ЗАДАНИЕ №2: АВТОЭНКОДЕРЫ ===")

# 1. Загрузка и подготовка данных MNIST
print("1. Загрузка данных MNIST...")
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Нормализация и reshaping
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print(f"Размер тренировочной выборки: {x_train.shape}")
print(f"Размер тестовой выборки: {x_test.shape}")

# 2. Добавление шума
print("2. Добавление шума...")
def add_noise(images, noise_factor=0.5):
    noise = np.random.normal(loc=0.5, scale=0.5, size=images.shape)
    noisy_images = images + noise_factor * noise
    return np.clip(noisy_images, 0.0, 1.0)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# 3. БАЗОВАЯ МОДЕЛЬ (из примера) - 2 слоя свертки/развертки
print("3. Создание базовой модели (2 слоя)...")

def create_basic_autoencoder():
    # Энкодер
    encoder_inputs = keras.Input(shape=(28, 28, 1), name='encoder_input')
    
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(encoder_inputs)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    
    shape_before_flatten = x.shape
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(16, name='latent_vector')(x)
    
    encoder = keras.Model(encoder_inputs, latent, name='encoder')
    
    # Декодер
    latent_inputs = keras.Input(shape=(16,), name='decoder_input')
    x = keras.layers.Dense(shape_before_flatten[1] * shape_before_flatten[2] * shape_before_flatten[3])(latent_inputs)
    x = keras.layers.Reshape((shape_before_flatten[1], shape_before_flatten[2], shape_before_flatten[3]))(x)
    
    x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = keras.layers.Conv2DTranspose(1, (3, 3), padding='same')(x)
    outputs = keras.layers.Activation('sigmoid', name='decoder_output')(x)
    
    decoder = keras.Model(latent_inputs, outputs, name='decoder')
    
    # Автоэнкодер
    autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name='autoencoder_basic')
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder, decoder

# 4. АЛЬТЕРНАТИВНАЯ МОДЕЛЬ - 3 слоя свертки/развертки
print("4. Создание альтернативной модели (3 слоя)...")
def create_advanced_autoencoder():
    # Энкодер
    encoder_inputs = keras.Input(shape=(28, 28, 1), name='encoder_input')
    
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    shape_before_flatten = x.shape
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(32, name='latent_vector')(x)  # Увеличили размерность латентного пространства
    
    encoder = keras.Model(encoder_inputs, latent, name='encoder_advanced')
    
    # Декодер
    latent_inputs = keras.Input(shape=(32,), name='decoder_input')
    x = keras.layers.Dense(shape_before_flatten[1] * shape_before_flatten[2] * shape_before_flatten[3])(latent_inputs)
    x = keras.layers.Reshape((shape_before_flatten[1], shape_before_flatten[2], shape_before_flatten[3]))(x)
    
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(1, (3, 3), padding='same')(x)
    outputs = keras.layers.Activation('sigmoid', name='decoder_output')(x)
    
    decoder = keras.Model(latent_inputs, outputs, name='decoder_advanced')
    
    # Автоэнкодер
    autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)), name='autoencoder_advanced')
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder, decoder

# 5. Обучение моделей
print("5. Обучение моделей...")

# Базовая модель (2 слоя)
autoencoder_basic, encoder_basic, decoder_basic = create_basic_autoencoder()
print("Базовая модель (2 слоя):")
autoencoder_basic.summary()

history_basic = autoencoder_basic.fit(
    x_train_noisy, x_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# Альтернативная модель (3 слоя)
autoencoder_advanced, encoder_advanced, decoder_advanced = create_advanced_autoencoder()
print("\nАльтернативная модель (3 слоя):")
autoencoder_advanced.summary()

history_advanced = autoencoder_advanced.fit(
    x_train_noisy, x_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (пункты 2.1 и 2.2)
print("6. Визуализация результатов...")

# Предсказания обеих моделей
decoded_imgs_basic = autoencoder_basic.predict(x_test_noisy)
decoded_imgs_advanced = autoencoder_advanced.predict(x_test_noisy)

# Визуализация сравнения моделей
n = 10  # Количество примеров для отображения
plt.figure(figsize=(20, 8))

for i in range(n):
    # Оригинальные зашумленные
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title('Noisy')
    plt.axis('off')
    
    # Базовая модель (2 слоя)
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_imgs_basic[i].reshape(28, 28), cmap='gray')
    plt.title('Basic (2 layers)')
    plt.axis('off')
    
    # Альтернативная модель (3 слоя)
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs_advanced[i].reshape(28, 28), cmap='gray')
    plt.title('Advanced (3 layers)')
    plt.axis('off')
    
    # Оригинальные чистые
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

plt.suptitle('Сравнение моделей автоэнкодеров', fontsize=16)
plt.tight_layout()
plt.show()

# 7. СРАВНЕНИЕ КАЧЕСТВА (пункт 2.3)
print("7. Сравнение качества моделей...")

# Оценка моделей на тестовых данных
test_loss_basic = autoencoder_basic.evaluate(x_test_noisy, x_test, verbose=0)
test_loss_advanced = autoencoder_advanced.evaluate(x_test_noisy, x_test, verbose=0)
print("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
print(f"Базовая модель (2 слоя):")
print(f"  - Test Loss: {test_loss_basic:.4f}")
print(f"  - Параметры: {autoencoder_basic.count_params():,}")

print(f"\nАльтернативная модель (3 слоя):")
print(f"  - Test Loss: {test_loss_advanced:.4f}")
print(f"  - Параметры: {autoencoder_advanced.count_params():,}")

# Сравнение потери
if test_loss_advanced < test_loss_basic:
    improvement = ((test_loss_basic - test_loss_advanced) / test_loss_basic) * 100
    print(f"\n✅ Альтернативная модель лучше на {improvement:.1f}%")
else:
    improvement = ((test_loss_advanced - test_loss_basic) / test_loss_advanced) * 100
    print(f"\n✅ Базовая модель лучше на {improvement:.1f}%")

# 8. Графики обучения
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_basic.history['loss'], label='Basic Training')
plt.plot(history_basic.history['val_loss'], label='Basic Validation')
plt.plot(history_advanced.history['loss'], label='Advanced Training')
plt.plot(history_advanced.history['val_loss'], label='Advanced Validation')
plt.title('Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
# PSNR (Peak Signal-to-Noise Ratio) - метрика качества изображений
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr

psnr_basic = calculate_psnr(x_test, decoded_imgs_basic)
psnr_advanced = calculate_psnr(x_test, decoded_imgs_advanced)

models = ['Basic (2 layers)', 'Advanced (3 layers)']
psnr_values = [psnr_basic, psnr_advanced]

plt.bar(models, psnr_values, color=['blue', 'orange'])
plt.title('PSNR Comparison (Higher is Better)')
plt.ylabel('PSNR (dB)')

plt.tight_layout()
plt.show()

# 9. Сохранение моделей
autoencoder_basic.save('autoencoder_basic_2layers.h5')
autoencoder_advanced.save('autoencoder_advanced_3layers.h5')
print("\nМодели сохранены:")
print("- autoencoder_basic_2layers.h5")
print("- autoencoder_advanced_3layers.h5")

print("\n=== ЗАДАНИЕ №2 ВЫПОЛНЕНО! ===")