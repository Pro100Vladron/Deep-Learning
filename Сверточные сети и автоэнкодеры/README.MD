1) Image Classification (Transfer Learning)

Реализована классификация изображений Fashion-MNIST с использованием MobileNetV2 и подхода transfer learning.
Проведено сравнение двух стратегий обучения:

frozen pretrained layers
full fine-tuning

Сравнение выполнялось по:

accuracy
loss
training speed
generalization quality
2) Image Denoising (Autoencoder)

Построены две архитектуры convolutional autoencoder для удаления шума с изображений MNIST:

базовая 2-layer модель
улучшенная 3-layer модель

Качество сравнивалось по:

MSE
PSNR
visual reconstruction quality
