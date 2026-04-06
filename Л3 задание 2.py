# ========== Задание 2: RUL prediction (C-MAPSS FD001) ==========
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# ------------- Параметры -------------
MIN_SEQ = 50
MAX_SEQ = 150    # мы будем обрезать до 150 последних циклов для долгоживущих двигателей (опционально)
BATCH_SIZE = 64
EPOCHS = 30
LSTM_UNITS = 128
VALIDATION_SPLIT = 0.1
np.random.seed(42)
tf.random.set_seed(42)

# ------------- Загрузка файлов (ожидаем, что файлы в рабочей директории) -------------
# train_FD001.txt: columns: engine_id, cycle, 3 settings, 21 sensors = 26 cols
train_file = 'train_FD001.txt'
test_file = 'test_FD001.txt'   # для FD001
rul_file = 'RUL_FD001.txt'     # RUL targets for test set

if not os.path.exists(train_file):
    raise FileNotFoundError(f"{train_file} not found. Please download FD001 train file from NASA or mirror.")

# читаем (разделитель — пробелы, без заголовка)
col_names = ['engine_id','cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
train_df = pd.read_csv(train_file, sep='\s+', header=None, names=col_names)
print("Train shape:", train_df.shape)
train_df.head()

# ------------- 2) Для каждого engine собрать последовательность и посчитать RUL -------------
# RUL per engine = max(cycle) for that engine (в train сетах)
rul_by_engine = train_df.groupby('engine_id')['cycle'].max().to_dict()
print("Number of engines in train:", len(rul_by_engine))

# ------------- 3) Формируем для каждого двигателя подпоследовательности длиной в диапазоне [MIN_SEQ, MAX_SEQ] -------------
# Функции подготовки подпоследовательностей и целей (RUL relative to last record)
def generate_sequences(df, min_len=MIN_SEQ, max_len=MAX_SEQ, sensors=None, truncate_long=True):
    X_list = []
    y_list = []
    for engine_id, group in df.groupby('engine_id'):
        group = group.sort_values('cycle')
        cycles = group['cycle'].values
        features = group[sensors].values if sensors is not None else group.iloc[:,2:].values
        total_len = len(group)
        # compute RUL for each timestep: RUL = max_cycle - current_cycle
        max_cycle = cycles[-1]
        RULs = max_cycle - cycles  # zero at last record
        # optionally truncate to last max_len cycles
        if truncate_long and total_len > max_len:
            start_idx = total_len - max_len
            features = features[start_idx:]
            RULs = RULs[start_idx:]
            total_len = max_len
        # generate subsequences: for each endpoint i (from min_len to total_len), take last L timesteps ending at i-1
        for end in range(min_len, total_len+1):
            seq = features[:end] if end <= features.shape[0] else features
            if len(seq) >= min_len:
                # target: RUL at last timestep of this subsequence (i.e., how many cycles remain after last record)
                y = float(RULs[end-1])
                X_list.append(seq.copy())
                y_list.append(y)
    return X_list, np.array(y_list, dtype='float32')

# выберем колонки 3-26 (в нашем df это op_1..op_3 + sensor_1..sensor_21) => всего 24?
# По заданию: "использовать колонки 3-26 в качестве входов" — в нашей нумерации это op_1..op_3 + sensor_1..sensor_21
feature_cols = ['op_1','op_2','op_3'] + [f'sensor_{i}' for i in range(1,22)]
print("Using features:", feature_cols[:5], "...", feature_cols[-3:])

# Удаляем константные колонки (по всему train)
train_tmp = train_df.copy()
# detect constant columns
const_cols = [c for c in feature_cols if train_tmp[c].nunique() <= 1]
print("Constant columns (to drop):", const_cols)
active_features = [c for c in feature_cols if c not in const_cols]
print("Active features used:", len(active_features))

# Нормализация (fit scaler на train по каждому двигателю в целом)
scaler = MinMaxScaler()
train_features_all = train_tmp[active_features].values
scaler.fit(train_features_all)  # fit on all train data
# генерируем последовательности
X_seqs, y_vals = generate_sequences(train_tmp, min_len=MIN_SEQ, max_len=MAX_SEQ, sensors=active_features, truncate_long=True)
print("Generated sequences:", len(X_seqs), "Targets shape:", y_vals.shape)

# Нормализуем каждую последовательность
X_seqs_norm = [scaler.transform(seq) for seq in X_seqs]

# ------------- Подготовим данные: упорядочивание по длине и создание батчей -------------
# получим длины
lengths = np.array([s.shape[0] for s in X_seqs_norm])
# разделим на тренировочную и валидационную части (shuffle before split)
indices = np.arange(len(X_seqs_norm))
np.random.shuffle(indices)
split_idx = int(len(indices)*(1.0-VALIDATION_SPLIT))
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

# вспомогательная функция чтобы паддить
def pad_batch(seqs):
    maxlen = max(s.shape[0] for s in seqs)
    return pad_sequences(seqs, maxlen=maxlen, padding='post', dtype='float32')

# Подготовка unordered dataset (random batches with global max padding)
X_train_unordered = [X_seqs_norm[i] for i in train_idx]
y_train_unordered = y_vals[train_idx]
X_val_unordered = [X_seqs_norm[i] for i in val_idx]
y_val = y_vals[val_idx]

# pad everything to global max (for comparison)
global_maxlen = max(s.shape[0] for s in X_train_unordered + X_val_unordered)
X_train_unordered_padded = pad_sequences(X_train_unordered, maxlen=global_maxlen, padding='post', dtype='float32')
X_val_unordered_padded = pad_sequences(X_val_unordered, maxlen=global_maxlen, padding='post', dtype='float32')

# ========== Модель регрессии LSTM (с маскированием) ==========
num_features = len(active_features)
num_responses = 1

def build_regressor(num_features, lstm_units=LSTM_UNITS):
    model = models.Sequential([
        layers.Masking(mask_value=0., input_shape=(None, num_features)),
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        layers.LSTM(max(lstm_units//2, 8), return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_responses, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# ========== Обучение на "неупорядоченных" данных (все padded до global_maxlen) ==========
model_unordered = build_regressor(num_features)
model_unordered.summary()
t0 = time.time()
history_un = model_unordered.fit(X_train_unordered_padded, y_train_unordered,
                                 validation_data=(X_val_unordered_padded, y_val),
                                 epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
time_unordered = time.time() - t0
print("Unordered training time (s):", time_unordered)

# ========== Обучение на "упорядоченных" данных (sorted by length, minimal padding per-batch) ==========
# Создаём пары (seq, y, len) и сортируем по длине затем формируем батчи с динамическим паддингом
train_pairs = [(X_seqs_norm[i], y_vals[i]) for i in train_idx]
train_pairs.sort(key=lambda p: p[0].shape[0])  # sort by length

def batches_from_pairs(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        seqs = [p[0] for p in chunk]
        ys = np.array([p[1] for p in chunk], dtype='float32')
        Xpad = pad_batch(seqs)
        yield Xpad, ys

model_ordered = build_regressor(num_features)
model_ordered.summary()

t0 = time.time()
# обучим несколько эпох вручную по батчам, чтобы каждый батч паддился минимально
for epoch in range(EPOCHS):
    # shuffle groups of similar length? Здесь уже отсортировано, можно разбивать на shuffled groups if desired
    for Xb, yb in batches_from_pairs(train_pairs, BATCH_SIZE):
        model_ordered.train_on_batch(Xb, yb)
    # оценка на валидации (паддим val на максимум длины val)
    # Для простоты pad val to global_maxlen
    val_eval = model_ordered.evaluate(X_val_unordered_padded, y_val, verbose=0)
    # (опционально) печатаем прогресс
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, val_mse: {val_eval[0]:.4f}")
time_ordered = time.time() - t0
print("Ordered training time (s):", time_ordered)

# ========== Оценка: сравнение времени и базового качества на валидации ==========
print("Unordered time:", time_unordered, "Ordered time:", time_ordered)
mse_un = model_unordered.evaluate(X_val_unordered_padded, y_val, verbose=0)
mse_ord = model_ordered.evaluate(X_val_unordered_padded, y_val, verbose=0)  # оцениваем на одном и том же padded val
print("Val MSE unordered:", mse_un, "Val MSE ordered:", mse_ord)

# ========== Прогноз на последней записи двигателя (пример) ==========
# пример: для некоторого двигателя взять последние max_len записей и предсказать RUL
example_engine = 1
group = train_df[train_df['engine_id']==example_engine].sort_values('cycle')
seq_example = group[active_features].values
if seq_example.shape[0] > MAX_SEQ:
    seq_example = seq_example[-MAX_SEQ:]
seq_example_norm = scaler.transform(seq_example)
seq_padded = pad_sequences([seq_example_norm], maxlen=seq_example_norm.shape[0], padding='post', dtype='float32')
pred = model_ordered.predict(seq_padded)
print("Predicted RUL (engine last record):", pred[0][0], "True RUL:", (group['cycle'].max() - group['cycle'].values[-1]))

# ========== Советы по улучшению: ==========
# - ранняя остановка (уже применена для unordered), batch normalization, нормировка по операционному режиму (FD002/FD004)
# - подбор гиперпараметров (LSTM units, learning rate), использование attention, CNN-LSTM комбинаций
# - учёт операционных режимов (кластеризация по op_1..op_3 и нормализация per-regime) для FD002/FD004
# - использование взвешенной ошибки или piecewise-linear target clipping (обычно обрезают RUL до 125/150)
