import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = "F:\\pythonProject程序设计\\python大作业\\数据\\预测.xlsx"
data = pd.read_excel(file_path)
print(data.head())
# 将日期列转换为日期时间格式，并按日期排序
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')
# 检查缺失值
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
# 填充缺失值，暂定用0填充
data = data.fillna(0)

# 对数量数据进行归一化处理
scaler = MinMaxScaler()
data['qty_scaled'] = scaler.fit_transform(data[['qty']])

# 创建时间序列数据
data_grouped = data.groupby(['seller_no', 'product_no', 'warehouse_no'])

# 创建滑动窗口的函数
def create_sequences(input_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:i+seq_length]
        y = input_data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return xs, ys

seq_length = 30 # 选择序列长度,可能调参

X, y, groups = [], [], []
for name, group in data_grouped:
    group_qty = group['qty_scaled'].values
    xs, ys = create_sequences(group_qty, seq_length)
    X.extend(xs)
    y.extend(ys)
    groups.extend([name] * len(xs))

X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))# 调整数据输入形状，LSTM层需要输入数据的形状为(样本数, 时间步数, 特征数)
print("X shape:", X.shape)
print("y shape:", y.shape)

# 划分数据集，这里也要调参数，其中random_state不用调，它确保每次运行代码时，数据集的划分是一致的。
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups,train_size=0.7, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# 构建一个包含两个LSTM层和一个全连接层的神经网络模型。每个LSTM层有50个单元
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])

# 编译模型，使用Adam优化器和均方误差损失函数，学习率参数可以尝试修改
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009), loss='mean_squared_error')

# 训练模型,这里可能要调参
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
model.save('lstm_model.keras')

# 用测试数据进行预测
y_pred = model.predict(X_test)

# 将预测结果逆归一化
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 评估模型，选择均方误差mse
mse = mean_squared_error(y_test_inv, y_pred_inv)
print("Mean Squared Error:", mse)

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 输出测试集上的预测结果
results_test = []
for i, (seller_no, product_no, warehouse_no) in enumerate(groups_test):
    results_test.append([seller_no, product_no, warehouse_no, y_test_inv[i][0], y_pred_inv[i][0]])

results_test_df = pd.DataFrame(results_test, columns=['seller_no', 'product_no', 'warehouse_no', 'qty_true', 'qty_pred'])
print(results_test_df.head())
results_test_df.to_csv('prediction_results_test.csv', index=False)

# 预测2023-05-16至2023-05-30的需求量
future_dates = pd.date_range(start='2023-05-16', end='2023-05-30')
predictions = []

for name, group in data_grouped:
    group_data = group[group['date'] < '2023-05-16']['qty_scaled'].values
    if len(group_data) >= seq_length: #若该组数据长度足够用于预测。
        input_seq = group_data[-seq_length:]
        input_seq = np.reshape(input_seq, (1, seq_length, 1))

        group_predictions = []
        for date in future_dates:
            pred = model.predict(input_seq)
            group_predictions.append([date, scaler.inverse_transform(pred)[0, 0]])
            input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        for date, qty in group_predictions:
            predictions.append([name[0], name[1], name[2], date, qty])

results_future_df = pd.DataFrame(predictions, columns=['seller_no', 'product_no', 'warehouse_no', 'date', 'predicted_qty'])
print(results_future_df.head())
results_future_df.to_csv('预测结果.csv', index=False)