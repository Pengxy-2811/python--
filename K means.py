import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.rcParams['font.sans-serif']=['SimHei']

file_path = "F:\\pythonProject程序设计\\python大作业\\数据\\merged_data.xlsx"
df = pd.read_excel(file_path)
state = df.groupby(['seller_no', 'product_no', 'warehouse_no'])['qty'].describe()
print(state)
state = state[['mean', 'std']]
x = state.values

ls = []
# 整数表示聚类的数量，计算每个聚类数对应的轮廓系数均值，画图找合适的聚类数
for n_clusters in range(2, 100):
    # 使用KMeans算法进行聚类，设置n_init参数（表示算法运行了多少轮，可调整）
    cluster = KMeans(n_clusters=n_clusters, random_state=10, n_init=10).fit(x)
    # 获取每个数据点的聚类标签
    y_pred = cluster.labels_
    # 计算轮廓系数（silhouette_score）均值
    silhouette_avg = silhouette_score(x, y_pred)
    # 将轮廓系数均值添加到ls列表中
    ls.append(silhouette_avg)
# 创建一个新的DataFrame存储轮廓系数
kmean_result = pd.DataFrame(ls, columns=['轮廓系数'])

# 绘制轮廓系数随聚类数变化的折线图
kmean_result['轮廓系数'].plot(kind="line", figsize=(9, 6), color='red', linewidth=2)
plt.xlabel("聚类数", fontsize=14)
plt.ylabel("轮廓系数", fontsize=14)
plt.title("轮廓系数随聚类数的变化", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

#根据图决定用30聚类数进行聚类
cluster = KMeans(n_clusters=30,random_state=10).fit(x)
y_pred = cluster.labels_
state['kmeans'] = y_pred

# 创建一个字典dic，将每个样本的索引与其对应的聚类标签关联起来。
dic = {}
for index,value in zip(state.index,state['kmeans'].values):
    dic[index] = value
# 通过遍历原始数据df中的seller_no、product_no和warehouse_no列，将每个样本的聚类标签添加到新的cluster列中。
ls = []
for i,o,p in zip(df['seller_no'].values,df['product_no'].values,df['warehouse_no'].values):
    ls.append(dic[(i,o,p)])
df['cluster'] = ls
print(df)

train = df.groupby(['date','cluster'])['qty'].mean() # 按日期和cluster分组并计算qty平均值
train = pd.DataFrame(train)
train['d'] = [i[0] for i in train.index]
train['c'] = [i[1] for i in train.index]
train = train.reset_index(drop=True)
train.columns = ['qty','date','cluster']
print(train)

output_file_path = "F:\\pythonProject程序设计\\python大作业\\数据\\聚类结果.xlsx"
train.to_excel(output_file_path, index=False)