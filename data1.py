import pandas as pd

df1 = pd.read_excel("F:\\pythonProject程序设计\\python大作业\\数据\\商家历史出货量表.xlsx")
df2 = pd.read_excel("F:\\pythonProject程序设计\\python大作业\\数据\\商品信息表.xlsx")
df3 = pd.read_excel("F:\\pythonProject程序设计\\python大作业\\数据\\商家信息表.xlsx")
df4 = pd.read_excel("F:\\pythonProject程序设计\\python大作业\\数据\\仓库信息表.xlsx")

# 按照1个商家对应一类商品对应一个仓库合并
df = df1.merge(df3, on='seller_no', how='inner').merge(df2, on='product_no', how='inner').merge(df4, on='warehouse_no',
 how='inner')
columns = ['seller_no', 'product_no', 'warehouse_no', 'category3']
for col in columns:
    df[col] = [int(i.split('_')[1]) if isinstance(i, str) and '_' in i else i for i in df[col].values]

print(df.head())
result = pd.DataFrame()

result = pd.DataFrame()
# 遍历每个商家
for seller in df['seller_no'].unique():
    # 筛选出当前商家的所有数据
    seller_data = df[df.seller_no == seller]
    # 遍历该商家的每个产品
    for product in seller_data['product_no'].unique():
        # 筛选出当前商家当前产品的所有数据，并复制一份
        product_data = seller_data[seller_data.product_no == product].copy()

        for warehouse in product_data['warehouse_no'].unique():
            # 筛选出当前商家当前产品在当前仓库的所有数据
            warehouse_data = product_data[product_data.warehouse_no == warehouse]
            # 按日期分组并汇总数量（qty）
            daily_qty_sum = warehouse_data.groupby('date')['qty'].sum()

            # 截取与汇总结果相同长度的原始数据，确保 warehouse_data 和 daily_qty_sum 的长度相同
            warehouse_data = warehouse_data[:len(daily_qty_sum)]
            warehouse_data['date'] = daily_qty_sum.index   #求和得到每个商家对应每个商品每个仓库对应每天的库存量
            warehouse_data['qty'] = daily_qty_sum.values

            result = pd.concat([result, warehouse_data])

# 将最终的汇总结果赋值回 df
df = result

output_file = 'F:\\pythonProject程序设计\\python大作业\\数据\\merged_data.xlsx'
df.to_excel(output_file, index=False)

print(f"处理后的数据已保存到 {output_file}")
