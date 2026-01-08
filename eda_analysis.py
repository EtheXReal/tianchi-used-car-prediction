"""
二手车价格预测数据集 - 探索性数据分析（EDA）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

print("=" * 80)
print("1. 数据基本信息")
print("=" * 80)

# 读取数据
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"\n训练集大小: {train.shape[0]} 行, {train.shape[1]} 列")
print(f"测试集大小: {test.shape[0]} 行, {test.shape[1]} 列")

print("\n训练集列名:")
print(train.columns.tolist())

print("\n训练集前5行:")
print(train.head())

print("\n训练集数据类型:")
print(train.dtypes)

print("\n" + "=" * 80)
print("2. 缺失值分析")
print("=" * 80)

# 计算缺失值
missing_train = train.isnull().sum()
missing_pct_train = (train.isnull().sum() / len(train) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing_train,
    '缺失比例(%)': missing_pct_train
})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)

if len(missing_df) > 0:
    print("\n有缺失值的字段:")
    print(missing_df)
else:
    print("\n训练集没有缺失值")

# 检查特殊值（如 '-' 等）
print("\n检查可能的特殊缺失值标记:")
for col in train.columns:
    if train[col].dtype == 'object':
        unique_vals = train[col].unique()
        special_vals = [v for v in unique_vals if v in ['-', 'nan', 'null', 'NA', 'N/A', '']]
        if special_vals:
            print(f"  {col}: {special_vals}")

print("\n" + "=" * 80)
print("3. 目标变量 Price 分析")
print("=" * 80)

print("\nPrice 基本统计量:")
print(train['price'].describe())

print(f"\nPrice 偏度: {train['price'].skew():.4f}")
print(f"Price 峰度: {train['price'].kurtosis():.4f}")

# 价格分布
print("\nPrice 分位数分布:")
quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for q in quantiles:
    print(f"  {int(q*100)}%分位数: {train['price'].quantile(q):.0f}")

print("\n" + "=" * 80)
print("4. 数值型特征分布分析")
print("=" * 80)

# 分离数值型和类别型特征
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"\n数值型特征 ({len(numeric_cols)}个): {numeric_cols}")
print(f"\n类别型特征 ({len(categorical_cols)}个): {categorical_cols}")

print("\n数值型特征统计量:")
numeric_stats = train[numeric_cols].describe().T
numeric_stats['skew'] = train[numeric_cols].skew()
numeric_stats['kurtosis'] = train[numeric_cols].kurtosis()
print(numeric_stats.round(4))

print("\n" + "=" * 80)
print("5. 类别型特征分析")
print("=" * 80)

for col in categorical_cols:
    print(f"\n{col} 唯一值数量: {train[col].nunique()}")
    if train[col].nunique() <= 20:
        print(f"值分布:")
        print(train[col].value_counts())
    else:
        print(f"前10个高频值:")
        print(train[col].value_counts().head(10))

print("\n" + "=" * 80)
print("6. 特征与 Price 的相关性分析")
print("=" * 80)

# 计算数值特征与price的相关系数
numeric_features = [col for col in numeric_cols if col != 'price' and col != 'SaleID']
correlations = train[numeric_features + ['price']].corr()['price'].drop('price').sort_values(key=abs, ascending=False)

print("\n数值特征与 Price 的相关系数 (按绝对值排序):")
print(correlations.round(4))

# 强相关特征
print("\n强相关特征 (|r| > 0.3):")
strong_corr = correlations[abs(correlations) > 0.3]
if len(strong_corr) > 0:
    print(strong_corr.round(4))
else:
    print("没有强相关特征")

# 中等相关特征
print("\n中等相关特征 (0.1 < |r| <= 0.3):")
medium_corr = correlations[(abs(correlations) > 0.1) & (abs(correlations) <= 0.3)]
if len(medium_corr) > 0:
    print(medium_corr.round(4))
else:
    print("没有中等相关特征")

print("\n" + "=" * 80)
print("7. 关键特征深入分析")
print("=" * 80)

# 分析一些关键字段
key_features = ['v_0', 'v_3', 'v_8', 'v_12', 'power', 'kilometer', 'regDate', 'creatDate']

for col in key_features:
    if col in train.columns:
        print(f"\n--- {col} ---")
        print(f"唯一值数量: {train[col].nunique()}")
        print(f"基本统计:")
        print(train[col].describe())

        # 检查异常值
        if train[col].dtype in [np.int64, np.float64]:
            Q1 = train[col].quantile(0.25)
            Q3 = train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = train[(train[col] < lower) | (train[col] > upper)][col]
            print(f"异常值数量 (IQR方法): {len(outliers)} ({len(outliers)/len(train)*100:.2f}%)")

print("\n" + "=" * 80)
print("8. 特征唯一值统计")
print("=" * 80)

print("\n各特征唯一值数量:")
unique_counts = train.nunique().sort_values(ascending=False)
print(unique_counts)

print("\n" + "=" * 80)
print("9. 数据质量检查")
print("=" * 80)

# 检查重复行
duplicates = train.duplicated().sum()
print(f"\n重复行数量: {duplicates}")

# 检查ID唯一性
if 'SaleID' in train.columns:
    print(f"SaleID 唯一性检查: {train['SaleID'].nunique() == len(train)}")

# 检查power异常值
if 'power' in train.columns:
    print(f"\npower = 0 的数量: {(train['power'] == 0).sum()}")
    print(f"power > 600 的数量: {(train['power'] > 600).sum()}")

# 检查notRepairedDamage特殊值
if 'notRepairedDamage' in train.columns:
    print(f"\nnotRepairedDamage 值分布:")
    print(train['notRepairedDamage'].value_counts())

print("\n" + "=" * 80)
print("10. 关键发现总结")
print("=" * 80)

print("""
【关键发现总结】

1. 数据规模:
   - 训练集: {} 条记录, {} 个特征
   - 测试集: {} 条记录

2. 目标变量 Price:
   - 均值: {:.0f}, 中位数: {:.0f}
   - 范围: {:.0f} - {:.0f}
   - 呈现右偏分布 (偏度: {:.2f})

3. 缺失值情况:
   - 缺失字段数: {} 个
{}

4. 与Price强相关的特征 (|r| > 0.3):
{}

5. 异常值/数据质量问题:
   - power字段存在异常值 (0值和超大值)
   - notRepairedDamage 存在 '-' 值需要处理

6. 建议:
   - 对Price进行log变换处理右偏分布
   - 处理power异常值
   - 处理notRepairedDamage的'-'值
   - 考虑基于v系列特征构建新特征
""".format(
    len(train), train.shape[1],
    len(test),
    train['price'].mean(), train['price'].median(),
    train['price'].min(), train['price'].max(),
    train['price'].skew(),
    len(missing_df),
    missing_df.to_string() if len(missing_df) > 0 else "   无缺失值",
    strong_corr.to_string() if len(strong_corr) > 0 else "   无强相关特征"
))

print("\nEDA分析完成!")
