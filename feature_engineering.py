"""
二手车价格预测 - 特征工程脚本
基于EDA发现构建高级特征
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


def parse_date(date_int):
    """将整数日期转换为datetime"""
    try:
        date_str = str(int(date_int))
        if len(date_str) == 8:
            return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
    except:
        pass
    return pd.NaT


def create_time_features(df):
    """创建时间相关特征"""
    print("  创建时间特征...")

    # 解析日期
    df['regDate_dt'] = df['regDate'].apply(parse_date)
    df['creatDate_dt'] = df['creatDate'].apply(parse_date)

    # 从regDate提取特征
    df['reg_year'] = df['regDate_dt'].dt.year
    df['reg_month'] = df['regDate_dt'].dt.month
    df['reg_day'] = df['regDate_dt'].dt.day
    df['reg_weekday'] = df['regDate_dt'].dt.weekday
    df['reg_quarter'] = df['regDate_dt'].dt.quarter

    # 从creatDate提取特征
    df['creat_year'] = df['creatDate_dt'].dt.year
    df['creat_month'] = df['creatDate_dt'].dt.month
    df['creat_day'] = df['creatDate_dt'].dt.day
    df['creat_weekday'] = df['creatDate_dt'].dt.weekday

    # 车辆使用年限 (关键特征)
    df['car_age_days'] = (df['creatDate_dt'] - df['regDate_dt']).dt.days
    df['car_age_years'] = df['car_age_days'] / 365.25

    # 是否周末注册
    df['reg_is_weekend'] = (df['reg_weekday'] >= 5).astype(int)

    # 删除临时日期列
    df.drop(['regDate_dt', 'creatDate_dt'], axis=1, inplace=True)

    time_features = ['reg_year', 'reg_month', 'reg_day', 'reg_weekday', 'reg_quarter',
                     'creat_year', 'creat_month', 'creat_day', 'creat_weekday',
                     'car_age_days', 'car_age_years', 'reg_is_weekend']

    print(f"    生成 {len(time_features)} 个时间特征")
    return df, time_features


def create_v_series_features(df):
    """创建V系列统计特征"""
    print("  创建V系列统计特征...")

    v_cols = [f'v_{i}' for i in range(15)]

    # V系列统计特征
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)
    df['v_median'] = df[v_cols].median(axis=1)
    df['v_range'] = df['v_max'] - df['v_min']
    df['v_sum'] = df[v_cols].sum(axis=1)

    # V系列交互特征 (基于EDA发现的强相关特征)
    df['v3_v12_interaction'] = df['v_3'] * df['v_12']
    df['v0_v12_interaction'] = df['v_0'] * df['v_12']
    df['v0_v8_interaction'] = df['v_0'] * df['v_8']
    df['v3_v8_interaction'] = df['v_3'] * df['v_8']

    # v_0相关特征 (强相关)
    df['v0_squared'] = df['v_0'] ** 2
    df['v0_v3_ratio'] = df['v_0'] / (df['v_3'] + 1e-5)

    # v_3相关特征 (最强相关)
    df['v3_squared'] = df['v_3'] ** 2
    df['v3_abs'] = df['v_3'].abs()

    v_features = ['v_mean', 'v_std', 'v_max', 'v_min', 'v_median', 'v_range', 'v_sum',
                  'v3_v12_interaction', 'v0_v12_interaction', 'v0_v8_interaction',
                  'v3_v8_interaction', 'v0_squared', 'v0_v3_ratio', 'v3_squared', 'v3_abs']

    print(f"    生成 {len(v_features)} 个V系列特征")
    return df, v_features


def create_category_cross_features(df):
    """创建类别交叉特征"""
    print("  创建类别交叉特征...")

    # Brand + Model 组合
    df['brand_model'] = df['brand'].astype(str) + '_' + df['model'].astype(str)

    # Brand + BodyType 组合
    df['brand_bodyType'] = df['brand'].astype(str) + '_' + df['bodyType'].astype(str)

    # Brand + FuelType 组合
    df['brand_fuelType'] = df['brand'].astype(str) + '_' + df['fuelType'].astype(str)

    # Brand + Gearbox 组合
    df['brand_gearbox'] = df['brand'].astype(str) + '_' + df['gearbox'].astype(str)

    # Model + BodyType 组合
    df['model_bodyType'] = df['model'].astype(str) + '_' + df['bodyType'].astype(str)

    cross_features = ['brand_model', 'brand_bodyType', 'brand_fuelType',
                      'brand_gearbox', 'model_bodyType']

    print(f"    生成 {len(cross_features)} 个类别交叉特征")
    return df, cross_features


def create_target_encoding_features(df, train_mask, target_col='price'):
    """
    创建基于训练集的统计编码特征
    注意：所有统计量仅基于训练集计算，防止数据泄露
    """
    print("  创建统计编码特征 (基于训练集计算)...")

    # 只使用训练集数据计算统计量
    train_data = df[train_mask].copy()

    encoding_features = []

    # 基于brand的price统计编码
    print("    计算 brand 统计编码...")
    brand_stats = train_data.groupby('brand')[target_col].agg(['mean', 'median', 'std', 'count'])
    brand_stats.columns = ['brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']

    # 全局统计量用于填充未见过的brand
    global_mean = train_data[target_col].mean()
    global_median = train_data[target_col].median()
    global_std = train_data[target_col].std()

    # 合并到原数据
    df = df.merge(brand_stats, on='brand', how='left')
    df['brand_price_mean'].fillna(global_mean, inplace=True)
    df['brand_price_median'].fillna(global_median, inplace=True)
    df['brand_price_std'].fillna(global_std, inplace=True)
    df['brand_count'].fillna(1, inplace=True)

    encoding_features.extend(['brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count'])

    # 基于model的price统计编码
    print("    计算 model 统计编码...")
    model_stats = train_data.groupby('model')[target_col].agg(['mean', 'median', 'std', 'count'])
    model_stats.columns = ['model_price_mean', 'model_price_median', 'model_price_std', 'model_count']

    df = df.merge(model_stats, on='model', how='left')
    df['model_price_mean'].fillna(global_mean, inplace=True)
    df['model_price_median'].fillna(global_median, inplace=True)
    df['model_price_std'].fillna(global_std, inplace=True)
    df['model_count'].fillna(1, inplace=True)

    encoding_features.extend(['model_price_mean', 'model_price_median', 'model_price_std', 'model_count'])

    # 基于bodyType的price统计编码
    print("    计算 bodyType 统计编码...")
    bodyType_stats = train_data.groupby('bodyType')[target_col].agg(['mean', 'median'])
    bodyType_stats.columns = ['bodyType_price_mean', 'bodyType_price_median']

    df = df.merge(bodyType_stats, on='bodyType', how='left')
    df['bodyType_price_mean'].fillna(global_mean, inplace=True)
    df['bodyType_price_median'].fillna(global_median, inplace=True)

    encoding_features.extend(['bodyType_price_mean', 'bodyType_price_median'])

    # 基于regionCode的price统计编码
    print("    计算 regionCode 统计编码...")
    region_stats = train_data.groupby('regionCode')[target_col].agg(['mean', 'count'])
    region_stats.columns = ['region_price_mean', 'region_count']

    df = df.merge(region_stats, on='regionCode', how='left')
    df['region_price_mean'].fillna(global_mean, inplace=True)
    df['region_count'].fillna(1, inplace=True)

    encoding_features.extend(['region_price_mean', 'region_count'])

    # 基于brand_model组合的统计编码
    print("    计算 brand_model 统计编码...")
    brand_model_stats = train_data.groupby('brand_model')[target_col].agg(['mean', 'count'])
    brand_model_stats.columns = ['brand_model_price_mean', 'brand_model_count']

    df = df.merge(brand_model_stats, on='brand_model', how='left')
    df['brand_model_price_mean'].fillna(global_mean, inplace=True)
    df['brand_model_count'].fillna(1, inplace=True)

    encoding_features.extend(['brand_model_price_mean', 'brand_model_count'])

    print(f"    生成 {len(encoding_features)} 个统计编码特征")
    return df, encoding_features


def create_other_features(df):
    """创建其他衍生特征"""
    print("  创建其他衍生特征...")

    other_features = []

    # power相关特征
    df['power_per_age'] = df['power'] / (df['car_age_years'] + 0.1)
    df['power_bin'] = pd.cut(df['power'], bins=[0, 75, 110, 150, 200, 600],
                              labels=[0, 1, 2, 3, 4]).astype(float)
    other_features.extend(['power_per_age', 'power_bin'])

    # kilometer相关特征
    df['km_per_year'] = df['kilometer'] / (df['car_age_years'] + 0.1)
    df['km_bin'] = pd.cut(df['kilometer'], bins=[0, 3, 6, 10, 12.5, 15],
                           labels=[0, 1, 2, 3, 4]).astype(float)
    other_features.extend(['km_per_year', 'km_bin'])

    # 价格预估特征 (基于关键特征)
    df['value_score'] = df['v_0'] * 0.3 + df['v_8'] * 0.3 + df['v_12'] * 0.3 - df['v_3'] * 0.3
    other_features.append('value_score')

    # name 特征 (可能代表车型热度)
    df['name_count_bin'] = pd.qcut(df['name'], q=10, labels=False, duplicates='drop')
    other_features.append('name_count_bin')

    print(f"    生成 {len(other_features)} 个其他特征")
    return df, other_features


def fill_missing_values(df, train_mask):
    """
    填充缺失值
    数值型: median (基于训练集)
    类别型: mode (基于训练集)
    """
    print("  填充缺失值...")

    train_data = df[train_mask]

    # 数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 类别型列 (object类型)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 用训练集的median填充数值型
    filled_numeric = 0
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = train_data[col].median()
            df[col].fillna(median_val, inplace=True)
            filled_numeric += 1

    # 用训练集的mode填充类别型
    filled_cat = 0
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = train_data[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
                filled_cat += 1

    print(f"    填充 {filled_numeric} 个数值列, {filled_cat} 个类别列")

    # 验证
    remaining_missing = df.isnull().sum().sum()
    print(f"    剩余缺失值: {remaining_missing}")

    return df


def select_features(df, time_features, v_features, cross_features, encoding_features, other_features):
    """选择最终特征集"""
    print("  选择最终特征...")

    # 原始重要特征
    original_features = ['SaleID', 'brand', 'model', 'bodyType', 'fuelType', 'gearbox',
                         'power', 'kilometer', 'notRepairedDamage', 'regionCode',
                         'v_0', 'v_3', 'v_8', 'v_12', 'regDate', 'creatDate']

    # V系列其他特征 (保留部分)
    v_keep = ['v_1', 'v_2', 'v_4', 'v_5', 'v_6', 'v_7', 'v_9', 'v_10', 'v_11', 'v_13', 'v_14']

    # 合并所有特征
    all_features = (original_features + v_keep + time_features +
                    v_features + encoding_features + other_features)

    # 添加目标变量
    if 'price' in df.columns:
        all_features.append('price')

    # 去重
    all_features = list(dict.fromkeys(all_features))

    # 过滤存在的列
    final_features = [f for f in all_features if f in df.columns]

    # 不保留交叉特征的字符串形式（用于编码后删除）
    # cross_features 已经用于计算统计编码，可以删除
    for cf in cross_features:
        if cf in final_features:
            final_features.remove(cf)

    print(f"    最终选择 {len(final_features)} 个特征")
    return final_features


def main():
    print("=" * 70)
    print("特征工程开始")
    print("=" * 70)

    # ========== 1. 读取数据 ==========
    print("\n[1/7] 读取清洗后的数据...")

    # 读取合并后的数据 (包含训练集和测试集)
    data = pd.read_csv('all_data_clean.csv')
    print(f"数据规模: {data.shape[0]} 行, {data.shape[1]} 列")

    # 创建训练集标记
    train_mask = data['is_train'] == 1
    print(f"训练集: {train_mask.sum()} 行, 测试集: {(~train_mask).sum()} 行")

    # ========== 2. 时间特征 ==========
    print("\n[2/7] 创建时间特征...")
    data, time_features = create_time_features(data)

    # ========== 3. V系列统计特征 ==========
    print("\n[3/7] 创建V系列统计特征...")
    data, v_features = create_v_series_features(data)

    # ========== 4. 类别交叉特征 ==========
    print("\n[4/7] 创建类别交叉特征...")
    data, cross_features = create_category_cross_features(data)

    # ========== 5. 统计编码特征 ==========
    print("\n[5/7] 创建统计编码特征...")
    data, encoding_features = create_target_encoding_features(data, train_mask)

    # ========== 6. 其他衍生特征 ==========
    print("\n[6/7] 创建其他衍生特征...")
    data, other_features = create_other_features(data)

    # ========== 7. 缺失值处理 ==========
    print("\n[7/7] 填充缺失值...")
    data = fill_missing_values(data, train_mask)

    # ========== 选择特征并保存 ==========
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    # 选择最终特征
    final_features = select_features(data, time_features, v_features,
                                      cross_features, encoding_features, other_features)

    # 添加is_train标记
    if 'is_train' not in final_features:
        final_features.append('is_train')

    # 提取最终数据
    data_final = data[final_features].copy()

    # 保存完整数据
    data_final.to_csv('data_with_advanced_features.csv', index=False)
    print(f"\n已保存: data_with_advanced_features.csv")
    print(f"  - 形状: {data_final.shape}")

    # 分离训练集和测试集保存
    train_final = data_final[data_final['is_train'] == 1].drop('is_train', axis=1)
    test_final = data_final[data_final['is_train'] == 0].drop('is_train', axis=1)

    train_final.to_csv('train_features.csv', index=False)
    test_final.to_csv('test_features.csv', index=False)

    print(f"\n已保存: train_features.csv")
    print(f"  - 形状: {train_final.shape}")
    print(f"\n已保存: test_features.csv")
    print(f"  - 形状: {test_final.shape}")

    # ========== 输出特征列表 ==========
    print("\n" + "=" * 70)
    print("特征列表总结")
    print("=" * 70)

    print(f"\n时间特征 ({len(time_features)}):")
    print(f"  {time_features}")

    print(f"\nV系列统计特征 ({len(v_features)}):")
    print(f"  {v_features}")

    print(f"\n统计编码特征 ({len(encoding_features)}):")
    print(f"  {encoding_features}")

    print(f"\n其他衍生特征 ({len(other_features)}):")
    print(f"  {other_features}")

    print(f"\n最终特征数量: {len(final_features) - 1} (不含is_train)")  # -1 for is_train

    # ========== 验证 ==========
    print("\n" + "=" * 70)
    print("数据验证")
    print("=" * 70)

    print(f"\n缺失值检查:")
    missing = data_final.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("  无缺失值")

    print(f"\n数据类型统计:")
    print(data_final.dtypes.value_counts())

    print(f"\n训练集price统计:")
    print(train_final['price'].describe())

    # 新特征与price的相关性
    print(f"\n新特征与price的相关性 (Top 20):")
    new_features = time_features + v_features + encoding_features + other_features
    new_features = [f for f in new_features if f in train_final.columns]

    correlations = train_final[new_features + ['price']].corr()['price'].drop('price')
    correlations = correlations.sort_values(key=abs, ascending=False).head(20)
    print(correlations.round(4))

    print("\n" + "=" * 70)
    print("特征工程完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
