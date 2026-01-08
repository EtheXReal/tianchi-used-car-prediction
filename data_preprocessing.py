"""
二手车价格预测 - 数据预处理脚本
基于EDA发现进行数据清洗和优化
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def reduce_memory_usage(df, verbose=True):
    """优化DataFrame内存占用"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # 整数类型优化
            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            # 浮点类型优化
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"内存占用: {start_mem:.2f} MB -> {end_mem:.2f} MB (减少 {100*(start_mem-end_mem)/start_mem:.1f}%)")

    return df


def main():
    print("=" * 70)
    print("数据预处理开始")
    print("=" * 70)

    # ========== 1. 读取数据 ==========
    print("\n[1/6] 读取数据...")
    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

    print(f"训练集: {train.shape[0]} 行, {train.shape[1]} 列")
    print(f"测试集: {test.shape[0]} 行, {test.shape[1]} 列")

    # ========== 2. 合并数据集 ==========
    print("\n[2/6] 合并训练集和测试集...")

    # 添加标记列区分训练集和测试集
    train['is_train'] = 1
    test['is_train'] = 0

    # 合并
    data = pd.concat([train, test], axis=0, ignore_index=True)
    print(f"合并后数据: {data.shape[0]} 行, {data.shape[1]} 列")

    # ========== 3. 处理 notRepairedDamage ==========
    print("\n[3/6] 处理 notRepairedDamage 的 '-' 值...")

    # 统计处理前的分布
    print("处理前分布:")
    print(data['notRepairedDamage'].value_counts())

    # 将 '-' 替换为 NaN，然后转为数值类型
    data['notRepairedDamage'] = data['notRepairedDamage'].replace('-', np.nan)
    data['notRepairedDamage'] = data['notRepairedDamage'].astype(float)

    print("\n处理后分布:")
    print(data['notRepairedDamage'].value_counts(dropna=False))
    print(f"缺失值数量: {data['notRepairedDamage'].isnull().sum()}")

    # ========== 4. 处理 power 异常值 ==========
    print("\n[4/6] 处理 power 异常值 (截断处理)...")

    # 统计处理前的情况
    print(f"处理前 power 统计:")
    print(f"  - power = 0 的数量: {(data['power'] == 0).sum()}")
    print(f"  - power > 600 的数量: {(data['power'] > 600).sum()}")
    print(f"  - power 范围: [{data['power'].min()}, {data['power'].max()}]")

    # 截断策略：
    # 1. power = 0 视为缺失，用中位数填充
    # 2. power > 600 截断为 600

    power_median = data.loc[data['power'] > 0, 'power'].median()
    print(f"  - 正常power中位数: {power_median}")

    # 处理 power = 0
    data.loc[data['power'] == 0, 'power'] = power_median

    # 截断 power > 600
    data.loc[data['power'] > 600, 'power'] = 600

    print(f"\n处理后 power 统计:")
    print(f"  - power 范围: [{data['power'].min()}, {data['power'].max()}]")
    print(f"  - power 均值: {data['power'].mean():.2f}")
    print(f"  - power 中位数: {data['power'].median():.2f}")

    # ========== 5. 其他数据清洗 ==========
    print("\n[5/6] 其他数据清洗...")

    # 删除无用特征 (seller 和 offerType 几乎无变化)
    # 注意：保留这些列但标记，让用户自行决定是否删除
    print(f"seller 唯一值: {data['seller'].nunique()}")
    print(f"offerType 唯一值: {data['offerType'].nunique()}")

    # 检查处理后的缺失值情况
    print("\n各字段缺失值统计:")
    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    # ========== 6. 优化数据类型 ==========
    print("\n[6/6] 优化数据类型以减少内存占用...")

    # 分离 is_train 列，避免被优化影响
    is_train = data['is_train'].copy()

    # 优化内存
    data = reduce_memory_usage(data)

    # 恢复 is_train 列
    data['is_train'] = is_train.astype(np.uint8)

    # ========== 输出数据 ==========
    print("\n" + "=" * 70)
    print("输出处理后的数据")
    print("=" * 70)

    # 分离训练集和测试集
    train_clean = data[data['is_train'] == 1].drop('is_train', axis=1)
    test_clean = data[data['is_train'] == 0].drop('is_train', axis=1)

    # 保存
    train_clean.to_csv('train_data_clean.csv', index=False)
    test_clean.to_csv('test_data_clean.csv', index=False)

    # 也保存合并后的完整数据（便于后续特征工程）
    data.to_csv('all_data_clean.csv', index=False)

    print(f"\n已保存文件:")
    print(f"  - train_data_clean.csv: {train_clean.shape[0]} 行, {train_clean.shape[1]} 列")
    print(f"  - test_data_clean.csv: {test_clean.shape[0]} 行, {test_clean.shape[1]} 列")
    print(f"  - all_data_clean.csv: {data.shape[0]} 行, {data.shape[1]} 列")

    # ========== 验证输出 ==========
    print("\n" + "=" * 70)
    print("数据验证")
    print("=" * 70)

    # 重新读取验证
    train_verify = pd.read_csv('train_data_clean.csv')
    print(f"\n验证 train_data_clean.csv:")
    print(f"  - 形状: {train_verify.shape}")
    print(f"  - 内存占用: {train_verify.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n数据类型:")
    print(train_verify.dtypes)
    print(f"\npower 范围: [{train_verify['power'].min()}, {train_verify['power'].max()}]")
    print(f"notRepairedDamage 分布:")
    print(train_verify['notRepairedDamage'].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("数据预处理完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
