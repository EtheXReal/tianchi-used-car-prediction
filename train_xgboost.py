import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time

warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBoost 最终优化版 - 增强特征工程 + 早停优化")
print("=" * 70)

t0 = time.time()

# =====================================================================
# 1. 数据加载
# =====================================================================
print("\n[1] 加载数据...")
train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
test_ids = test['SaleID'].copy()
print(f"    训练集: {train.shape}, 测试集: {test.shape}")

# =====================================================================
# 2. 日期特征工程 (增强版)
# =====================================================================
print("\n[2] 日期特征工程...")

BASE = pd.to_datetime('1990-01-01')

def parse_date(d):
    try:
        s = str(int(d))
        y = int(s[:4])
        m = int(s[4:6]) if len(s) >= 6 else 1
        day = int(s[6:8]) if len(s) >= 8 else 1
        m = max(1, min(12, m)) if m else 1
        day = max(1, min(28, day)) if day else 1
        return pd.to_datetime(f'{y}-{m:02d}-{day:02d}')
    except:
        return pd.NaT

for df in [train, test]:
    # 解析日期
    df['reg_date'] = df['regDate'].apply(parse_date)
    df['creat_date'] = df['creatDate'].apply(parse_date)

    # 日期diff (距离基准)
    df['reg_diff'] = (df['reg_date'] - BASE).dt.days
    df['creat_diff'] = (df['creat_date'] - BASE).dt.days

    # 车龄
    df['car_age_days'] = (df['creat_date'] - df['reg_date']).dt.days
    df['car_age'] = df['car_age_days'] / 365.0

    # 年月提取
    df['reg_year'] = df['reg_date'].dt.year
    df['reg_month'] = df['reg_date'].dt.month
    df['reg_quarter'] = df['reg_date'].dt.quarter
    df['creat_year'] = df['creat_date'].dt.year
    df['creat_month'] = df['creat_date'].dt.month

    # 是否年初/年末购车
    df['reg_year_start'] = (df['reg_month'] <= 3).astype(int)
    df['reg_year_end'] = (df['reg_month'] >= 10).astype(int)

    # 删除临时列
    df.drop(['regDate', 'creatDate', 'reg_date', 'creat_date'], axis=1, inplace=True)

# =====================================================================
# 3. 缺失值和异常值处理
# =====================================================================
print("\n[3] 缺失值和异常值处理...")

train['notRepairedDamage'] = train['notRepairedDamage'].replace('-', '-1').astype(float)
test['notRepairedDamage'] = test['notRepairedDamage'].replace('-', '-1').astype(float)

for col in ['model', 'bodyType', 'fuelType', 'gearbox']:
    mode = train[col].mode()[0]
    train[col].fillna(mode, inplace=True)
    test[col].fillna(mode, inplace=True)

# power处理
train['power'] = train['power'].clip(0, 600)
test['power'] = test['power'].clip(0, 600)

# power=0 用分组中位数填充
power_by_brand = train[train['power'] > 0].groupby('brand')['power'].median()
overall_power = train[train['power'] > 0]['power'].median()

for df in [train, test]:
    mask = df['power'] == 0
    df.loc[mask, 'power'] = df.loc[mask, 'brand'].map(power_by_brand).fillna(overall_power)

# =====================================================================
# 4. 增强特征工程
# =====================================================================
print("\n[4] 增强特征工程...")

def enhanced_features(df, ref=None):
    df = df.copy()
    if ref is None:
        ref = df

    # ---------- 目标编码特征 ----------
    if 'price' in ref.columns:
        # 单特征统计
        for col in ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode']:
            agg = ref.groupby(col)['price'].agg(['mean', 'median', 'std', 'min', 'max'])
            agg.columns = [f'{col}_p_{x}' for x in ['mean', 'med', 'std', 'min', 'max']]
            df = df.merge(agg, on=col, how='left')

        # 二阶交叉编码
        cross_pairs = [
            ('brand', 'bodyType'),
            ('brand', 'fuelType'),
            ('brand', 'gearbox'),
            ('model', 'bodyType'),
            ('bodyType', 'fuelType'),
        ]
        for c1, c2 in cross_pairs:
            grp = ref.groupby([c1, c2])['price'].mean()
            df[f'{c1}_{c2}_pmean'] = df.apply(
                lambda x: grp.get((x[c1], x[c2]), np.nan), axis=1
            )

        # 车龄×品牌交叉
        age_brand = ref.groupby(['brand', ref['car_age'].round().astype(int)])['price'].mean()
        df['brand_age_pmean'] = df.apply(
            lambda x: age_brand.get((x['brand'], int(round(x['car_age']))), np.nan), axis=1
        )

        # 车龄×车型交叉
        age_model = ref.groupby(['model', ref['car_age'].round().astype(int)])['price'].mean()
        df['model_age_pmean'] = df.apply(
            lambda x: age_model.get((x['model'], int(round(x['car_age']))), np.nan), axis=1
        )

    # ---------- 计数特征 ----------
    for col in ['brand', 'model', 'regionCode']:
        cnt = ref.groupby(col).size().rename(f'{col}_cnt')
        df = df.merge(cnt, on=col, how='left')

    # ---------- 比率特征 ----------
    df['power_per_age'] = df['power'] / (df['car_age'] + 0.1)
    df['km_per_age'] = df['kilometer'] / (df['car_age'] + 0.1)
    df['km_per_day'] = df['kilometer'] * 10000 / (df['car_age_days'] + 1)  # 日均公里
    df['power_per_km'] = df['power'] / (df['kilometer'] + 0.1)
    df['age_km_ratio'] = df['car_age_days'] / (df['kilometer'] + 0.1)

    # ---------- 分箱特征 ----------
    df['power_bin'] = pd.cut(df['power'], bins=[0, 75, 100, 125, 150, 200, 600],
                              labels=[0, 1, 2, 3, 4, 5]).astype(float)
    df['km_bin'] = pd.cut(df['kilometer'], bins=[0, 1, 3, 6, 10, 15],
                           labels=[0, 1, 2, 3, 4]).astype(float)
    df['age_bin'] = pd.cut(df['car_age'], bins=[-1, 1, 2, 4, 6, 10, 30],
                            labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # ---------- 组合编码 ----------
    le = LabelEncoder()
    df['brand_body'] = le.fit_transform(df['brand'].astype(str) + '_' + df['bodyType'].astype(str))
    df['brand_fuel'] = le.fit_transform(df['brand'].astype(str) + '_' + df['fuelType'].astype(str))
    df['brand_gear'] = le.fit_transform(df['brand'].astype(str) + '_' + df['gearbox'].astype(str))
    df['body_fuel'] = le.fit_transform(df['bodyType'].astype(str) + '_' + df['fuelType'].astype(str))
    df['model_body'] = le.fit_transform(df['model'].astype(str) + '_' + df['bodyType'].astype(str))

    # ---------- v特征工程 ----------
    v_cols = [f'v_{i}' for i in range(15)]
    df['v_sum'] = df[v_cols].sum(axis=1)
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)
    df['v_range'] = df['v_max'] - df['v_min']
    df['v_skew'] = df[v_cols].skew(axis=1)

    # 高相关v组合
    df['v_high'] = df['v_0'] + df['v_8'] + df['v_12'] - df['v_3']
    df['v_high2'] = df['v_0'] * df['v_8'] + df['v_12']
    df['v_neg'] = df['v_3'] + df['v_11'] + df['v_10'] + df['v_9']

    # v与其他特征交互
    df['v0_power'] = df['v_0'] * df['power']
    df['v3_age'] = df['v_3'] * df['car_age']
    df['v12_km'] = df['v_12'] * df['kilometer']
    df['v8_power'] = df['v_8'] * df['power']
    df['vhigh_age'] = df['v_high'] / (df['car_age'] + 0.1)

    return df

train = enhanced_features(train)
test = enhanced_features(test, train)

# 填充缺失值
for col in train.columns:
    if train[col].isnull().sum() > 0:
        med = train[col].median()
        train[col].fillna(med, inplace=True)
        test[col].fillna(med, inplace=True)

# 删除无用列
drop_cols = ['SaleID', 'name', 'seller', 'offerType']
train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)

# 准备数据
y = train['price']
X = train.drop('price', axis=1)
y_log = np.log1p(y)
test = test[X.columns]

print(f"    特征数量: {X.shape[1]}")
print(f"    预处理耗时: {time.time() - t0:.1f}秒")

# =====================================================================
# 5. XGBoost训练 (优化参数 + 早停)
# =====================================================================
print("\n[5] XGBoost训练...")

# 最优参数 (基于之前测试)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',  # 直接用MAE作为评估指标
    'max_depth': 7,
    'learning_rate': 0.008,
    'n_estimators': 15000,  # 更多树，依赖早停
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'min_child_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 200  # 早停轮数
}

print(f"    参数: depth={params['max_depth']}, lr={params['learning_rate']}, "
      f"trees={params['n_estimators']}, early_stop={params['early_stopping_rounds']}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_iters = []
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=0
    )

    val_pred = np.expm1(model.predict(X_va))
    val_true = np.expm1(y_va)
    mae = mean_absolute_error(val_true, val_pred)

    scores.append(mae)
    best_iters.append(model.best_iteration)
    test_preds += np.expm1(model.predict(test)) / 5

    print(f"    Fold {fold+1}: MAE={mae:.2f}, BestIter={model.best_iteration}")

cv_mae = np.mean(scores)
cv_std = np.std(scores)
avg_iter = np.mean(best_iters)

print(f"\n    5-Fold CV MAE: {cv_mae:.2f} (+/- {cv_std:.2f})")
print(f"    平均最佳迭代: {avg_iter:.0f}")

# =====================================================================
# 6. 保存结果
# =====================================================================
print("\n[6] 保存结果...")

submission = pd.DataFrame({
    'SaleID': test_ids,
    'price': test_preds.clip(min=0)
})
submission.to_csv('submission_final.csv', index=False)
print(f"    保存: submission_final.csv")

# =====================================================================
# 结果对比
# =====================================================================
print("\n" + "=" * 70)
print("结果对比")
print("=" * 70)
print(f"""
| 版本           | CV MAE  | 提升    |
|----------------|---------|---------|
| V2 基准        | 521.08  | -       |
| V3 特征增强    | 511.25  | -9.83   |
| 最终优化版     | {cv_mae:.2f}  | {511.25 - cv_mae:.2f}   |
""")
print(f"总耗时: {time.time() - t0:.1f}秒")
print("=" * 70)
