"""
二手车价格预测 - CatBoost训练脚本
5折交叉验证，log1p目标变换
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# 配置
RANDOM_STATE = 42
N_FOLDS = 5
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """加载数据"""
    print("=" * 60)
    print("加载数据...")

    data = pd.read_csv('data_with_advanced_features.csv')
    train = data[data['is_train'] == 1].drop('is_train', axis=1).reset_index(drop=True)
    test = data[data['is_train'] == 0].drop('is_train', axis=1).reset_index(drop=True)

    print(f"训练集: {train.shape}, 测试集: {test.shape}")
    return train, test


def get_feature_lists(train):
    """获取特征列表"""
    target_col = 'price'
    id_col = 'SaleID'

    cat_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox',
                    'notRepairedDamage', 'regionCode', 'reg_month', 'reg_weekday',
                    'reg_quarter', 'creat_month', 'creat_weekday', 'power_bin',
                    'km_bin', 'name_count_bin', 'reg_year']
    cat_features = [f for f in cat_features if f in train.columns]
    feature_cols = [c for c in train.columns if c not in [target_col, id_col]]

    print(f"特征数: {len(feature_cols)}, 类别特征数: {len(cat_features)}")
    return feature_cols, cat_features, target_col, id_col


def train_catboost_cv(train, test, feature_cols, cat_features, target_col, id_col):
    """5折交叉验证训练"""

    print("\n" + "=" * 60)
    print("开始5折交叉验证训练 (CatBoost)")
    print("=" * 60)

    X = train[feature_cols].copy()
    y = train[target_col].values
    X_test = test[feature_cols].copy()
    test_ids = test[id_col].values

    # log1p变换
    y_log = np.log1p(y)
    print(f"目标变换: log1p, 范围 [{y.min():.0f}, {y.max():.0f}] -> [{y_log.min():.2f}, {y_log.max():.2f}]")

    # 类别特征转字符串
    for col in cat_features:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    cat_indices = [feature_cols.index(f) for f in cat_features]

    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []
    models = []

    # 原始参数 - 保持精度
    params = {
        'iterations': 10000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 10,
        'min_data_in_leaf': 50,
        'random_strength': 0.5,
        'bagging_temperature': 0.2,
        'border_count': 254,
        'random_seed': RANDOM_STATE,
        'task_type': 'CPU',
        'verbose': 500,  # 每500轮显示进度
        'early_stopping_rounds': 200,
        'loss_function': 'RMSE',
        'eval_metric': 'MAE'
    }

    print(f"\n参数: iterations={params['iterations']}, lr={params['learning_rate']}, depth={params['depth']}")
    print(f"早停: {params['early_stopping_rounds']} 轮无改进则停止")
    print("-" * 60)

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*60}")
        print(f"[Fold {fold+1}/{N_FOLDS}] 开始训练... (train: {len(train_idx)}, valid: {len(valid_idx)})")
        print(f"{'='*60}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_log[train_idx], y_log[valid_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)

        # 训练
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        # 预测
        valid_pred_log = model.predict(X_valid)
        test_pred_log = model.predict(X_test)

        valid_pred = np.clip(np.expm1(valid_pred_log), 0, None)
        y_valid_orig = np.expm1(y_valid)
        fold_mae = mean_absolute_error(y_valid_orig, valid_pred)
        fold_scores.append(fold_mae)

        print(f"\n>>> Fold {fold+1} 完成! MAE: {fold_mae:.2f}, Best iteration: {model.get_best_iteration()}")

        oof_pred[valid_idx] = valid_pred
        test_pred += test_pred_log / N_FOLDS

        # 保存模型
        model.save_model(f'{MODEL_DIR}/catboost_fold{fold+1}.cbm')
        models.append(model)

    test_pred_final = np.clip(np.expm1(test_pred), 0, None)
    oof_mae = mean_absolute_error(y, oof_pred)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"各折MAE: {[f'{s:.2f}' for s in fold_scores]}")
    print(f"平均MAE: {np.mean(fold_scores):.2f} (+/- {np.std(fold_scores):.2f})")
    print(f"OOF MAE: {oof_mae:.2f}")

    return oof_pred, test_pred_final, models, fold_scores, test_ids


def save_results(train, test, oof_pred, test_pred, test_ids, fold_scores, models, feature_cols):
    """保存结果"""

    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)

    # OOF预测
    pd.DataFrame({
        'SaleID': train['SaleID'],
        'price_true': train['price'],
        'price_pred': oof_pred
    }).to_csv('oof_predictions.csv', index=False)

    # 测试集预测
    pd.DataFrame({
        'SaleID': test_ids,
        'price': test_pred
    }).to_csv('test_predictions.csv', index=False)

    # 提交文件
    pd.DataFrame({
        'SaleID': test_ids.astype(int),
        'price': test_pred
    }).to_csv('submission.csv', index=False)

    # 特征重要性
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.get_feature_importance() / len(models)

    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    imp_df.to_csv(f'{MODEL_DIR}/feature_importance.csv', index=False)

    print("已保存:")
    print("  - oof_predictions.csv")
    print("  - test_predictions.csv")
    print("  - submission.csv")
    print("  - models/catboost_fold*.cbm")
    print("  - models/feature_importance.csv")

    # Top 10特征
    print("\nTop 10 特征重要性:")
    print(imp_df.head(10).to_string(index=False))

    return imp_df


def main():
    print("\n" + "=" * 60)
    print("CatBoost 5折交叉验证训练")
    print("=" * 60)

    train, test = load_data()
    feature_cols, cat_features, target_col, id_col = get_feature_lists(train)

    oof_pred, test_pred, models, fold_scores, test_ids = train_catboost_cv(
        train, test, feature_cols, cat_features, target_col, id_col
    )

    imp_df = save_results(train, test, oof_pred, test_pred, test_ids, fold_scores, models, feature_cols)

    print("\n" + "=" * 60)
    print(f"最终结果: 平均MAE = {np.mean(fold_scores):.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
