from collections.abc import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from lightgbm import LGBMClassifier

class PurgedTimeSeriesSplit:
    def __init__(self, dates, n_splits=5, gap=0, window_size=None):
        self.dates = pd.to_datetime(dates).reset_index(drop=True)
        self.n_splits = n_splits
        self.gap = pd.Timedelta(days=gap) if isinstance(gap, int) else gap
        self.window_size = (
            pd.Timedelta(days=window_size) if isinstance(window_size, int)
            else window_size
        )

        self.unique_dates = pd.Index(self.dates.unique()).sort_values()

    def split(self, X, y=None, groups=None):

        U = self.unique_dates
        cut_points = np.linspace(
            0, len(U), self.n_splits + 2, dtype=int
        )[1:-1]

        for i, cp in enumerate(cut_points):

            train_cutoff = U[cp]

            if self.window_size is not None:
                train_start = train_cutoff - self.window_size
            else:
                train_start = U[0]

            test_start = train_cutoff + self.gap

            if i + 1 < len(cut_points):
                test_end = U[cut_points[i + 1]]
            else:
                test_end = U[-1]

            tr = self.dates[
                (self.dates >= train_start) &
                (self.dates <= train_cutoff)
            ].index.values

            te = self.dates[
                (self.dates >= test_start) &
                (self.dates <= test_end)
            ].index.values

            if len(tr) and len(te):
                yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def rolling_window_classifier_consistency_check(
    df,
    target_column,
    time_column,
    window_size=3,
    step_size=1,
    threshold=0.5, # Default AUC threshold usually 0.5 (random)
    weight_scheme='exponential',
    weight_param=0.2,
    *,
    time_weight_map=None,
    normalize_time_weights=False
):
    """
    Check time consistency for each feature using a rolling window approach
    using Logistic Regression and ROC AUC.
    """
    feature_columns = [c for c in df.columns if c not in [target_column, time_column]]
    df_sorted = df.sort_values(by=time_column)
    time_points = df_sorted[time_column].unique()

    def _calc_weights_from_scheme(time_values, scheme='exponential', param=0.2):
        unique_times = np.unique(time_values)
        time_to_index = {t: i for i, t in enumerate(unique_times)}
        idx = np.array([time_to_index[t] for t in time_values])

        if scheme == 'none':
            w = np.ones(len(time_values), dtype=float)
        elif scheme == 'linear':
            w = 1.0 + idx * float(param)
        elif scheme == 'exponential':
            w = np.exp(idx * float(param))
        else:
            raise ValueError(f"Unknown weight_scheme: {scheme}")
        return w

    def _calc_weights_from_map(time_values, twm):
        if isinstance(twm, (dict, pd.Series)):
            w = np.array([float(twm.get(t, 1.0)) for t in time_values], dtype=float)
        elif isinstance(twm, Callable):
            w = np.array([float(twm(t)) for t in time_values], dtype=float)
        else:
            raise TypeError("time_weight_map must be dict, pandas Series, or callable")
        return w

    results = []

    for feature in tqdm(feature_columns, desc="Checking features"):
        feature_results = []

        for i in range(0, len(time_points) - window_size, step_size):
            if i + window_size >= len(time_points):
                break

            train_time = time_points[i : i + window_size]
            val_time = time_points[i + window_size]

            train_data = df_sorted[df_sorted[time_column].isin(train_time)]
            X_train = train_data[[feature]]
            y_train = train_data[target_column].to_numpy()

            # Skip if only one class is present in training
            if len(np.unique(y_train)) < 2:
                continue

            # --- sample weights ---
            if time_weight_map is not None:
                sample_weights = _calc_weights_from_map(train_data[time_column].values, time_weight_map)
            else:
                sample_weights = _calc_weights_from_scheme(train_data[time_column].values, weight_scheme, weight_param)

            if normalize_time_weights and sample_weights.size > 0:
                m = sample_weights.mean()
                if m > 0:
                    sample_weights = sample_weights / m

            val_data = df_sorted[df_sorted[time_column] == val_time]
            X_val = val_data[[feature]]
            y_val = val_data[target_column].to_numpy()

            # Fit Logistic Regression
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Train ROC AUC
            y_prob_train = model.predict_proba(X_train)[:, 1]
            try:
                train_auc = roc_auc_score(y_train, y_prob_train, sample_weight=sample_weights)
            except ValueError:
                train_auc = np.nan

            # Validation ROC AUC
            y_prob_val = model.predict_proba(X_val)[:, 1]
            try:
                # Validation is usually unweighted unless specified
                val_auc = roc_auc_score(y_val, y_prob_val) if len(np.unique(y_val)) > 1 else np.nan
            except ValueError:
                val_auc = np.nan

            feature_results.append({
                'Feature': feature,
                'Train_Time': train_time,
                'Val_Time': val_time,
                'Train_AUC': train_auc,
                'Val_AUC': val_auc
            })

        # Summaries
        if feature_results:
            val_auc_scores = [res['Val_AUC'] for res in feature_results if np.isfinite(res['Val_AUC'])]
            if len(val_auc_scores) > 0:
                mean_val_auc = float(np.mean(val_auc_scores))
                std_val_auc  = float(np.std(val_auc_scores))
                recent_chunk = val_auc_scores[-3:] if len(val_auc_scores) >= 3 else val_auc_scores
                recent_min_auc = float(np.min(recent_chunk))
                max_val_auc = float(np.max(val_auc_scores))

                is_consistent = (mean_val_auc > threshold) and (recent_min_auc > threshold)

                results.append({
                    'Feature': feature,
                    'Mean_Val_AUC': mean_val_auc,
                    'Std_Val_AUC': std_val_auc,
                    'Max_Val_AUC': max_val_auc,
                    'Recent_Min_AUC': recent_min_auc,
                    'Is_Consistent': is_consistent
                })

    return pd.DataFrame(results)


def cvs(X, y, model=None, std=False, return_scores=False):

  if model is None:
    model = LGBMClassifier(verbose=-1)

  X=X.replace({np.inf: np.nan,-np.inf: np.nan})
  cv=PurgedTimeSeriesSplit(dates=pd.Series(X.index.get_level_values('earnings_ts')),gap=121)

  X=X.sort_index(level='earnings_ts')
  y=y.loc[X.index]

  scores=cross_val_score(model,X.fillna(-999),y,scoring='average_precision',error_score='raise',cv=cv)#np.mean(
  print('CV average_precision scores:', scores)

  if return_scores:
    return scores

  if std:
    return scores.mean()/scores.std()

  return scores.mean()


def chronological_split(df, y,val_ratio=0.2, date_level='earnings_ts'):
    """
    Splits a MultiIndex DataFrame chronologically based on the date level.

    Args:
        df (pd.DataFrame): MultiIndex DF (e.g., ['ticker', 'date'])
        val_ratio (float): Proportion of data to use for validation (0 to 1)
        date_level (str): The name of the index level containing dates

    Returns:
        train_df, val_df: Two DataFrames split by date
    """
    # 1. Get unique sorted dates from the index
    unique_dates = df.index.get_level_values(date_level).unique().sort_values()

    # 2. Calculate the split index
    split_idx = int(len(unique_dates) * (1 - val_ratio))
    split_date = unique_dates[split_idx]

    # 3. Slice the dataframe
    # We use xs or index slicing to ensure we don't mix dates
    train_df = df.iloc[df.index.get_level_values(date_level) < split_date]
    val_df = df.iloc[df.index.get_level_values(date_level) >= split_date]

    y_train=y.loc[train_df.index]
    y_val=y.loc[val_df.index]

    return train_df, val_df,y_train,y_val


def meta_cvs(
    X,
    y,
    ds,
    close,
    high,
    low,
    open_,
    earnings_tickers,
    volume=None,
    primary_model=None,
    meta_model=None,
    tp=0.07,
    sl=0.03,
    primary_cvs=False,
    horizon=5,
):
  from .meta_labeling import run_primary_plus_meta


  if primary_model is None:
    primary_model = LGBMClassifier(verbose=-1)

  if meta_model is None:
    meta_model = make_pipeline(SimpleImputer(fill_value=-999), LogisticRegression())

  if primary_cvs:
    primary_scores = cvs(X.fillna(-999), y, primary_model, return_scores=True)
    print('Primary CV mean average_precision:', primary_scores.mean())

  X=X.replace({np.inf: np.nan,-np.inf: np.nan})

  X_meta,y_meta=run_primary_plus_meta(
          X.fillna(-999), y, ds, close,open_,high,low,earnings_tickers,
          px_volume=volume,
          primary_model=primary_model,tp=tp,sl=sl,
          score_mode=True,horizon=horizon
      )

  print(X_meta.shape,y_meta.shape,close.shape)

  X_meta=X_meta.replace({np.inf: np.nan,-np.inf: np.nan})

  score = cvs(X_meta.fillna(-999), y_meta, meta_model,return_scores=True)
  print('Meta CV scores distribution:', score)

  return score


def _fold_imbalance_penalty(y_true, min_fold_pos_rate):
  """Penalize folds where class balance is too extreme."""
  fold_pos_rate = float(np.mean(y_true == 1))
  min_rate = float(min_fold_pos_rate)

  if min_rate <= 0:
    return 1.0

  left = fold_pos_rate / min_rate
  right = (1.0 - fold_pos_rate) / min_rate
  return float(max(0.0, min(1.0, left, right)))


def _cv_recall_skill(model, X, y, cv, min_fold_pos_rate=0.05):
  """Chance-corrected recall using predicted-positive rate as baseline."""
  scores = []

  for tr_idx, te_idx in cv.split(X, y):
    m = clone(model)
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)

    recall = recall_score(y_te, y_pred, zero_division=0)
    pred_pos_rate = float(np.mean(y_pred == 1))

    if pred_pos_rate >= 1.0:
      recall_skill = 0.0
    else:
      recall_skill = (recall - pred_pos_rate) / (1.0 - pred_pos_rate)

    imbalance_penalty = _fold_imbalance_penalty(y_te, min_fold_pos_rate)
    scores.append(np.clip(recall_skill, 0.0, 1.0) * imbalance_penalty)

  return float(np.mean(scores))


def _cv_average_precision_skill(model, X, y, cv, min_fold_pos_rate=0.05):
  """Chance-corrected average precision using prevalence as baseline."""
  scores = []

  for tr_idx, te_idx in cv.split(X, y):
    m = clone(model)
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    m.fit(X_tr, y_tr)
    y_prob = m.predict_proba(X_te)[:, 1]

    pos_rate = float(np.mean(y_te == 1))
    if pos_rate >= 1.0:
      ap_skill = 0.0
    else:
      ap = average_precision_score(y_te, y_prob)
      ap_skill = (ap - pos_rate) / (1.0 - pos_rate)

    imbalance_penalty = _fold_imbalance_penalty(y_te, min_fold_pos_rate)
    scores.append(np.clip(ap_skill, 0.0, 1.0) * imbalance_penalty)

  return float(np.mean(scores))


def meta_cvs_composite(
    X,
    y,
    ds,
    close,
    high,
    low,
    open_,
    earnings_tickers,
    volume=None,
    primary_model=None,
    meta_model=None,
    tp=0.07,
    sl=0.03,
    horizon=5,
    adjust_for_imbalance=True,
    min_fold_pos_rate=0.05,
):
  from .meta_labeling import run_primary_plus_meta

  if primary_model is None:
    primary_model = LGBMClassifier(verbose=-1)

  if meta_model is None:
    meta_model = make_pipeline(SimpleImputer(fill_value=-999), LogisticRegression())

  X = X.replace({np.inf: np.nan, -np.inf: np.nan})

  primary_cv = PurgedTimeSeriesSplit(
      dates=pd.Series(X.index.get_level_values('earnings_ts')),
      gap=121,
  )
  X_primary = X.sort_index(level='earnings_ts')
  y_primary = y.loc[X_primary.index]

  if adjust_for_imbalance:
    primary_score = _cv_recall_skill(
        primary_model,
        X_primary.fillna(-999),
        y_primary,
        primary_cv,
        min_fold_pos_rate=min_fold_pos_rate,
    )
  else:
    primary_score = cross_val_score(
        primary_model,
        X_primary.fillna(-999),
        y_primary,
        scoring='recall',
        error_score='raise',
        cv=primary_cv,
    ).mean()

  X_meta, y_meta = run_primary_plus_meta(
      X.fillna(-999),
      y,
      ds,
      close,
      open_,
      high,
      low,
      earnings_tickers,
      px_volume=volume,
      primary_model=primary_model,
      tp=tp,
      sl=sl,
      score_mode=True,
      horizon=horizon,
  )

  X_meta = X_meta.replace({np.inf: np.nan, -np.inf: np.nan})
  X_meta = X_meta.sort_index(level='earnings_ts')
  y_meta = y_meta.loc[X_meta.index]

  meta_cv = PurgedTimeSeriesSplit(
      dates=pd.Series(X_meta.index.get_level_values('earnings_ts')),
      gap=121,
  )

  if adjust_for_imbalance:
    meta_score = _cv_average_precision_skill(
        meta_model,
        X_meta.fillna(-999),
        y_meta,
        meta_cv,
        min_fold_pos_rate=min_fold_pos_rate,
    )
  else:
    meta_score = cvs(
        X_meta.fillna(-999),
        y_meta,
        meta_model,
    )

  return (primary_score + meta_score) / 2


def cv_predict_proba_purged(model, X, y, cv):
    """Return OOF proba aligned to X.index using provided (purged) CV splits."""
    oof = pd.Series(index=X.index, dtype=float)

    # PurgedTimeSeriesSplit usually yields (train_idx, test_idx) as integer positions
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        #m = model.__class__(**model.get_params())
        model.fit(X_tr, y_tr)
        oof.iloc[te_idx] = model.predict_proba(X_te)[:, 1]

    return oof
