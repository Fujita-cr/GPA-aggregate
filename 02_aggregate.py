import pandas as pd
import numpy as np
import os
import sys

# 対象
if len(sys.argv) > 1:
    TARGET_STRATEGY = sys.argv[1]
else:
    TARGET_STRATEGY = "AIZE"  # デフォルト
    # TARGET_STRATEGY = "AIZE-Rapid EX"
    # TARGET_STRATEGY = "AIZE EX"
    # TARGET_STRATEGY = "AIZE-Rapid"

# ファイル名用（スペース対策）
SAFE_STRATEGY = TARGET_STRATEGY.replace(" ", "_")
INPUT_FILE = f"01_filtered/filtered_{SAFE_STRATEGY}.csv"
OUTPUT_DIR = "02_output"


# =========================
# データ読み込み
# =========================
def load_data(path):
    print("読み込み中...")
    return pd.read_csv(path, dtype=str, low_memory=False)


# =========================
# 前処理（ソート＋同一日データは最新1件のみ残す）
# =========================
def preprocess(df):
    # 日付変換
    df["Exam. Day"] = pd.to_datetime(df["Exam. Day"])

    # MDを数値化
    df["MD"] = pd.to_numeric(df["MD"], errors="coerce")

    # ★ 日付部分だけ作る
    df["ExamDate"] = df["Exam. Day"].dt.date

    # ★ ソート（時間が新しい順にする）
    df = df.sort_values(
        ["ID", "Exam. EYE", "Pattern", "Exam. Day"],
        ascending=[True, True, True, False]
    )

    # ★ 同一日で重複削除（最新だけ残す）
    df = df.drop_duplicates(
        subset=["ID", "Exam. EYE", "Pattern", "ExamDate"],
        keep="first"
    )

    return df


# =========================
# shift列作成
# =========================
def add_shift_columns(df):
    print("shift列作成中...")

    group_cols = ["ID", "Exam. EYE", "Pattern"]

    # ===== 年齢計算 =====
    df["ExamDate"] = pd.to_datetime(df["ExamDate"], errors="coerce")
    df["BOD"] = pd.to_datetime(df["BOD"], errors="coerce")
    df["age"] = (df["ExamDate"] - df["BOD"]).dt.days / 365.25

    # ===== shift =====
    df["ExamDate_1"] = df.groupby(group_cols)["ExamDate"].shift(-1)
    df["MD_1"] = df.groupby(group_cols)["MD"].shift(-1)
    df["age_1"] = df.groupby(group_cols)["age"].shift(-1)

    df["ExamDate_2"] = df.groupby(group_cols)["ExamDate"].shift(-2)
    df["MD_2"] = df.groupby(group_cols)["MD"].shift(-2)
    df["age_2"] = df.groupby(group_cols)["age"].shift(-2)

    # ===== 平均年齢（3回） =====
    df["age_mean_last3"] = df[["age", "age_1", "age_2"]].mean(axis=1)

    return df

# =========================
# Thr shift列作成
# =========================
def add_thr_shift(df):
    print("Thr shift作成中...")

    group_cols = ["ID", "Exam. EYE", "Pattern"]
    thr_cols = [c for c in df.columns if c.startswith("Thr(")]

    for col in thr_cols:
        df[f"{col}_1"] = df.groupby(group_cols)[col].shift(-1)
        df[f"{col}_2"] = df.groupby(group_cols)[col].shift(-2)

    return df

# =========================
# スロープ計算（1行）
# =========================
def calc_slope_row(row):
    # 欠損チェック
    if pd.isna(row["MD"]) or pd.isna(row["MD_1"]) or pd.isna(row["MD_2"]):
        return np.nan

    if pd.isna(row["ExamDate"]) or pd.isna(row["ExamDate_1"]) or pd.isna(row["ExamDate_2"]):
        return np.nan

    dates = [
        row["ExamDate"],
        row["ExamDate_1"],
        row["ExamDate_2"]
    ]

    mds = [
        row["MD"],
        row["MD_1"],
        row["MD_2"]
    ]

    # 年換算
    base = min(dates)
    years = [(d - base).days / 365.25 for d in dates]

    # # 同一日データは無視
    # if len(set(years)) < 2:
    #     return np.nan

    # 回帰
    try:
        return np.polyfit(years, mds, 1)[0]
    except:
        return np.nan

# =========================
# スロープ追加
# =========================
def add_slope(df):
    print("スロープ計算中...")
    df["MD_slope_last3"] = df.apply(calc_slope_row, axis=1)
    return df

# =========================
# 対象データ抽出（キーのみ）
# =========================
def get_target_records(df):
    print("対象データ抽出中...")

    df_filtered = df[
        (df["MD_slope_last3"].abs() <= 0.5)
    ]

    df_latest = df_filtered.groupby(
        ["ID", "Exam. EYE", "Pattern"]
    ).head(1)

    # ★ キーだけにする
    result = df_latest[
        ["ID", "Exam. EYE", "Pattern", "ExamDate"]
    ]

    print(f"対象件数: {len(result)}")

    return result

# =========================
# targetに対応する元データ抽出
# =========================
def extract_target_df(df, target):
    print("target対応データ抽出中...")

    merged = df.merge(
        target,
        on=["ID", "Exam. EYE", "Pattern", "ExamDate"],
        how="inner"
    )

    print(f"対象レコード数: {len(merged)}")
    return merged

# =========================
# Thrデータ整形（縦持ち）
# =========================
def reshape_thr_data(df):
    print("閾値データ整形中...")

    thr_cols = [c for c in df.columns if c.startswith("Thr(")]
    records = []

    for _, row in df.iterrows():
        # 3回揃ってないものは除外
        if pd.isna(row["MD_2"]):
            continue


        for col in thr_cols:
            base = row.get(f"{col}_2")
            after1 = row.get(f"{col}_1")
            after2 = row.get(col)

            # ★ baseが空欄ならスキップ
            if pd.isna(base) or base == "":
                continue

            # 数値変換
            try:
                base = float(base)
                after1 = float(after1)
                after2 = float(after2)
            except:
                continue

            records.append({
                "ID": row["ID"],
                "Exam. EYE": row["Exam. EYE"],
                "Pattern": row["Pattern"],
                "Point": col,

                "base": base,
                "after1": after1,
                "after2": after2,

                "diff1": after1 - base,
                "diff2": after2 - base
            })

    result = pd.DataFrame(records)
    print(f"整形後件数: {len(result)}")

    return result

def remove_blind_spot(thr_df):

    thr_df = thr_df[~(
        ((thr_df["Exam. EYE"] == "右眼") & (thr_df["Point"] == "Thr(15・-3)")) |
        ((thr_df["Exam. EYE"] == "左眼") & (thr_df["Point"] == "Thr(-15・-3)"))
    )]
    print(f"盲点除外後件数: {len(thr_df)}")

    return thr_df

# =========================
# 分布統計（after1 + after2 合算）
# =========================
def calc_distribution(thr_data):
    print("分布計算中...")

    # after1, after2 を縦に結合
    df_long = pd.concat([
        thr_data[["base", "after1"]].rename(columns={"after1": "after"}),
        thr_data[["base", "after2"]].rename(columns={"after2": "after"})
    ])

    # NaN除去（afterが空のもの）
    df_long = df_long.dropna(subset=["after"])

    # 数値変換（念のため）
    df_long["after"] = pd.to_numeric(df_long["after"], errors="coerce")
    df_long = df_long.dropna(subset=["after"])

    # ===== 集計 =====
    result = df_long.groupby("base")["after"].agg(
        p95=lambda x: x.quantile(0.95),
        p75=lambda x: x.quantile(0.75),
        p25=lambda x: x.quantile(0.25),
        p05=lambda x: x.quantile(0.05),
        count="count"
    ).reset_index()

    print(f"分布算出完了: {len(result)} グループ")

    return result

# =========================
# サマリ（母集団＋対象まとめ）
# =========================
def summarize_target(df, df_target):
    print("\n=== サマリ ===")

    group_cols = ["ID", "Exam. EYE", "Pattern"]

    # ===== 母集団 =====
    total_eyes = df[group_cols].drop_duplicates().shape[0]

    counts = df.groupby(group_cols).size()
    eyes_3plus = (counts >= 3).sum()

    # ===== 対象 =====
    target_eyes = len(df_target)

    # ===== 初回MD =====
    md_base = pd.to_numeric(df_target["MD_2"], errors="coerce").dropna()
    md_mean = md_base.mean()
    md_std = md_base.std()

    # ===== 右眼 / 左眼 =====
    eye_counts = df_target["Exam. EYE"].value_counts()
    right_eye = eye_counts.get("右眼", 0)
    left_eye = eye_counts.get("左眼", 0)

    # ===== 患者単位 =====
    patient_count = df_target["ID"].nunique()

    df_patient = df_target.groupby("ID").agg({
        "age_mean_last3": "mean",
        "Gender": "first"
    }).reset_index()

    mean_age = df_patient["age_mean_last3"].mean()
    std_age = df_patient["age_mean_last3"].std()

    male = (df_patient["Gender"] == "男").sum()
    female = (df_patient["Gender"] == "女").sum()

    # ===== 表示 =====
    print(f"全眼数: {total_eyes}")
    print(f"3回以上検査眼数: {eyes_3plus}")
    print(f"対象眼数: {target_eyes}")
    print(f"右眼: {right_eye}")
    print(f"左眼: {left_eye}")
    print(f"初回MD平均: {md_mean:.2f}")
    print(f"初回MD SD: {md_std:.2f}")


    print(f"対象者数: {patient_count}")
    print(f"男性: {male}")
    print(f"女性: {female}")
    print(f"平均年齢(3回平均): {mean_age:.2f}")
    print(f"年齢SD: {std_age:.2f}")

    # ===== 保存 =====
    summary_df = pd.DataFrame([{
        "Strategy": TARGET_STRATEGY,

        "全眼数": total_eyes,
        "3回以上検査眼数": eyes_3plus,
        "対象眼数": target_eyes,

        "右眼": right_eye,
        "左眼": left_eye,
        "初回MD平均": md_mean,
        "初回MD_SD": md_std,

        "対象者数": patient_count,
        "男性": male,
        "女性": female,
        "平均年齢(3回平均)": mean_age,
        "年齢SD": std_age,
    }])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_00_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("＝サマリ出力完了＝")

    return summary_df

# =========================
# デバッグ表示
# =========================
def debug_print(df):
    cols = [
        "ID", "Exam. EYE", "Pattern",
        "ExamDate", "MD",
        "ExamDate_1", "MD_1",
        "ExamDate_2", "MD_2",
        "MD_slope_last3"
    ]

    print("\n=== デバッグ表示 ===")
    print(f"df 件数: {len(df)}")
    print("=== head ===")
    print(df.head(20)[cols])

    # 任意のIDで確認
    # test_id = df["ID"].iloc[0]
    # print(f"\n=== ID別確認: {test_id} ===")
    # print(df[df["ID"] == test_id][cols])


# =========================
# メイン
# =========================
def main():
    print("解析開始")

    df = load_data(INPUT_FILE)
    df = preprocess(df)
    df = add_shift_columns(df)
    df = add_thr_shift(df)
    df = add_slope(df)

    # ===== target抽出 =====
    target = get_target_records(df)

    # ===== 元データ復元 =====
    df_target = extract_target_df(df, target)

    # ===== サマリ =====
    summarize_target(df, df_target)

    # ===== Thr処理 =====
    thr_data = reshape_thr_data(df_target)
    thr_data = remove_blind_spot(thr_data)

    # ===== 分布 =====
    dist = calc_distribution(thr_data)

    # ===== 出力 =====
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_01_preprocess.csv"), index=False, encoding="utf-8-sig")
    target.to_csv(os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_02_target_keys.csv"), index=False, encoding="utf-8-sig")
    df_target.to_csv(os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_03_target_full.csv"), index=False, encoding="utf-8-sig")
    thr_data.to_csv(os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_04_thr_analysis.csv"), index=False, encoding="utf-8-sig")
    dist.to_csv(os.path.join(OUTPUT_DIR, f"_{SAFE_STRATEGY}_05_distribution.csv"), index=False, encoding="utf-8-sig")

    print("\n出力完了")

if __name__ == "__main__":
    main()