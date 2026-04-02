import pandas as pd
import numpy as np

INPUT_FILE = "filtered_AIZE.csv"


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

    df["Exam. Day_1"] = df.groupby(group_cols)["Exam. Day"].shift(-1)
    df["MD_1"] = df.groupby(group_cols)["MD"].shift(-1)

    df["Exam. Day_2"] = df.groupby(group_cols)["Exam. Day"].shift(-2)
    df["MD_2"] = df.groupby(group_cols)["MD"].shift(-2)

    # print("=== Debug Print 3 Exam. Day ===")
    # print(df.head(20)[["ID", "Exam. EYE", "Pattern", "Exam. Day", "Exam. Day_1",  "Exam. Day_2"]])
    return df


# =========================
# スロープ計算（1行）
# =========================
def calc_slope_row(row):
    # 欠損チェック
    if pd.isna(row["MD"]) or pd.isna(row["MD_1"]) or pd.isna(row["MD_2"]):
        return np.nan

    if pd.isna(row["Exam. Day"]) or pd.isna(row["Exam. Day_1"]) or pd.isna(row["Exam. Day_2"]):
        return np.nan

    dates = [
        row["Exam. Day"],
        row["Exam. Day_1"],
        row["Exam. Day_2"]
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
# 対象データ抽出
# =========================
def get_target_records(df):
    print("対象データ抽出中...")

    # ① スロープ条件
    df_filtered = df[
        (df["MD_slope_last3"].abs() <= 0.5)
    ]

    # ② 最新1件だけ取得
    df_latest = df_filtered.groupby(
        ["ID", "Exam. EYE", "Pattern"]
    ).head(1)

    # ③ 必要な列だけ
    result = df_latest[
        ["ID", "Exam. EYE", "Pattern", "Exam. Day"]
    ]

    print(f"対象件数: {len(result)}")

    return result

# =========================
# デバッグ表示
# =========================
def debug_print(df):
    cols = [
        "ID", "Exam. EYE", "Pattern",
        "Exam. Day", "MD",
        "Exam. Day_1", "MD_1",
        "Exam. Day_2", "MD_2",
        "MD_slope_last3"
    ]

    print(f"\ndf 件数: {len(df)}")
    print("\n=== head ===")
    print(df.head(20)[cols])

    # 任意のIDで確認
    # test_id = df["ID"].iloc[0]
    # print(f"\n=== ID別確認: {test_id} ===")
    # print(df[df["ID"] == test_id][cols])


# =========================
# メイン
# =========================
def main():
    df = load_data(INPUT_FILE)
    df = preprocess(df)
    df = add_shift_columns(df)
    df = add_slope(df)
    target = get_target_records(df)

    debug_print(df)
    print(f"\ntarget 件数: {len(target)}")
    print("\n=== target sample ===")
    print(target.sample(10))

    # デバッグ出力
    df.to_csv("_debug_output.csv", index=False, encoding="utf-8-sig")

    # スロープ付きの対象データデバッグ出力
    target_with_slope = df.merge(
        target,
        on=["ID", "Exam. EYE", "Pattern", "Exam. Day"],
        how="inner"
    )[
        ["ID", "Exam. EYE", "Pattern", "Exam. Day", "MD_slope_last3"]
    ]
    target_with_slope.to_csv("_debug_target_with_slope.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()