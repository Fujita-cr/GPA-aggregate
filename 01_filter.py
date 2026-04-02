import pandas as pd

# =========================
# 設定
# =========================
INPUT_FILE = "_marge_Result.csv"

# =========================
# 関数定義
# =========================
def export_by_strategy(df, strategy):
    filtered = df[df["Strategy"] == strategy]

    # ファイル名用に整形
    safe_name = strategy.replace(" ", "_")
    output_file = f"filtered_{safe_name}.csv"

    # 保存
    filtered.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"出力完了: {output_file} ({len(filtered)}件)")


# =========================
# メイン処理
# =========================
def main():
    print("読み込み中...")
    df = pd.read_csv(INPUT_FILE, dtype=str)

    print("抽出開始")

    export_by_strategy(df, "AIZE")
    export_by_strategy(df, "AIZE EX")
    export_by_strategy(df, "AIZE-Rapid")
    export_by_strategy(df, "AIZE-Rapid EX")

    print("すべて完了")


if __name__ == "__main__":
    main()