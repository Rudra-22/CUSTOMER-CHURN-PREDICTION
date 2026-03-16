# =============================================
#  CHUNK 1 — LOAD AND EXPLORE THE DATA
#  Run this first to see what data we have
# =============================================

# Step 1: Import the libraries we need
import pandas as pd   # type: ignore # for working with data tables


def load_and_clean(path: str = 'WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """Load the Telco churn CSV and clean the TotalCharges column."""
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df


def summarize(df: pd.DataFrame) -> str:
    """Generate a printable summary of the dataset."""
    churn_rate = (df['Churn'] == 'Yes').mean() * 100

    parts = [
        "=" * 40,
        "  DATASET LOADED SUCCESSFULLY",
        "=" * 40,
        f"\n Total Rows    : {df.shape[0]}",
        f" Total Columns : {df.shape[1]}",
        "\n First 5 Rows of Data:",
        str(df.head()),
        "\n Column Names:",
        str(df.columns.tolist()),
        "\n Data Types:",
        str(df.dtypes),
        "\n Missing Values:",
        str(df.isnull().sum()),
        "\n Churn Breakdown:",
        str(df['Churn'].value_counts()),
        f"\n Churn Rate = {churn_rate:.1f}%",
    ]
    return "\n".join(parts)


def save_clean(df: pd.DataFrame, path: str = 'clean_data.csv') -> None:
    """Save the cleaned DataFrame for downstream steps."""
    df.to_csv(path, index=False)


if __name__ == '__main__':
    df = load_and_clean()
    print(summarize(df))
    save_clean(df)
    print("\n Saved clean_data.csv — ready for next step!")