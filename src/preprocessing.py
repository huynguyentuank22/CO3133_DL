"""
Data preprocessing: cleaning, full_text creation, stratified splitting.
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(csv_path: str = config.RAW_CSV) -> pd.DataFrame:
    """Load raw CSV and standardize columns."""
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Rename for easier access
    col_map = {
        "Clothing_ID": "clothing_id", "Age": "age",
        "Title": "title", "Review_Text": "review_text",
        "Rating": "rating", "Recommended_IND": "recommended",
        "Positive_Feedback_Count": "feedback_count",
        "Division_Name": "division", "Department_Name": "department",
        "Class_Name": "class_name",
    }
    df.rename(columns=col_map, inplace=True)
    logger.info(f"Loaded raw data: {len(df)} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean: drop missing review/rating, fill title, deduplicate, short reviews."""
    initial = len(df)

    # Drop rows where review_text or rating is missing
    df = df.dropna(subset=["review_text", "rating"]).copy()
    logger.info(f"After dropping missing review_text/rating: {len(df)} rows (removed {initial - len(df)})")

    # Fill missing title with empty string
    df["title"] = df["title"].fillna("")

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title", "review_text", "rating"]).copy()
    logger.info(f"After deduplication: {len(df)} rows (removed {before_dedup - len(df)})")

    # Normalize whitespace
    df["review_text"] = df["review_text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
    df["title"] = df["title"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

    # Remove very short reviews
    df["_token_count"] = df["review_text"].apply(lambda x: len(x.split()))
    before_short = len(df)
    df = df[df["_token_count"] >= config.MIN_TOKEN_LENGTH].copy()
    df.drop(columns=["_token_count"], inplace=True)
    logger.info(f"After removing short reviews (<{config.MIN_TOKEN_LENGTH} tokens): {len(df)} rows (removed {before_short - len(df)})")

    # Ensure rating is int
    df["rating"] = df["rating"].astype(int)

    return df.reset_index(drop=True)


def create_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create full_text = Title [SEP] Review Text, and map labels."""
    df = df.copy()
    df["full_text"] = df.apply(
        lambda row: (row["title"] + " [SEP] " + row["review_text"]).strip()
        if row["title"] else row["review_text"],
        axis=1,
    )
    df["label"] = df["rating"].map(config.LABEL_MAP)
    return df


def stratified_split(df: pd.DataFrame, seed: int = config.SEED):
    """Split into train/val/test with stratification on label."""
    train_df, temp_df = train_test_split(
        df, test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=df["label"], random_state=seed,
    )
    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=seed,
    )
    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df, out_dir=config.DATA_SPLITS_DIR):
    """Save splits to CSV."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    logger.info(f"Saved splits to {out_dir}")


def run_preprocessing_pipeline():
    """Full preprocessing pipeline."""
    config.ensure_dirs()
    df = load_raw_data()
    df = clean_data(df)
    df = create_full_text(df)
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)

    # Log class distributions
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts().sort_index()
        logger.info(f"{name} distribution:\n{dist.to_string()}")

    return train_df, val_df, test_df
