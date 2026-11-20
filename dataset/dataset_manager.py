from dataclasses import dataclass
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from textwrap import wrap

from config.tca_config import TCAConfig


class DatasetManager:

    def __init__(self, config: TCAConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        filepath = self.config.file_path
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df.dropna(axis=1, how="all")

    def extract_conversation_pairs(self, row: pd.Series) -> List[Tuple[str, str]]:
        """
            Extracts assistant and human conversation pairs from a row of a DataFrame,
            removing the "role" field from the messages.

            Args:
                row (pd.Series): A row of the DataFrame containing conversation messages.

            Returns:
                List[Tuple[str, str]]: A list of (assistant_message, human_message) tuples.
        """
        # Extract message columns dynamically
        message_cols = [col for col in row.index if col.startswith("message_")]

        # Sort columns to ensure proper order (e.g., message_0, message_1, ...)
        message_cols = sorted(message_cols, key=lambda x: int(x.split("_")[1]))

        # Collect messages into a list and remove the "role" field if present
        messages = []
        for col in message_cols:
            raw_message = row[col]
            if pd.notna(raw_message) and str(raw_message).strip():
                # Attempt to parse message as JSON and remove "role"
                try:
                    message_data = json.loads(raw_message)
                    message_body = message_data.get("body", raw_message)  # Default to raw_message if no "body"
                    messages.append(message_body)
                except json.JSONDecodeError:
                    messages.append(raw_message)

        # Create conversation pairs (only complete pairs with both turns)
        conversation_pairs = []
        for i in range(0, len(messages) - 1, 2):
            assistant_message = messages[i]
            human_message = messages[i + 1]
            if assistant_message and human_message:
                conversation_pairs.append((assistant_message, human_message))

        return conversation_pairs

    # def process_row(self, row: pd.Series) -> Dict[str, Any]:
    #     return {
    #         "source": row["Source"],
    #         "temperature": row["temperature"],
    #         "tactic": row["tactic"],
    #         "question_id": row["question_id"],
    #         "time_spent": row["time_spent"],
    #         "submission_message": row["submission_message"],
    #         "conversation": self.get_conversation_history(row),
    #     }

    def get_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        metadata = {
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "unique_sources": df["Source"].unique().tolist(),
            "unique_tactics": df["tactic"].unique().tolist(),
            "unique_tactics_count": len(df["tactic"].unique()),
            "tactic_counts": df["tactic"].value_counts().to_dict(),
            "avg_time_by_tactic" : df.groupby('tactic')['time_spent'].mean(),
            "avg_time_spent": df["time_spent"].mean(),
        }

        print("Dataset Metadata:")
        for key, value in metadata.items():
            print(f"  {key}:")
            if isinstance(value, list):
                for item in value:
                    print(f"    - {item}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"    - {sub_key}: {sub_value}")
            else:
                print(f"    {value}")

        return metadata

    def display_beautified_conversation(self, row):
        # Header with conversation metadata
        print("\n" + "╔" + "═" * 80 + "╗")
        header = f"║ Tactic: {row['tactic']:<20} | Temp: {row['temperature']:<5} | Source: {row['Source']:<20} ║"
        print(header)
        print("╠" + "═" * 80 + "╣")

        # Get and display messages
        message_cols = [col for col in row.index if col.startswith("message_")]

        for col in message_cols:
            message = row[col]
            if pd.notna(message) and str(message).strip():
                # Determine speaker and color
                is_assistant = int(col.split("_")[1]) % 2 == 0
                speaker = "🤖 Assistant" if is_assistant else "👤 Human"

                # Format message with proper wrapping
                # Split message into lines of max 70 characters
                words = message.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 <= 70:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    lines.append(" ".join(current_line))

                # Print formatted message
                print(f"║ {speaker:<12} │")
                for line in lines:
                    print(f"║ {line:<79}║")
                print("║" + " " * 80 + "║")

        print("╚" + "═" * 80 + "╝")

    # ---------------------------------------------------------------------- #
    # ZN Prompt-Injection Benchmark Integration                              #
    # (usezn/zn-prompt-injection-bench)                                      #
    # ---------------------------------------------------------------------- #

    def load_zn_benchmark(
        self,
        split: Optional[str] = None,
        family: Optional[str] = None,
        label: Optional[str] = None,
        filepath: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Loads the ZN Prompt Injection Benchmark corpus (23,699 samples).

        Args:
            split (str, optional): 'train', 'val', or 'test'.
            family (str, optional): Family filter (e.g., 'plain', 'red_team', 'conversacional').
            label (str, optional): 'ataque' or 'benigno'.
            filepath (str, optional): Custom path to the CSV file.
            limit (int, optional): Maximum number of rows to load.

        Returns:
            pd.DataFrame: Loaded and filtered DataFrame.
        """
        if filepath is None:
            # Default to dataset/zn_prompt_injection_bench.csv relative to project root
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, "zn_prompt_injection_bench.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ZN Prompt-Injection benchmark file not found at: {filepath}")

        df = pd.read_csv(filepath)

        if split is not None:
            df = df[df["split"] == split]

        if family is not None:
            df = df[df["family"] == family]

        if label is not None:
            df = df[df["label"] == label]

        if limit is not None and limit > 0:
            df = df.head(limit)

        return df.reset_index(drop=True)

    def get_zn_metadata(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Computes detailed descriptive statistics and analytics for the ZN Benchmark.

        Args:
            df (pd.DataFrame, optional): Loaded ZN benchmark dataframe. If None, loads full corpus.

        Returns:
            Dict[str, Any]: Analytics metadata.
        """
        if df is None:
            df = self.load_zn_benchmark()

        metadata = {
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "splits": df["split"].value_counts().to_dict() if "split" in df.columns else {},
            "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {},
            "family_distribution": df["family"].value_counts().to_dict() if "family" in df.columns else {},
            "source_distribution": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "language_distribution": df["lang"].value_counts(dropna=False).to_dict() if "lang" in df.columns else {},
            "char_length_stats": {
                "min": int(df["char_len"].min()),
                "max": int(df["char_len"].max()),
                "mean": float(round(df["char_len"].mean(), 2)),
                "median": float(df["char_len"].median()),
                "std": float(round(df["char_len"].std(), 2)),
            } if "char_len" in df.columns else {},
        }
        return metadata

    def extract_zn_conversations(
        self,
        df: Optional[pd.DataFrame] = None,
        split: Optional[str] = "test",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts multi-turn conversational sequences from the 'conversacional' family
        in the ZN benchmark, parsing 'Human:' and 'Assistant:' message turns.

        Args:
            df (pd.DataFrame, optional): Source DataFrame.
            split (str, optional): Split filter ('test', 'val', 'train'). Defaults to 'test'.
            limit (int, optional): Maximum conversations to return.

        Returns:
            List[Dict[str, Any]]: List of structured conversation dictionaries.
        """
        if df is None:
            df = self.load_zn_benchmark(split=split, family="conversacional")
        else:
            if "family" in df.columns:
                df = df[df["family"] == "conversacional"]
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split]

        conversations = []
        pattern = re.compile(r"\b(Human|Assistant):")

        for idx, row in df.iterrows():
            text = str(row["text"])
            parts = pattern.split(text)
            turns = []
            current_role = None

            for part in parts:
                cleaned = part.strip()
                if not cleaned:
                    continue
                if cleaned in ("Human", "Assistant"):
                    current_role = cleaned.lower()
                elif current_role:
                    turns.append({"role": current_role, "text": cleaned})

            if len(turns) >= 2:
                conversations.append({
                    "id": f"zn_conv_{idx}",
                    "source": row.get("source", "hh_rlhf"),
                    "split": row.get("split", "test"),
                    "label": row.get("label", "benigno"),
                    "family": "conversacional",
                    "turns": turns,
                    "turn_count": len(turns),
                })

            if limit is not None and len(conversations) >= limit:
                break

        return conversations


# Example usage
# config = TCAConfig("config/config.yaml")
# print(config)
# manager = DatasetManager(config)  # Pass entire config object
# df = manager.load_data()  # load_data will use config.file_path
# # Iterate through each unique tactic
# for tactic in df["tactic"].unique():
#     # Filter rows for the current tactic
#     tactic_df = df[df["tactic"] == tactic]

#     # Iterate through each row in the filtered DataFrame
#     for _, row in tactic_df.iterrows():
#         # Extract conversation pairs
#         conversation_pairs = manager.extract_conversation_pairs(row)

#         # Print the results for each row
#         print(f"Tactic: {tactic}")
#         print(conversation_pairs)
#         print("\n" + "-" * 80 + "\n")
