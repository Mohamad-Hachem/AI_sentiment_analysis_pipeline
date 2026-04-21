from hardware_preparation import torch, gpu_preparation 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from datasets import load_dataset
from IPython.display import display, Markdown
import random
from model_manager import ModelManager
from dataset_loader import DatasetLoader
import re
import os
from transformers.utils import logging
logging.set_verbosity_error()
class SentimentPipeline:
    """
        an llm pipeline for sentiment analysis of datasets
    """

    def __init__(self, model_manager: ModelManager, dataset_loader: DatasetLoader):
        self.model_manager = model_manager
        self.dataset_loader = dataset_loader
        self.llm_pipeline = pipeline("text-generation", model = model_manager.model,
                                      tokenizer=model_manager.tokenizer, dtype="auto", device_map="auto" )
        
        # remove the default length limit that conflicts with max_new_tokens
        self.llm_pipeline.model.generation_config.max_length = None
    
    
    

    def running_pipeline_on_a_record(self, phrase, max_retries=2):
        """
        Return:
        1  -> positive
        0  -> neutral
        -1 -> negative
        """

        prompt = f"""Classify the sentiment of this phrase.

                    Reply with only one of these exact values:
                    1
                    0
                    -1

                    Phrase: {phrase}

                    Rules:
                    1 = positive
                    0 = neutral
                    -1 = negative

                    Answer:"""

        for attempt in range(max_retries + 1):
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=6,
                do_sample=False,
                return_full_text=False,
                # optional if your transformers version supports it:
                # stop_strings=["\n"]
            )

            raw = outputs[0]["generated_text"].strip()

            # take only the first line
            first_line = raw.splitlines()[0].strip()

            # take only the first token on that line
            first_token = first_line.split()[0] if first_line else ""

            # direct exact parse
            if first_token in {"1", "0", "-1"}:
                return int(first_token)

            # relaxed parse for cases like ": -1" or "label=1"
            match = re.search(r"(-1|0|1)", first_line)
            if match:
                return int(match.group(1))

            print(f"Invalid output: {raw!r}. Retrying...")

        return None
        

    def iterating_over_a_column_dataset(self):
        column_name = self.dataset_loader.ask_user_to_choose_a_column_to_work_on_the_dataset()

        def add_sentiment(example):
            phrase = str(example[column_name])
            sentiment = self.running_pipeline_on_a_record(phrase)

            """
            print("\n--- ROW ---")
            print("id:", example.get("id"))
            print("text:", phrase)
            print("sentiment:", sentiment)
            """

            return {"sentiment": sentiment}

        self.dataset_loader.dataset = self.dataset_loader.dataset.map(add_sentiment)
        df = self.dataset_loader.dataset.to_pandas()
        os.makedirs("reports_csv_files", exist_ok=True)
        df.to_csv(os.path.join("reports_csv_files", "sentiment_debug_output.csv"), index=False)


    def generating_report(self, csv_path=None):
        import os
        import json
        import re
        from datetime import datetime
        from collections import Counter
        import pandas as pd

        if csv_path is None:
            csv_path = os.path.join("reports_csv_files", "sentiment_debug_output.csv")

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)

        if "sentiment" not in df.columns:
            print("No 'sentiment' column found in the CSV.")
            return

        text_column = getattr(self, "selected_column", None)

        if text_column is None or text_column not in df.columns:
            if "text" in df.columns:
                text_column = "text"
            else:
                possible_text_cols = [col for col in df.columns if col != "sentiment" and df[col].dtype == "object"]
                if not possible_text_cols:
                    print("Could not identify the text column in the CSV.")
                    return
                text_column = possible_text_cols[0]

        df[text_column] = df[text_column].astype(str)

        valid_df = df[df["sentiment"].notna()].copy()

        if valid_df.empty:
            print("All sentiment values are empty/None. No report can be generated.")
            return

        valid_df["sentiment"] = pd.to_numeric(valid_df["sentiment"], errors="coerce")
        valid_df = valid_df[valid_df["sentiment"].notna()].copy()
        valid_df["sentiment"] = valid_df["sentiment"].astype(int)

        if valid_df.empty:
            print("No valid numeric sentiment values found.")
            return

        sentiment_map = {
            -1: "negative",
            0: "neutral",
            1: "positive"
        }

        valid_df["sentiment_name"] = valid_df["sentiment"].map(sentiment_map)
        valid_df["char_count"] = valid_df[text_column].apply(len)
        valid_df["word_count"] = valid_df[text_column].apply(lambda x: len(x.split()))

        total_rows = len(valid_df)

        sentiment_counts = valid_df["sentiment_name"].value_counts().to_dict()
        sentiment_percentages = {
            k: round((v / total_rows) * 100, 2)
            for k, v in sentiment_counts.items()
        }

        avg_lengths = (
            valid_df.groupby("sentiment_name")[["char_count", "word_count"]]
            .mean()
            .round(2)
            .to_dict(orient="index")
        )

        polarity_score = round(valid_df["sentiment"].mean(), 4)

        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for",
            "with", "is", "it", "this", "that", "was", "were", "are", "am", "be",
            "been", "being", "i", "you", "he", "she", "they", "we", "my", "your",
            "his", "her", "their", "me", "him", "them", "our", "at", "by", "from",
            "as", "about", "so", "very", "just", "too", "not", "no", "yes", "do",
            "does", "did", "have", "has", "had"
        }

        def extract_top_words(series, n=15):
            words = []
            for text in series.dropna():
                tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
                tokens = [token for token in tokens if token not in stopwords]
                words.extend(tokens)
            return Counter(words).most_common(n)

        top_words_by_sentiment = {}
        for label_name in ["positive", "neutral", "negative"]:
            subset = valid_df[valid_df["sentiment_name"] == label_name]
            top_words_by_sentiment[label_name] = extract_top_words(subset[text_column], n=15)

        example_rows = {}
        for label_name in ["positive", "neutral", "negative"]:
            subset = valid_df[valid_df["sentiment_name"] == label_name]
            example_rows[label_name] = subset[text_column].head(5).tolist()

        confusion = None
        agreement = None

        if "label" in valid_df.columns:
            try:
                comparison_df = valid_df.copy()
                comparison_df["label"] = pd.to_numeric(comparison_df["label"], errors="coerce")
                comparison_df = comparison_df[comparison_df["label"].notna()].copy()

                if not comparison_df.empty:
                    comparison_df["label"] = comparison_df["label"].astype(int)
                    unique_labels = set(comparison_df["label"].unique().tolist())

                    if unique_labels.issubset({0, 1, 2}):
                        label_map = {0: -1, 1: 0, 2: 1}
                        comparison_df["label_mapped"] = comparison_df["label"].map(label_map)
                    elif unique_labels.issubset({-1, 0, 1}):
                        comparison_df["label_mapped"] = comparison_df["label"]
                    else:
                        comparison_df["label_mapped"] = None

                    comparison_df = comparison_df[comparison_df["label_mapped"].notna()].copy()

                    if not comparison_df.empty:
                        comparison_df["label_name"] = comparison_df["label_mapped"].map(sentiment_map)

                        confusion = pd.crosstab(
                            comparison_df["label_name"],
                            comparison_df["sentiment_name"],
                            rownames=["actual"],
                            colnames=["predicted"]
                        )

                        agreement = round(
                            (comparison_df["label_mapped"] == comparison_df["sentiment"]).mean() * 100,
                            2
                        )
            except Exception as e:
                print(f"Could not compute agreement metrics: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0].replace("-", "_")
        report_dir = os.path.join("reports", f"sentiment_report_{timestamp}_{csv_stem}")
        os.makedirs(report_dir, exist_ok=True)

        valid_df.to_csv(os.path.join(report_dir, "sentiment_dataset.csv"), index=False)

        for label_name, words in top_words_by_sentiment.items():
            pd.DataFrame(words, columns=["word", "count"]).to_csv(
                os.path.join(report_dir, f"top_words_{label_name}.csv"),
                index=False
            )

        if confusion is not None:
            confusion.to_csv(os.path.join(report_dir, "confusion_matrix.csv"))

        summary = {
            "csv_path": csv_path,
            "text_column_used": text_column,
            "total_rows_analyzed": int(total_rows),
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "average_lengths": avg_lengths,
            "polarity_score": polarity_score,
            "agreement_with_labels_percent": agreement,
            "top_words_by_sentiment": {
                k: [{"word": w, "count": c} for w, c in v]
                for k, v in top_words_by_sentiment.items()
            },
            "example_rows": example_rows
        }

        with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        lines = []
        lines.append("=" * 70)
        lines.append("SENTIMENT ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Source CSV: {csv_path}")
        lines.append(f"Text column used: {text_column}")
        lines.append(f"Rows analyzed: {total_rows}")
        lines.append("")

        lines.append("Sentiment distribution:")
        for label_name in ["positive", "neutral", "negative"]:
            count = sentiment_counts.get(label_name, 0)
            pct = sentiment_percentages.get(label_name, 0.0)
            lines.append(f"  - {label_name:<8}: {count:>6} rows ({pct}%)")

        lines.append("")
        lines.append(f"Polarity score: {polarity_score}")
        lines.append("  closer to 1 = more positive, closer to -1 = more negative")
        lines.append("")

        lines.append("Average text length by sentiment:")
        for label_name, values in avg_lengths.items():
            lines.append(
                f"  - {label_name:<8}: {values['char_count']} avg chars, {values['word_count']} avg words"
            )

        lines.append("")
        lines.append("Top keywords by sentiment:")
        for label_name, words in top_words_by_sentiment.items():
            word_str = ", ".join([f"{word}({count})" for word, count in words[:10]])
            lines.append(f"  - {label_name:<8}: {word_str}")

        if agreement is not None:
            lines.append("")
            lines.append(f"Agreement with original labels: {agreement}%")

        lines.append("")
        lines.append("Sample examples:")
        for label_name, examples in example_rows.items():
            lines.append(f"\n{label_name.upper()} EXAMPLES:")
            for i, ex in enumerate(examples[:3], start=1):
                lines.append(f"  {i}. {ex[:180].replace(chr(10), ' ')}")

        report_text = "\n".join(lines)

        with open(os.path.join(report_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {report_dir}")
    

if __name__ == "__main__":
    new_model_manager = ModelManager()
    new_dataset_loader = DatasetLoader()
    new_pipeline = SentimentPipeline(new_model_manager, new_dataset_loader)

    #new_pipeline.iterating_over_a_column_dataset()
    #new_pipeline.dataset_loader.checking_dataset_as_pandas()
    new_pipeline.generating_report(os.path.join("reports_csv_files", "sentiment_debug_output.csv"))