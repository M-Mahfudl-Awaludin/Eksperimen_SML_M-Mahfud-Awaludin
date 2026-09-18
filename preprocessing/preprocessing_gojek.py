from pathlib import Path
import re

import pandas as pd
import numpy as np

from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ============================================================
# PATH PROJECT
# ============================================================

# Folder tempat file preprocessing_gojek.py berada
# = repository/preprocessing/
BASE_DIR = Path(__file__).resolve().parent

# Folder root repository
# = repository/
PROJECT_DIR = BASE_DIR.parent

# Dataset input berada di root repository
INPUT_FILE = PROJECT_DIR / "reviews_gojek.csv"

# Dataset hasil preprocessing berada di folder preprocessing/
OUTPUT_FILE = BASE_DIR / "processed_data.csv"


# ============================================================
# TQDM
# ============================================================

tqdm.pandas()


# ============================================================
# SLANG DICTIONARY
# ============================================================

slang_dict = {
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "tdk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "kagak": "tidak",
    "udah": "sudah",
    "udh": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "belom": "belum",
    "bgt": "banget",
    "aja": "saja",
    "aj": "saja",
    "doang": "saja",
    "kalo": "kalau",
    "klo": "kalau",
    "klu": "kalau",
    "krn": "karena",
    "karna": "karena",
    "dpt": "dapat",
    "dapet": "dapat",
    "gmn": "bagaimana",
    "knp": "mengapa",
    "dmn": "di mana",
    "disini": "di sini",
    "dsini": "di sini",
    "disana": "di sana",
    "bs": "bisa",
    "hrs": "harus",
    "sm": "sama",
    "sbg": "sebagai",
    "utk": "untuk",
    "pd": "pada",
    "dr": "dari",
    "dri": "dari",
    "tau": "tahu",
    "pengen": "ingin",
    "pgn": "ingin",
    "ngerti": "mengerti",
    "ngert": "mengerti",
    "gitu": "begitu",
    "gtu": "begitu",
    "gini": "begini",
    "gni": "begini",
    "emang": "memang",
    "emg": "memang",
    "bener": "benar",
    "bnr": "benar",
    "kayak": "seperti",
    "kaya": "seperti",
    "kek": "seperti",
    "temen": "teman",
    "tmn": "teman",
    "org": "orang",
    "makasih": "terima kasih",
    "mksh": "terima kasih",
    "thanks": "terima kasih",
    "thx": "terima kasih",
    "pls": "tolong",
    "plis": "tolong",
    "sorry": "maaf",
    "sori": "maaf",
    "lg": "lagi",
    "lagi2": "lagi-lagi",
    "trus": "terus",
    "trs": "terus",
    "jd": "jadi",
    "jdi": "jadi",
    "bkn": "bukan",
    "spt": "seperti",
    "sbnrnya": "sebenarnya",
    "sebenernya": "sebenarnya",
    "kyknya": "sepertinya",
    "kayaknya": "sepertinya",
    "mgkn": "mungkin",
    "msih": "masih",
    "bwt": "buat",
    "pdhl": "padahal",
    "cuma": "hanya",
    "cm": "hanya",
    "tp": "tetapi",
    "tpi": "tetapi",
    "soalnya": "karena",
    "soalny": "karena",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "loe": "kamu",
    "elo": "kamu",
    "aq": "aku",
    "km": "kamu",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Membersihkan teks dari URL, mention, hashtag,
    karakter khusus, dan whitespace berlebih.
    """

    text = str(text)

    # Remove URL
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Lowercase
    text = text.lower()

    # Remove mention
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag
    text = re.sub(
        r"#\w+",
        "",
        text
    )

    # Remove apostrophe + word
    text = re.sub(
        r"'\w+",
        "",
        text
    )

    # Remove punctuation dan karakter khusus
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SLANG NORMALIZATION
# ============================================================

def ganti_slang(teks):
    """
    Mengubah kata slang menjadi bentuk baku
    berdasarkan slang_dict.
    """

    def replace_word(match):
        kata = match.group(0)

        return slang_dict.get(
            kata.lower(),
            kata
        )

    return re.sub(
        r"\b[\w]+\b",
        replace_word,
        str(teks)
    )


# ============================================================
# LABEL SENTIMENT
# ============================================================

def get_sentiment(score):
    """
    Label sentimen berdasarkan score:

    4-5 = positive
    3   = neutral
    1-2 = negative
    """

    if score >= 4:
        return "positive"

    elif score == 3:
        return "neutral"

    else:
        return "negative"


def get_label(score):
    """
    Label numerik:

    positive = 1
    neutral  = 0
    negative = -1
    """

    if score >= 4:
        return 1

    elif score == 3:
        return 0

    else:
        return -1


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 60)
    print("GOJEK SENTIMENT PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK PATH
    # --------------------------------------------------------

    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n[1] Membaca dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Jumlah data awal: {len(df)}")
    print(f"Jumlah kolom     : {len(df.columns)}")

    # --------------------------------------------------------
    # DROP COLUMN
    # --------------------------------------------------------

    print("\n[2] Menghapus kolom yang tidak diperlukan...")

    columns_to_drop = [
        "reviewId",
        "userImage",
        "thumbsUpCount",
        "reviewCreatedVersion",
        "replyContent",
        "repliedAt",
        "appVersion",
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # --------------------------------------------------------
    # REMOVE MISSING VALUE
    # --------------------------------------------------------

    df = df.dropna()

    print(
        f"Jumlah data setelah dropna: {len(df)}"
    )

    # --------------------------------------------------------
    # CHECK TEXT COLUMN
    # --------------------------------------------------------

    if "content" not in df.columns:
        raise KeyError(
            "Kolom 'content' tidak ditemukan "
            "pada reviews_gojek.csv"
        )

    if "score" not in df.columns:
        raise KeyError(
            "Kolom 'score' tidak ditemukan "
            "pada reviews_gojek.csv"
        )

    # --------------------------------------------------------
    # LABELING
    # --------------------------------------------------------

    print("\n[3] Membuat label sentimen...")

    df["sentiment"] = df["score"].apply(
        get_sentiment
    )

    df["label"] = df["score"].apply(
        get_label
    )

    print(
        df["sentiment"].value_counts()
    )

    # --------------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------------

    print("\n[4] Text cleaning...")

    df["text_Clean"] = df["content"].progress_apply(
        clean_text
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print("\n[5] Slang normalization...")

    df["text_Normalization"] = (
        df["text_Clean"].progress_apply(
            ganti_slang
        )
    )

    # --------------------------------------------------------
    # STOPWORD REMOVAL
    # --------------------------------------------------------

    print("\n[6] Stopword removal...")

    nltk.download(
        "stopwords",
        quiet=True
    )

    stop_words = set(
        stopwords.words("indonesian")
    )

    # Tambahan stopword
    stop_words.update({
        "nya",
        "sih",
        "nih",
        "dong",
        "deh",
        "lah",
        "pun",
    })

    df["text_StopWord"] = (
        df["text_Normalization"].progress_apply(
            lambda x: " ".join(
                word
                for word in str(x).split()
                if word not in stop_words
            )
        )
    )

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    print("\n[7] Tokenization...")

    tokenizer = RegexpTokenizer(
        r"\w+"
    )

    df["text_Tokenization"] = (
        df["text_StopWord"].progress_apply(
            tokenizer.tokenize
        )
    )

    # --------------------------------------------------------
    # STEMMING
    # --------------------------------------------------------

    print("\n[8] Stemming dengan Sastrawi...")

    stemmer = StemmerFactory().create_stemmer()

    df["text_Stemmindo"] = (
        df["text_Tokenization"].progress_apply(
            lambda tokens: [
                stemmer.stem(token)
                for token in tokens
            ]
        )
    )

    # --------------------------------------------------------
    # CONVERT TOKEN TO STRING
    # --------------------------------------------------------

    print("\n[9] Membuat text_String...")

    df["text_String"] = (
        df["text_Stemmindo"].progress_apply(
            lambda tokens: " ".join(
                token
                for token in tokens
                if len(token) > 3
            )
        )
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    print("\n[10] Menyimpan hasil preprocessing...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI")
    print("=" * 60)

    print(f"Jumlah data : {len(df)}")
    print(f"Jumlah kolom: {len(df.columns)}")
    print(f"Output      : {OUTPUT_FILE}")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
