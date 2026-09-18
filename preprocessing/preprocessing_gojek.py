from pathlib import Path
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = "reviews_gojek.csv"
OUTPUT_FILE = "preprocessing/hasil_Preprocessing_gojek.csv"

tqdm.pandas()

# Download NLTK resource bila belum tersedia
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Kamus slang: kata tidak baku -> kata baku
slang_dict = {
    "yg":"yang","dgn":"dengan","dg":"dengan","tdk":"tidak","gak":"tidak",
    "ga":"tidak","gk":"tidak","nggak":"tidak","ngga":"tidak","kagak":"tidak",
    "udah":"sudah","udh":"sudah","dah":"sudah","blm":"belum","belom":"belum",
    "bgt":"banget","aja":"saja","aj":"saja","doang":"saja","kalo":"kalau",
    "klo":"kalau","klu":"kalau","krn":"karena","karna":"karena","dpt":"dapat",
    "dapet":"dapat","gmn":"bagaimana","knp":"mengapa","dmn":"di mana",
    "disini":"di sini","dsini":"di sini","disana":"di sana","bs":"bisa",
    "hrs":"harus","sm":"sama","sbg":"sebagai","utk":"untuk","pd":"pada",
    "dr":"dari","dri":"dari","tau":"tahu","pengen":"ingin","pgn":"ingin",
    "ngerti":"mengerti","ngert":"mengerti","gitu":"begitu","gtu":"begitu",
    "gini":"begini","gni":"begini","emang":"memang","emg":"memang",
    "bener":"benar","bnr":"benar","kayak":"seperti","kaya":"seperti",
    "kek":"seperti","temen":"teman","tmn":"teman","org":"orang",
    "makasih":"terima kasih","mksh":"terima kasih","thanks":"terima kasih",
    "thx":"terima kasih","pls":"tolong","plis":"tolong","sorry":"maaf",
    "sori":"maaf","lg":"lagi","lagi2":"lagi-lagi","trus":"terus","trs":"terus",
    "jd":"jadi","jdi":"jadi","bkn":"bukan","spt":"seperti",
    "sbnrnya":"sebenarnya","sebenernya":"sebenarnya","kyknya":"sepertinya",
    "kayaknya":"sepertinya","mgkn":"mungkin","msih":"masih","bwt":"buat",
    "pdhl":"padahal","cuma":"hanya","cm":"hanya","tp":"tetapi","tpi":"tetapi",
    "soalnya":"karena","soalny":"karena","gw":"saya","gue":"saya","gua":"saya",
    "lu":"kamu","loe":"kamu","elo":"kamu","aq":"aku","km":"kamu"
}

def preprocess_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
    text = text.lower()
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"'\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def ganti_slang(teks):
    def replace_word(match):
        kata = match.group(0)
        return slang_dict.get(kata.lower(), kata)
    return re.sub(r"\b[\w]+\b", replace_word, str(teks))

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}. "
            "Letakkan reviews_gojek.csv satu folder dengan script."
        )

    print("Membaca dataset...")
    df = pd.read_csv(INPUT_FILE)

    if "content" not in df.columns:
        raise ValueError(f"Kolom 'content' tidak ditemukan. Kolom: {df.columns.tolist()}")
    if "score" not in df.columns:
        raise ValueError(f"Kolom 'score' tidak ditemukan. Kolom: {df.columns.tolist()}")

    columns_to_drop = [
        "reviewId", "userImage", "thumbsUpCount", "reviewCreatedVersion",
        "replyContent", "repliedAt", "appVersion"
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    df = df.dropna(subset=["content", "score"]).copy()
    df["content"] = df["content"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()

    # Label: 1-2 negatif, 3 netral, 4-5 positif
    df["label_num"] = np.select(
        [df["score"] > 3, df["score"] == 3, df["score"] < 3],
        [1, 0, -1],
        default=0
    )
    df["label"] = np.select(
        [df["score"] > 3, df["score"] == 3, df["score"] < 3],
        ["positive", "neutral", "negative"],
        default="neutral"
    )

    print("Text cleaning...")
    df["text_Clean"] = df["content"].progress_apply(preprocess_text)

    print("Normalisasi slang...")
    df["text_Normalization"] = df["text_Clean"].progress_apply(ganti_slang)

    stop_words = set(stopwords.words("indonesian"))
    stop_words.update({"nya", "sih", "nih", "dong", "deh", "lah", "pun"})

    print("Stopword removal...")
    df["text_StopWord"] = df["text_Normalization"].progress_apply(
        lambda x: " ".join(w for w in str(x).split() if w not in stop_words)
    )

    print("Tokenisasi...")
    tokenizer = RegexpTokenizer(r"\w+")
    df["text_Tokenization"] = df["text_StopWord"].progress_apply(tokenizer.tokenize)

    print("Stemming Sastrawi...")
    stemmer = StemmerFactory().create_stemmer()
    df["text_Stemmindo"] = df["text_Tokenization"].progress_apply(
        lambda tokens: [stemmer.stem(token) for token in tokens]
    )

    df["text_String"] = df["text_Stemmindo"].progress_apply(
        lambda tokens: " ".join(token for token in tokens if len(token) > 3)
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n=== SELESAI ===")
    print(f"Data akhir : {len(df)}")
    print(f"Output     : {OUTPUT_FILE}")
    print("\nDistribusi label:")
    print(df["label"].value_counts())
    print("\nContoh:")
    print(df[["content", "score", "label", "text_String"]].head().to_string(index=False))

if __name__ == "__main__":
    main()
