import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os
import csv

w2v_models = [
    ("Lemmatized CBOW Win2 Dim100", "lemmatized_model_cbow_window2_dim100.model"),
    ("Lemmatized CBOW Win2 Dim300", "lemmatized_model_cbow_window2_dim300.model"),
    ("Lemmatized CBOW Win4 Dim100", "lemmatized_model_cbow_window4_dim100.model"),
    ("Lemmatized CBOW Win4 Dim300", "lemmatized_model_cbow_window4_dim300.model"),
    ("Lemmatized Skipgram Win2 Dim100", "lemmatized_model_skipgram_window2_dim100.model"),
    ("Lemmatized Skipgram Win2 Dim300", "lemmatized_model_skipgram_window2_dim300.model"),
    ("Lemmatized Skipgram Win4 Dim100", "lemmatized_model_skipgram_window4_dim100.model"),
    ("Lemmatized Skipgram Win4 Dim300", "lemmatized_model_skipgram_window4_dim300.model"),
    ("Stemmed CBOW Win2 Dim100", "stemmed_model_cbow_window2_dim100.model"),
    ("Stemmed CBOW Win2 Dim300", "stemmed_model_cbow_window2_dim300.model"),
    ("Stemmed CBOW Win4 Dim100", "stemmed_model_cbow_window4_dim100.model"),
    ("Stemmed CBOW Win4 Dim300", "stemmed_model_cbow_window4_dim300.model"),
    ("Stemmed Skipgram Win2 Dim100", "stemmed_model_skipgram_window2_dim100.model"),
    ("Stemmed Skipgram Win2 Dim300", "stemmed_model_skipgram_window2_dim300.model"),
    ("Stemmed Skipgram Win4 Dim100", "stemmed_model_skipgram_window4_dim100.model"),
    ("Stemmed Skipgram Win4 Dim300", "stemmed_model_skipgram_window4_dim300.model"),
]

df_lemma = pd.read_csv("lemmatized.csv")
df_stem = pd.read_csv("stemmed.csv")

lemma_sentences = df_lemma["original_sentence"].astype(str).tolist()
stem_sentences = df_stem["original_sentence"].astype(str).tolist()

# processed_tokens sütunu kullanılacak
lemma_texts = df_lemma["processed_tokens"].astype(str).tolist()
stem_texts = df_stem["processed_tokens"].astype(str).tolist()

# Giriş metni olarak ilk cümle
input_text = lemma_texts[0]
print("Giriş metni:\n", input_text)

def tfidf_top5(input_text, texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    input_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vec, tfidf_matrix)[0]
    top5_idx = similarities.argsort()[-6:][::-1][1:]
    return [(idx, texts[idx], similarities[idx]) for idx in top5_idx]

def get_sentence_vector(model, sentence):
    words = word_tokenize(sentence)
    vectors = [model.wv[w] for w in words if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def w2v_top5(input_text, texts, model_path):
    model = Word2Vec.load(model_path)
    input_vec = get_sentence_vector(model, input_text).reshape(1, -1)
    sentence_vecs = np.array([get_sentence_vector(model, s) for s in texts])
    similarities = cosine_similarity(input_vec, sentence_vecs)[0]
    top5_idx = similarities.argsort()[-5:][::-1]
    return [(idx, texts[idx], similarities[idx]) for idx in top5_idx]

results = []

tfidf_lemma = tfidf_top5(input_text, lemma_texts)
results.append(set(idx for idx, _, _ in tfidf_lemma))

tfidf_stem = tfidf_top5(input_text, stem_texts)
results.append(set(idx for idx, _, _ in tfidf_stem))

# Word2Vec Modelleri
for name, path in w2v_models:
    if not os.path.exists(path):
        print(f"Model bulunamadı: {path}")
        results.append(set())
        continue
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_texts, path)
    else:
        top5 = w2v_top5(input_text, stem_texts, path)
    results.append(set(idx for idx, _, _ in top5))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

n = len(results)
jaccard_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        jaccard_matrix[i, j] = jaccard(results[i], results[j])

print("Jaccard Benzerlik Matrisi:")
print(jaccard_matrix)

print("\nTF-IDF (Lemmatized) ilk 5:")
for idx, metin, skor in tfidf_lemma:
    print(f"{idx}: {skor:.4f} - {lemma_sentences[idx][:80]}...")

print("\nTF-IDF (Stemmed) ilk 5:")
for idx, metin, skor in tfidf_stem:
    print(f"{idx}: {skor:.4f} - {stem_sentences[idx][:80]}...")

model_names = [
    "TF-IDF (Lemmatized)",
    "TF-IDF (Stemmed)",
    *[name for name, _ in w2v_models]
]

jaccard_df = pd.DataFrame(
    jaccard_matrix,
    index=model_names,
    columns=model_names
)

jaccard_df.to_csv("jaccard_similarity_matrix.csv", float_format="%.4f")

# Jaccard matrisini HTML olarak da kaydet
jaccard_df.to_html("jaccard_similarity_matrix.html", float_format="%.4f")

print("Jaccard benzerlik matrisi 'jaccard_similarity_matrix.csv' ve 'jaccard_similarity_matrix.html' dosyasına kaydedildi.")
for i, (name, path) in enumerate(w2v_models):
    print(f"\nWord2Vec {name} ilk 5:")
    if not os.path.exists(path):
        print("Model bulunamadı.")
        continue
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_texts, path)
        for idx, metin, skor in top5:
            print(f"{idx}: {skor:.4f} - {lemma_sentences[idx][:80]}...")
    else:
        top5 = w2v_top5(input_text, stem_texts, path)
        for idx, metin, skor in top5:
            print(f"{idx}: {skor:.4f} - {stem_sentences[idx][:80]}...")

def log_top5_to_csv_and_html(tfidf_lemma, tfidf_stem, w2v_models, lemma_sentences, stem_sentences, all_w2v_top5, input_text):
    rows = []
    with open("benzerlik_top5.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Sıra", "Index", "Skor", "Metin"])
        for i, (idx, _, skor) in enumerate(tfidf_lemma, 1):
            row = ["TF-IDF (Lemmatized)", i, idx, f"{skor:.4f}", lemma_sentences[idx][:120]]
            writer.writerow(row)
            rows.append(row)
        for i, (idx, _, skor) in enumerate(tfidf_stem, 1):
            row = ["TF-IDF (Stemmed)", i, idx, f"{skor:.4f}", stem_sentences[idx][:120]]
            writer.writerow(row)
            rows.append(row)
        for (model_name, _), top5 in zip(w2v_models, all_w2v_top5):
            for i, (idx, _, skor) in enumerate(top5, 1):
                metin = lemma_sentences[idx][:120] if "lemmatized" in model_name.lower() else stem_sentences[idx][:120]
                row = [model_name, i, idx, f"{skor:.4f}", metin]
                writer.writerow(row)
                rows.append(row)
    # Modern HTML çıktısı
    df = pd.DataFrame(rows, columns=["Model", "Sıra", "Index", "Skor", "Metin"])
    table_html = df.to_html(index=False, classes="table table-striped table-bordered table-hover", border=0, escape=False)
    html = f"""
    <!DOCTYPE html>
    <html lang='tr'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <title>Benzerlik Top 5 Sonuçları</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style>
            body {{ background: #f8f9fa; }}
            .container {{ margin-top: 40px; }}
            h1 {{ margin-bottom: 30px; }}
            table {{ background: white; }}
            thead th {{ background: #343a40; color: #fff; }}
        </style>
    </head>
    <body>
        <div class='container'>
            <h1>Benzerlik Top 5 Sonuçları</h1>
            {table_html}
        </div>
    </body>
    </html>
    """
    with open("benzerlik_top5.html", "w", encoding="utf-8") as f:
        f.write(html)

all_w2v_top5 = []
for name, path in w2v_models:
    if not os.path.exists(path):
        all_w2v_top5.append([])
        continue
    if "lemmatized" in path:
        top5 = w2v_top5(input_text, lemma_texts, path)
    else:
        top5 = w2v_top5(input_text, stem_texts, path)
    all_w2v_top5.append(top5)

log_top5_to_csv_and_html(tfidf_lemma, tfidf_stem, w2v_models, lemma_sentences, stem_sentences, all_w2v_top5, input_text) 