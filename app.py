import html
import re

import joblib
import streamlit as st

SLANG = {
    "brg": "barang", "bgt": "banget", "sdh": "sudah", "udh": "sudah",
    "dr": "dari", "yg": "yang", "tp": "tapi", "gk": "tidak", "ga": "tidak",
    "nga": "tidak", "gak": "tidak", "tdk": "tidak", "krn": "karena",
    "jg": "juga", "dgn": "dengan", "cpt": "cepat", "kwalitas": "kualitas",
    "sy": "saya", "blm": "belum", "garcep": "gerak cepat",
}


def clean_text(t):
    t = html.unescape(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    words = [SLANG.get(w, w) for w in t.split()]
    return " ".join(words[:200])


@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")


model = load_model()

st.title("Analisis Sentimen Review Tokopedia")
st.write(
    "Tulis review produk berbahasa Indonesia, lalu model akan memperkirakan "
    "apakah sentimennya positif, netral, atau negatif."
)

text = st.text_area("Review produk:", height=120,
                    placeholder="Contoh: barang bagus tapi pengiriman lama banget")

if st.button("Analisis"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Tulis review dengan kata-kata dulu ya.")
    else:
        probs = dict(zip(model.classes_, model.predict_proba([cleaned])[0]))
        label = max(probs, key=probs.get)
        st.subheader(f"Prediksi: {label}")
        for name in ["positive", "neutral", "negative"]:
            st.write(f"{name}: {probs[name]:.1%}")
            st.progress(float(probs[name]))

st.caption(
    "Catatan: label dataset berasal dari rating (1-2 negatif, 3 netral, 4-5 positif), "
    "jadi prediksi netral paling sulit dan paling tidak pasti."
)