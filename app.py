import os
import re
import psycopg2
import requests
import joblib
import numpy as np
from flask import Flask, render_template, request
from gensim.models import Word2Vec
import nltk
from bs4 import BeautifulSoup

# --- NLTK BULUT AYARI ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

# --- SİSTEM AYARLARI ---
NEWSDATA_API_KEY = "pub_43034969efcc5b0267f56cf8f5413df18b955"
POSTGRES_URI = "postgresql://postgres.pgzgqtzrvbzxlbdkrmyq:meoBKAAQ8aywkn83@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# --- MODELLERİN HAFIZAYA ALINMASI ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    w2v_model = Word2Vec.load(os.path.join(BASE_DIR, "word2vec_teyit.model"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_final.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    print(f"✅ Modeller başarıyla bağlandı. Sınıflar: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")

# --- METİN ÖN İŞLEME FONKSİYONU ---
def temizle_ve_vektorize_et(text, model, vector_size=100):
    if not text or str(text).strip() == "":
        return np.zeros(vector_size)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('turkish'))
    kelimeler = [k for k in text.split() if k not in stop_words]
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    if len(vektorler) > 0:
        return np.mean(vektorler, axis=0)
    else:
        return np.zeros(vector_size)

# --- ETİKET OKUMA YARDIMCI FONKSİYONU ---
def etiketten_skor_cikart(bulunan_muhur):
    bulunan_muhur = bulunan_muhur.lower().strip()

    if bulunan_muhur in ["yanlış", "yanlis", "yanliş", "yalan", "uydurma", "asılsız"]:
        return 0.0
    elif bulunan_muhur in ["doğru", "dogru", "gerçek", "gercek", "onaylandı"]:
        return 100.0
    elif bulunan_muhur in ["kısmen doğru", "yanıltıcı", "karma"]:
        return 35.0
    else:
        print(f"   Bilinmeyen mühür: '{bulunan_muhur}'")
        return 40.0

# --- TEYİT.ORG SAYFA OKUMA ---
# --- TEYİT.ORG SAYFA OKUMA ---
def teyit_sayfasini_oku(link):
    try:
        r = requests.get(link, headers=HEADERS, timeout=6)
        print(f"   Teyit HTTP status: {r.status_code}")
        if r.status_code != 200:
            return None, False
        soup = BeautifulSoup(r.text, 'html.parser')

        # Kaç tane text-uppercase span var?
        elementler = soup.find_all("span", class_="text-uppercase")
        print(f"   text-uppercase span sayısı: {len(elementler)}")
        for el in elementler:
            print(f"   → '{el.get_text(strip=True)}'")

        # Ham HTML'in ilk 2000 karakterini de logla
        print(f"   HTML önizleme: {r.text[:2000]}")

        element = soup.find("span", class_="text-uppercase")
        if element:
            muhur = element.get_text(strip=True).lower()
            print(f"   Teyit mühürü: '{muhur}'")
            return etiketten_skor_cikart(muhur), True
        return 40.0, True
    except Exception as e:
        print(f"Teyit sayfa okuma hatası: {e}")
        return None, False

# --- TEYİT.ORG ARŞİV ARAMA (Newsdata bulamazsa devreye girer) ---
def teyit_org_arsiv_ara(haber_basligi):
    """
    Teyit.org'un kendi arşivinde haberi arar.
    Önce WP REST API, sonra ?s= scraping dener.
    """
    sorgu = requests.utils.quote(haber_basligi)

    # Yöntem 1: WordPress REST API
    try:
        api_url = f"https://teyit.org/wp-json/wp/v2/posts?search={sorgu}&per_page=3&_fields=link,title"
        r = requests.get(api_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            posts = r.json()
            if posts and isinstance(posts, list):
                link = posts[0].get("link", "")
                print(f"✅ Teyit WP-API: {link}")
                skor, basarili = teyit_sayfasini_oku(link)
                if basarili:
                    return skor, f"Teyit.org Arşiv (WP-API)"
    except Exception as e:
        print(f"Teyit WP-API hatası: {e}")

    # Yöntem 2: Arama sayfası scraping
    try:
        arama_url = f"https://teyit.org/?s={sorgu}"
        r = requests.get(arama_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            link_el = soup.find("a", href=lambda h: h and "/analiz/" in h)
            if link_el:
                link = link_el["href"]
                print(f"✅ Teyit Arama Scrape: {link}")
                skor, basarili = teyit_sayfasini_oku(link)
                if basarili:
                    return skor, "Teyit.org Arşiv (Arama)"
    except Exception as e:
        print(f"Teyit arama scrape hatası: {e}")

    return None, "Teyit.org'da bulunamadı"

# --- DOĞRULUK PAYI SCRAPING ---
def dogrulukpayi_tara(kaynak_linki):
    try:
        r = requests.get(kaynak_linki, headers=HEADERS, timeout=6)
        if r.status_code != 200:
            return None, False
        soup = BeautifulSoup(r.text, 'html.parser')
        element = soup.find("div", class_="user-content")
        if element:
            muhur = element.get_text(strip=True).lower()[:50]
            print(f"   Doğruluk Payı mühürü: '{muhur}'")
            return etiketten_skor_cikart(muhur), True
        return 40.0, True
    except Exception as e:
        print(f"Doğruluk Payı scrape hatası: {e}")
        return None, False

# ==========================================================
# ANA NEWSDATA + MEDYA TARAMA FONKSİYONU
# ==========================================================
def medya_tara(haber_basligi):
    """
    Döndürdüğü değer: (api_skoru: float | None, kaynak_metni: str)
    Görseldeki akış şemasıyla %100 uyumlu çalışan iki döngülü öncelik yapısı.
    """
    guvenilir_kaynaklar = [
        "aa.com.tr", "trthaber.com", "bbc.com", "bbc.com/turkce",
        "sozcu.com.tr", "hurriyet.com.tr", "milliyet.com.tr",
        "haberturk.com", "ntv.com.tr", "cumhuriyet.com.tr"
    ]
    dogrulama_platformlari = ["teyit.org", "dogrulukpayi.com"]

    # --- ADIM 1: Newsdata API'den Sonuçları Çek ---
    sonuclar = []
    try:
        url = f"https://newsdata.io/api/1/latest?apikey={NEWSDATA_API_KEY}&q={requests.utils.quote(haber_basligi)}&language=tr"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            sonuclar = response.json().get("results", [])
            print(f"   Newsdata: {len(sonuclar)} sonuç bulundu")
    except Exception as e:
        print(f"Newsdata API hatası: {e}")

    # --- ADIM 2: Sonuçları İncele ---
    aranan_baslik = haber_basligi.lower()

    # 2a) --- Önce doğrulama platformlarını tara (Mutlak Öncelik) ---
    for sonuc in sonuclar:
        kaynak_linki = sonuc.get("link", "").lower()
        if any(domain in kaynak_linki for domain in dogrulama_platformlari):
            print(f"   Doğrulama platformu bulundu: {kaynak_linki}")
            if "teyit.org" in kaynak_linki:
                skor, basarili = teyit_sayfasini_oku(kaynak_linki)
            else:
                skor, basarili = dogrulukpayi_tara(kaynak_linki)
            if basarili:
                return skor, "Canlı Analiz (Hibrit: NLP + Doğrulama Platformu)"

    # 2b) --- Sonra güvenilir medyaya bak (Eğer doğrulama platformu yoksa) ---
    for sonuc in sonuclar:
        kaynak_linki = sonuc.get("link", "").lower()
        bulunan_baslik = sonuc.get("title", "").lower()
        ortak_kelimeler = set(bulunan_baslik.split()) & set(aranan_baslik.split())
        if any(domain in kaynak_linki for domain in guvenilir_kaynaklar) and len(ortak_kelimeler) >= 2:
            print(f"   Güvenilir medya bulundu: {kaynak_linki}")
            return 100.0, "Canlı Analiz (Hibrit: NLP + Güvenilir Medya)"

    # --- ADIM 3: Newsdata'da hiçbir şey bulunamazsa teyit.org arşivini doğrudan ara ---
    print("   Newsdata'da doğrulama platformu bulunamadı, Teyit.org arşivi aranıyor...")
    teyit_skor, teyit_aciklama = teyit_org_arsiv_ara(haber_basligi)
    if teyit_skor is not None:
        return teyit_skor, f"Canlı Analiz (Hibrit: NLP + {teyit_aciklama})"

    # --- ADIM 4: Hiçbir şey bulunamadıysa (API = None) ---
    if sonuclar:
        return None, "Canlı Analiz (Hibrit: NLP + Genel Medya - Eşleşme Yok)"
    else:
        return None, "Canlı Analiz (Hibrit: NLP + Medyada Bulunamadı)"


# --- ANA SAYFA VE ANALİZ MOTORU ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    haber_basligi = request.form.get("title", "").strip()
    haber_detayi = request.form.get("content", "").strip()

    if not haber_basligi:
        return render_template("index.html", error="Lütfen en azından bir haber başlığı girin.")

    # --- ADIM A: POSTGRESQL ÖNBELLEK KONTROLÜ ---
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(POSTGRES_URI, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_skoru, api_skoru, final_dogruluk_yuzdesi, karar FROM arama_gecmisi WHERE haber_basligi = %s",
            (haber_basligi,)
        )
        kayit = cursor.fetchone()
        if kayit:
            cursor.close()
            conn.close()
            return render_template(
                "index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi,
                model_skoru=kayit[0], api_skoru=kayit[1], final_skor=kayit[2], karar=kayit[3],
                kaynak="Bulut Veritabanı (Önbellek)"
            )
    except Exception as db_err:
        print(f"⚠️ Önbellek okuma hatası: {db_err}")
        conn = None
        cursor = None

    # --- ADIM B: MAKİNE ÖĞRENMESİ TAHMİNİ ---
    tam_metin = haber_basligi + " " + haber_detayi
    metin_vektoru = temizle_ve_vektorize_et(tam_metin, w2v_model).reshape(1, -1)
    olasiliklar = rf_model.predict_proba(metin_vektoru)[0]

    dogru_indeksi = None
    for i, sinif in enumerate(label_encoder.classes_):
        if str(sinif).lower() in ['doğru', 'dogru', 'true', 'gerçek', 'gercek']:
            dogru_indeksi = i
            break
    if dogru_indeksi is not None:
        model_skoru = round(float(olasiliklar[dogru_indeksi]) * 100, 2)
    else:
        model_skoru = round(float(np.max(olasiliklar)) * 100, 2)

    # --- ADIM C: MEDYA TARAMA (Senin yeni iki döngülü akışın tetikleniyor) ---
    api_skoru, kaynak_metni = medya_tara(haber_basligi)

    # --- ADIM D: NİHAİ HİBRİT KARAR ---
    if api_skoru is None:
        # Hiçbir kaynakta bulunamadı → sadece NLP (%100 ML ağırlığı)
        final_skor = round(model_skoru, 2)
        kaynak_metni = kaynak_metni.replace("Hibrit: NLP + ", "Sadece NLP - ")
    else:
        # %50 NLP + %50 API Dengeli Hibrit Formül
        final_skor = round((model_skoru * 0.50) + (api_skoru * 0.50), 2)

    if final_skor >= 70:
        karar = "GÜVENİLİR / DOĞRULANMIŞ HABER"
    elif final_skor >= 45:
        karar = "ŞÜPHELİ / KANIT YETERSİZ"
    else:
        karar = "YANLIŞ / DEZENFORMASYON"

    api_skoru_goster = api_skoru if api_skoru is not None else 0.0

    # --- ADIM E: POSTGRESQL'E KAYDET ---
    if conn and cursor:
        try:
            cursor.execute(
                "INSERT INTO arama_gecmisi (haber_basligi, haber_detayi, model_skoru, api_skoru, final_dogruluk_yuzdesi, karar) VALUES (%s, %s, %s, %s, %s, %s)",
                (haber_basligi, haber_detayi, model_skoru, api_skoru_goster, final_skor, karar)
            )
            conn.commit()
        except Exception as save_err:
            print(f"Veritabanı kaydetme hatası: {save_err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "index.html", sonuc=True, baslik=haber_basligi, detay=haber_detayi,
        model_skoru=model_skoru, api_skoru=api_skoru_goster, final_skor=final_skor,
        karar=karar, kaynak=kaynak_metni
    )

if __name__ == "__main__":
    app.run(debug=True)
