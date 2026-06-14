import os

# ── LLM Sağlayıcı Ayarı ──────────────────────────────────────────────────────
# "claude" veya "gemini" - environment variable ile kontrol edilir
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Etiketler ve Açıklamaları ────────────────────────────────────────────────
# LLM'e gönderilecek prompt'ta her etiketin ne anlama geldiğini açıklıyoruz
LABEL_DESCRIPTIONS = {
    "akici_ve_surukleyici": "Kitap akıcı, sürükleyici, okuyucuyu içine çeken bir anlatıma sahip",
    "psikolojik_derinlik": "Karakterlerin iç dünyası, ruhsal durumları derinlemesine işleniyor",
    "duygusal_yogunluk": "Güçlü duygular uyandırıyor, okuyucuyu duygusal olarak etkiliyor",
    "felsefi": "Varoluş, anlam, etik gibi felsefi sorgulamalar içeriyor",
    "toplumsal_elestiri": "Toplumsal sorunlara, adaletsizliklere veya sisteme eleştiri içeriyor",
    "tarihsel": "Tarihi bir dönem veya olayları konu alıyor",
    "karamsar": "Karamsar, hüzünlü veya umutsuz bir atmosfere sahip",
    "ask": "Aşk, romantizm veya ilişkiler temalı",
    "macera": "Macera, aksiyon veya heyecan dolu olaylar içeriyor",
    "bilimkurgu_distopya": "Bilim kurgu öğeleri veya distopik bir dünya tasviri içeriyor",
    "gizem_polisiye": "Gizem, sır, polisiye veya dedektiflik öğeleri içeriyor",
    "ogretici_farkindalik": "Okuyucuya bir konuda bilgi veriyor veya farkındalık kazandırıyor",
    "mizah": "Mizahi, eğlenceli veya güldürücü bir üslup içeriyor",
}

LABELS = list(LABEL_DESCRIPTIONS.keys())

# ── Eşik Değerleri ────────────────────────────────────────────────────────────
# LLM'den dönen confidence skoru bu eşiğin üzerindeyse etiket "detected" sayılır
THRESHOLDS = {
    "akici_ve_surukleyici": 0.30,
    "psikolojik_derinlik": 0.45,
    "duygusal_yogunluk": 0.40,
    "felsefi": 0.45,
    "toplumsal_elestiri": 0.35,
    "tarihsel": 0.60,
    "karamsar": 0.30,
    "ask": 0.30,
    "macera": 0.35,
    "bilimkurgu_distopya": 0.45,
    "gizem_polisiye": 0.30,
    "ogretici_farkindalik": 0.30,
    "mizah": 0.30,
}


def build_prompt(text: str) -> str:
    """LLM'e gönderilecek analiz prompt'unu oluşturur."""
    label_lines = "\n".join(
        f'- "{label}": {desc}' for label, desc in LABEL_DESCRIPTIONS.items()
    )

    return f"""Aşağıda bir kitap yorumu var. Bu yorumu analiz ederek, her etiketin yorumda
ne ölçüde mevcut olduğunu 0.0 ile 1.0 arasında bir skor olarak değerlendir.

Etiketler ve anlamları:
{label_lines}

Kitap Yorumu:
\"\"\"
{text}
\"\"\"

Yanıtını SADECE aşağıdaki formatta, geçerli bir JSON objesi olarak ver.
Başka hiçbir açıklama, markdown işareti veya ek metin ekleme:

{{
{", ".join(f'"{label}": <0.0-1.0 arası sayı>' for label in LABELS)}
}}"""
