import os

# ── LLM Sağlayıcı Ayarı ──────────────────────────────────────────────────────
# "claude" veya "gemini" - environment variable ile kontrol edilir
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


LABELS = [
    'akici_ve_surukleyici', 'psikolojik_derinlik', 'duygusal_yogunluk',
    'felsefi', 'toplumsal_elestiri', 'tarihsel', 'karamsar', 'ask',
    'macera', 'bilimkurgu_distopya', 'gizem_polisiye', 'ogretici_farkindalik', 'mizah'
]


ETIKET_ACIKLAMALARI = """
1. akici_ve_surukleyici
   TANIM: Yorumcu kitabı okurken zamanın nasıl geçtiğini fark etmediğini, okumayı bırakamadığını, kurgunun onu sürüklediğini anlatıyorsa bu etiket verilir.
   HAYIR: Kitabı beğenmiş ama okuma hızından veya akıcılığından hiç bahsetmiyorsa verilmez.

2. psikolojik_derinlik
   TANIM: Yorumcu karakterlerin iç dünyasından, psikolojik karmaşıklığından etkilendiğini anlatıyorsa verilir.
   HAYIR: Karakteri sevmiş ama psikolojik boyuttan bahsetmiyorsa verilmez.

3. duygusal_yogunluk
   TANIM: Yorumun genel tonu ve dili duygusal bir etkilenmeyi yansıtıyorsa verilir. Doğrudan söylemesi şart değil.
   HAYIR: Kitabı entelektüel olarak beğenmiş ama duygusal etkilenmeden söz etmiyorsa verilmez.

4. felsefi
   TANIM: Kitap yorumcuyu varoluş, anlam, özgür irade, ahlak, iyilik-kötülük, ölüm, kimlik gibi derin sorular üzerine düşündürmüşse verilir.
   HAYIR: Düşündürücü bulunmuş ama hangi felsefi soruları işlediği belli değilse verilmez.

5. toplumsal_elestiri
   TANIM: Sınıf farkı, yoksulluk, siyasi baskı, eşitsizlik, kapitalizm, ataerkillik gibi toplumsal sorunları ele aldığından bahsediliyorsa verilir.
   HAYIR: Bireysel ahlak eleştirisi var ama toplumsal boyut yoksa verilmez.

6. tarihsel
   TANIM: Kitabın konusu gerçek bir tarihi dönemi, olayı veya figürü işliyorsa ve yorumcu bunu fark edip bahsediyorsa verilir.
   HAYIR: Kitap geçmişte geçiyor ama tarihsel dönem/olay odaklı değilse verilmez.

7. karamsar
   TANIM: Kitabın genel atmosferi karanlık, bunaltıcı veya umutsuzsa ve yorumcu bunu hissettiriyorsa verilir.
   HAYIR: Üzücü sahneler var ama genel ton umut verici ya da katarsisle bitiyorsa verilmez.

8. ask
   TANIM: Romantik aşk kitabın ana temalarından biriyse ve yorumcu bundan bahsediyorsa verilir.
   HAYIR: Karakterler arasında sevgi var ama romantik aşk ana tema değilse verilmez.

9. macera
   TANIM: Kitabın ana kurgusu aksiyon, serüven, yolculuk, hayatta kalma mücadelesi üzerineyse verilir.
   HAYIR: Kitapta olaylar var ama macera/aksiyon odaklı değil; psikolojik gerilim macera sayılmaz.

10. bilimkurgu_distopya
    TANIM: Kitabın türü bilimkurgu veya distopyaysa — gelecek kurgusu, uzay, yapay zeka, totaliter rejim gibi unsurlar taşıyorsa verilir.
    HAYIR: Teknoloji geçiyor ama tür olarak bilimkurgu değilse verilmez.

11. gizem_polisiye
    TANIM: Kitabın ana kurgusu suç, cinayet, gizem veya soruşturma üzerineyse verilir.
    HAYIR: Gerilim var ama polisiye/gizem türünde değilse verilmez.

12. ogretici_farkindalik
    TANIM: Kitap ağırlıklı olarak bilgi aktarımı yapıyorsa — Sapiens, Kozmos, kişisel gelişim tarzı — verilir.
    HAYIR: Roman veya hikaye türünde bir kitap ne kadar düşündürücü olursa olsun bu etiketi ALMAZ.

13. mizah
    TANIM: Kitabın genel tonu komik, esprili veya hiciv doluysa ve yorumcu güldüğünden bahsediyorsa verilir.
    HAYIR: Hafif alaycı ton var ama ağırlıklı tema mizah değilse verilmez.
"""

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
    """LLM'e gönderilecek analiz prompt'unu oluşturur.

    BERTurk eğitim verisinin etiketlenmesinde kullanılan prompt ile
    aynı kriterleri ve formatı kullanır (tutarlılık için).
    """
    json_skeleton = ",\n".join(
        f'  "{label}": 0.0' for label in LABELS
    )

    return f"""Sen bir Türkçe edebiyat ve kitap yorumu analisti olarak görev yapıyorsun.

Aşağıdaki kitap yorumunu dikkatle oku. Verilen 13 etiketten hangilerinin uygun olduğunu belirle ve her etiket için 0.0 ile 1.0 arasında güven skoru ver.

ÖNEMLİ: Tek tek anahtar kelimelere değil, yorumun GENEL BAĞLAMINA ve TONUNA bak.

ETİKET TANIMLARI:
{ETIKET_ACIKLAMALARI}

PUANLAMA REHBERİ:
- 0.0 - 0.4 : Bu etiket bu yoruma uymuyor
- 0.5 - 0.7 : Belki uyuyor ama emin değilim
- 0.8 - 1.0 : Bu etiket bu yoruma açıkça uyuyor

Kitap Yorumu:
\"\"\"
{text}
\"\"\"

SADECE şu JSON formatında yanıt ver, başka hiçbir açıklama, markdown işareti veya ek metin ekleme:
{{
{json_skeleton}
}}"""