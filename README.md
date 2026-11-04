# sensa-ai (geçici ad)

**Videoda kan ve vahşet içeren bölgeleri 30 saniye önceden bulup, izlemeyi bozmadan hafifçe ve yalnız o bölgede bulanıklaştıran açık kaynak araç.** Kâr amacı yok (D-009).

Neden: ayna dokunuşu sinestezisi olan biri için kanlı bir görüntü fiziksel acı demek. Mevcut araçlar (ClearPlay, VidAngel) sahneyi atlıyor ya da sesi kesiyor; burada amaç filmin akışı sürerken yalnız rahatsız edici bölgeyi yumuşatmak.

> **Durum (3 Kasım 2026):** M0. Repo yapısı, güvenli önizleme, bulanıklaştırma izi biçimi ve vekil hedefle uçtan uca çevrimdışı hat var. Gerçek modeller (M1) henüz eklenmedi; buradaki eşiklerin hiçbiri ölçülmedi.

## Belgeler

| Belge | İçerik |
|---|---|
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Demo adımları M0 → M5, her adımın çıkış kapısı |
| [docs/MODEL_SECIMI.md](docs/MODEL_SECIMI.md) | SigLIP → Grounding DINO / OWLv2 → SAM 2.1 hattı, lisanslar, riskler |
| [docs/TEST_PLANI.md](docs/TEST_PLANI.md) | Kan görmeden test: kim neye bakar, hipotezler H-01…H-06 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Karar defteri |
| [docs/FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md) | Bırakılan yollar (elle etiketli Faster R-CNN) |

## Kurallar

- **Kanlı görüntü gösterilmez (D-001, D-004).** İnsan için diske görüntü yazmanın tek yolu `sensa_ai.safety.write_preview`. Videolar, kareler ve ağırlıklar repoya girmez; CI bunu denetler. Testler yalnız sentetik renkli şekil kullanır.
- Bütün sayılar [config/params.yaml](config/params.yaml)'da; her değerin `status`'ü var: `measured`, `datasheet`, `literature`, `sim`, `estimate`, `target`. Belgeler kopya sayı tutmaz.
- Kaçırma, yanlış alarmdan kötüdür: eşikler önce yakalamaya göre ayarlanır.
- Ölçütü tutmayan bir test, ölçüt değiştirilerek geçirilmez. Deney sonuçları `experiments/results/` altında tarihli.

## Hızlı başlangıç

```bash
pip install -e ".[dev]"
pytest                                                     # kansız sentetik testler
python -m sensa_ai.pipeline analyze video.mp4 --proxy      # vekil hedef: kırmızı alanlar
python -m sensa_ai.pipeline render video.mp4 video.blurtrack.json cikti.mp4
```

## Yapı

```
config/params.yaml        tek sayı kaynağı
config/prompts.yaml       pozitif, negatif ve vekil metin istemleri
src/sensa_ai/
  config.py               params.yaml okuyucu
  safety.py               güvenli önizleme (tek görüntü yazma yolu)
  blurtrack.py            .blurtrack.json: zaman içinde bölge izleri, rampa
  blur.py                 yumuşak kenarlı bölgesel bulanıklık, güvenli mod
  lookahead.py            30 sn ileri bakma tamponu, güvenli mod kararı
  detect.py               Detector arayüzü + modelsiz vekil dedektör
  pipeline.py             çevrimdışı analyze / render
tests/                    sentetik, kansız testler
data/                     yerel veri (repoya girmez), bkz. data/README.md
experiments/results/      tarihli deney sonuçları
archive/                  eski Faster R-CNN kodu, değiştirilmeden
```

## Lisans

MIT. Kullanılan modeller (SigLIP, Grounding DINO, OWLv2, SAM 2) Apache 2.0.
