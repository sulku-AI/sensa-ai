# sensa-ai (geçici ad)

**Videoda kan ve vahşet içeren bölgeleri 30 saniye önceden bulup, izlemeyi bozmadan hafifçe ve yalnız o bölgede bulanıklaştıran açık kaynak araç.** Kâr amacı yok (D-009).

Neden: ayna dokunuşu sinestezisi olan biri için kanlı bir görüntü fiziksel acı demek. Mevcut araçlar (ClearPlay, VidAngel) sahneyi atlıyor ya da sesi kesiyor; burada amaç filmin akışı sürerken yalnız rahatsız edici bölgeyi yumuşatmak.

> **Durum (4 Kasım 2026):** M0 tamam; M1-M3 kodu yazıldı (model sarmalayıcıları, takip, değerlendirme, otomatik etiketleme, damıtma eğitimi, 30 sn ileri bakan canlı oynatıcı). Gerçek ağırlıklarla **hiçbir şey ölçülmedi**: ilk ölçüm M5'te, adımlar [docs/M5_KOMUTLAR.md](docs/M5_KOMUTLAR.md).

## Belgeler

| Belge | İçerik |
|---|---|
| [docs/M5_KOMUTLAR.md](docs/M5_KOMUTLAR.md) | **MacBook Air M5'te sırayla çalıştırılacak komutlar** |
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Demo adımları M0 → M5, her adımın çıkış kapısı |
| [docs/MODEL_ARASTIRMA.md](docs/MODEL_ARASTIRMA.md) | Model ve veri seti araştırması, lisanslar, kaynaklar |
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
python -m sensa_ai.player video.mp4 --proxy                # canlı oynatıcı, 30 sn ileri bakma
```

Gerçek modeller: `pip install -e ".[models]"`, sonra `--proxy` olmadan. Ayrıntı: [docs/M5_KOMUTLAR.md](docs/M5_KOMUTLAR.md).

## Yapı

```
config/params.yaml        tek sayı kaynağı
config/prompts.yaml       pozitif, negatif ve vekil metin istemleri
config/models.yaml        Hugging Face model kimlikleri (hepsi Apache 2.0)
src/sensa_ai/
  config.py               params.yaml okuyucu
  safety.py               güvenli önizleme + kilitli veri klasörü (tek görüntü yazma yolu)
  blurtrack.py            .blurtrack.json: zaman içinde bölge izleri, rampa
  blur.py                 yumuşak kenarlı bölgesel bulanıklık, güvenli mod
  lookahead.py            30 sn ileri bakma tamponu, güvenli mod kararı
  detect.py               Detector arayüzü + modelsiz vekil dedektör
  models.py               SigLIP kapı, LLMDet / Grounding DINO / OWLv2 metinle bölge
  tracking.py             SAM 2.1 / EdgeTAM maske takibi
  metrics.py              kapsama ve yanlış alarm ölçütleri (yalnız sayı)
  pipeline.py             çevrimdışı analyze / render
  player.py               canlı oynatıcı: analiz iş parçacığı + bulanıklık
tools/                    bench, evaluate, autolabel, fetch_datasets, fetch_negatives,
                          train_rfdetr, make_proxy_video
tests/                    sentetik, kansız testler (model testleri rastgele ağırlıkla, indirmesiz)
data/                     yerel veri (repoya girmez), bkz. data/README.md
experiments/results/      tarihli deney sonuçları
archive/                  eski Faster R-CNN kodu, değiştirilmeden
```

## Lisans

MIT. Kullanılan modeller (SigLIP, LLMDet, Grounding DINO, OWLv2, SAM 2.1, EdgeTAM, RF-DETR-Seg) Apache 2.0.
