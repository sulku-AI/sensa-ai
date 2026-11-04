# Model ve veri araştırması (2026-11-04)

Amaç: kanlı bölgeleri **kimse görüntü görmeden** bulup takip edebilecek açık modelleri, lisanslarını ve Emirhan'ın donanımına (RTX 2060 6 GB, MacBook Air M5) uygunluğunu çıkarmak. Sayılar kaynakların bildirdiği değerler, **bizim tarafımızdan ölçülmedi** (`status: literature`). Ölçüm `tools/bench.py` ile yapılacak.

## 1. Özet karar

| Görev | 1. tercih | Yedek | Neden |
|---|---|---|---|
| Kare kapısı | SigLIP (base) | CLIP ViT-B/32 | Ucuz, sıfır atış, pozitif ve negatif istemleri aynı anda karşılaştırır |
| Metinle bölge | **LLMDet tiny/base** | OWLv2 base, Grounding DINO tiny | LLMDet: Apache 2.0, transformers'ta, MM-Grounding-DINO ile aynı arayüz, LVIS'te GD'den güçlü [3] |
| Maske takibi | **SAM 2.1 tiny** | EdgeTAM | İkisi de Apache 2.0. EdgeTAM SAM 2'nin cihaz üstü hafif sürümü [6], M5 için aday |
| Damıtma (hızlı model) | **RF-DETR-Seg Nano/Small** | — | Apache 2.0 bütün segmentasyon boyutlarında [4]; YOLOv8-seg (AGPL) yerine |
| Öğretmen (yalnız çevrimdışı) | SAM 3 | — | Metinle maske + takip tek modelde [5], ama lisansı açık kaynak değil ve 6 GB'a sığmaz (aşağıda) |

Bu seçim `docs/MODEL_SECIMI.md`'deki hattı değiştirmez, B aşamasının ilk adayını LLMDet yapar (D-011).

## 2. Metinle bölge bulan dedektörler

| Model | Lisans | Bildirilen başarım | Not |
|---|---|---|---|
| Grounding DINO 1.0 (tiny/base) | Apache 2.0 | COCO sıfır atış 52.5 AP; A100'de ~100-200 ms/görüntü [1] | Otomatik etiketleme için uygun, 30 fps için yavaş |
| Grounding DINO 1.5 / 1.6 / DINO-X | Yalnız ücretli API | — | Kullanılamaz: veri dışarı gider, açık değil [1] |
| **LLMDet** (tiny/base/large) | Apache 2.0 | LVIS minival AP 50.7 / 54.3 / 56.6 [3] | transformers `zero-shot-object-detection`; çıkarım MM-Grounding-DINO ile aynı |
| OWLv2 | Apache 2.0 | ~30-40 AP sıfır atış; T4'te ~50-80 ms [1] | Kısa isim istemleriyle iyi; basit ince ayar |
| Florence-2 (base 230M / large 770M) | MIT | ~34.7 mAP COCO; ~862 MB VRAM [1] | Kutu başına güven skoru yok: eşik ayarı zor, ikinci planda |
| YOLOE / YOLO-World | AGPL-3.0 | Gerçek zamanlı [1] | **Kullanılmaz** (D-006, D-009) |

Kan şekilsiz bir "madde"; bu modellerin hepsi nesne kutusu için eğitildi. Bu yüzden H-01 asıl risk. 2025'te zararlı içerik için yayımlanan sıfır atış bir çalışma (SAM maskeleri + VLM puanlama) öğe düzeyinde %85.8 yakalama ve %78.1 kesinlik bildiriyor; doğrudan VLM kutularıyla yakalama %59.4'te kalmış [7]. Yani salt metin istemli kutu yetmeyebilir, maske + puanlama birleşimi daha iyi.

## 3. Maske ve takip

| Model | Lisans | Boyut / hız | 2060 (6 GB) | M5 |
|---|---|---|---|---|
| **SAM 2.1** tiny/small | Apache 2.0 | tiny en küçük; video için bellek modülü [2] | Uygun (tahmin) | MPS ile mümkün, yavaş olabilir |
| **EdgeTAM** | Apache 2.0 | SAM 2'nin cihaz üstü hafif sürümü, RepViT omurga [6] | Uygun | **En iyi aday** |
| SAM 3 | SAM License (Meta, 2025-11-19) | ~840M parametre, ~3.4 GB; H200'de ~30 ms; 16 GB GPU'ya sığıyor [5] | **Sığmaz** | Sığmaz (pratikte) |

**SAM 3 lisansı:** ScanCode "Proprietary Free" sınıflıyor: açık kaynak değil; dağıtırken sözleşme eklenmeli; tersine mühendislik yasağı, kullanım kısıtları, tazminat maddesi, Meta'nın sözleşmeyi değiştirebilmesi var [8]. Kodumuza bağımlılık olarak girmez. Ama SAM 3'ün ürettiği etiketlerle kendi modelimizi eğitmek mümkün olabilir; bu, lisansın tam metniyle ayrıca kontrol edilecek (açık soru).

## 4. Önceki çalışma: aynı fikir, 2025

"Segmentation-Based Blood Blurring" (Sensors 25(7), 2093, 2025) [9]: tarayıcı eklentisinde kanlı bölgeyi segmentasyonla bulanıklaştırıp izleyicinin göz tepkisini ölçmüşler. Bizim için önemli olanlar:
- **YOLOv8s-seg**, üç açık kan segmentasyon veri setinin birleşimiyle (toplam **4,787 görüntü**, kırmızı giysi/meyve gibi negatifler dahil) ince ayarlanmış.
- Maske başarımı: kesinlik 0.819, **yakalama 0.655**, mAP50 0.718. Yazarlar bulanıklığın "tam doğru olmadığını" söylüyor.
- Hız ölçümü yok. Kod paylaşılmamış görünüyor. Zenodo verisi göz ölçümleri ve anketler (görüntü veri seti değil).

Çıkarım: denetimli tek bir küçük model ~%65 yakalamada kalıyor; bizim hedef %95 (`requirements.event_recall`). Fark ileri bakma (30 sn ile daha büyük model çalıştırabiliriz), takip (SAM 2 kaçan kareleri doldurur) ve eşiği yakalama lehine ayarlamaktan gelmeli. Bu paper'ın kendisi değerlendirmede referans taban çizgisi (baseline) olur.

## 5. Veri setleri (görüntüye bakılmadı; yalnız sayfa metni)

| Kaynak | İçerik | Lisans | Durum |
|---|---|---|---|
| Roboflow Universe: `joost-bos-rfw3p/blood-segmentation-y0ygo` | 1 sınıf "blood", segmentasyon | CC BY 4.0 [10] | Görüntü sayısı sayfada yok |
| Roboflow Universe: `backupgeneral/blood_segmentation_2` | segmentasyon | CC BY 4.0 [11] | Sayı ve sınıf sayfada yok |
| Roboflow Universe: `angel-macwan-6m3sx/blood-segmentation` | segmentasyon | **Lisans yazmıyor** [12] | Kullanma (lisans belirsiz) |
| HF: `NeuralShell/Gore-Blood-Dataset-v1.0` | <1K, Stable Diffusion ile üretilmiş | MIT [13] | Sentetik; etiket biçimi belirsiz. Gerçek video dağılımını temsil etmez, ek veri olabilir |
| VHD11k, DRUNalia vb. (zararlı içerik setleri) | Sınıflandırma ağırlıklı | Belirsiz / erişim onaylı [7] | Değerlendirme için aday, lisans kontrolü gerekli |

**Emirhan için uyarı:** Roboflow Universe ve Hugging Face veri seti sayfaları örnek görüntüleri **küçük resim olarak gösterir**. Bu sayfaları açma. İndirme `tools/fetch_datasets.py` ile, görüntüler doğrudan `data/pozitif/` içine yazılır ve hiçbir pencerede açılmaz. API anahtarı Roboflow hesabının Settings → API Keys sayfasından alınır (o sayfada görüntü yok).

Negatif set (Emirhan izleyebilir) Open Images / COCO'dan sınıf adına göre seçilir: domates, çilek, şarap, ketçap şişesi, kırmızı ışık, kırmızı giysi. Liste: `config/prompts.yaml` → `negative`.

## 6. Donanım planı

| İş | Makine | Neden |
|---|---|---|
| Otomatik etiketleme, değerlendirme, RF-DETR-Seg eğitimi | RTX 2060 (Linux) | CUDA; 6 GB'a tiny/small modeller FP16 ile sığar (tahmin, `tools/bench.py` ölçecek) |
| Oynatıcı denemesi, vekil videolar | MacBook Air M5 | MPS; fansız, uzun işte ısıl kısma olur |
| Bulut (bu oturum) | Kod, test, belge | Ekran kartı yok; HF ve Roboflow erişimi kapalı |

## Kaynaklar

1. Fora Soft, "Grounding DINO vs OWLv2, Florence-2, RF-DETR: 2026 Guide": https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr
2. Fora Soft, "SAM 2 for video": https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/sam-2-for-video-memory-module-rotoscoping
3. LLMDet model kartı (iSEE-Laboratory): https://huggingface.co/iSEE-Laboratory/llmdet_large/blob/main/README.md · makale: https://arxiv.org/html/2501.18954v1
4. Roboflow, "RF-DETR Segmentation" (Oca 2026): https://roboflow.com/blog/rf-detr-segmentation
5. Roboflow, "What is SAM 3": https://roboflow.com/blog/what-is-sam3
6. Qualcomm AI Hub, EdgeTAM: https://aihub.qualcomm.com/models/edgetam · EfficientSAM3: https://arxiv.org/abs/2511.15833v1
7. "Malicious Image Analysis via Vision-Language Segmentation Fusion" (2025): https://arxiv.org/html/2512.04599v1
8. ScanCode LicenseDB, SAM License 2025-11-19: https://scancode-licensedb.aboutcode.org/sam-2025-11-19.html
9. "Segmentation-Based Blood Blurring: Examining Eye-Response Differences in Gory Video Viewing", Sensors 2025: https://www.mdpi.com/1424-8220/25/7/2093
10. https://universe.roboflow.com/joost-bos-rfw3p/blood-segmentation-y0ygo (sayfayı Emirhan açmasın)
11. https://universe.roboflow.com/backupgeneral/blood_segmentation_2 (aynı uyarı)
12. https://universe.roboflow.com/angel-macwan-6m3sx/blood-segmentation (aynı uyarı)
13. https://huggingface.co/datasets/NeuralShell/Gore-Blood-Dataset-v1.0 (aynı uyarı)
