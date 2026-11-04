# Model seçimi (2026-11-03)

Kısa cevap: **elle etiketli veri seti yok.** Hedef bölgeyi metin isteminden ("blood", "open wound") bulan açık modeller ve bulduğu bölgeyi karelere yayan SAM 2 kullanılır. Bütün sayılar `config/params.yaml`'da; burada kopya sayı yok.

## Boru hattı

```
video ──► [örnekleme: lookahead.sample_fps] ──► A. Kapı (SigLIP, kare başına)
                                                   │ skor < gate.threshold → temiz kare, geç
                                                   ▼
                                               B. Bölge (Grounding DINO / OWLv2, metin istemi)
                                                   │ kutular
                                                   ▼
                                               C. Takip (SAM 2.1 video, kutu istemli)
                                                   │ her kare için maske
                                                   ▼
                                     .blurtrack.json  ──►  oynatıcı: yumuşak bölgesel bulanıklık
```

A pahalı modelleri çoğu karede hiç çalıştırmamak için var. C, saniyede 1-2 tespiti tam kare hızına çıkarır ve kutu yerine bölgenin şeklini verir.

## Adaylar

| Aşama | Model | Lisans | Neden | Risk |
|---|---|---|---|---|
| A. Kapı | **SigLIP** (google/siglip, HF transformers) | Apache 2.0 | Sıfır atış kare sınıflama; pozitif istemler negatiflerle (ketçap, boya, şarap…) aynı anda karşılaştırılır, yanlış alarmı kapıda keser | Web verisiyle eğitildi; kan için kalibrasyonu bilinmiyor (H-02) |
| B. Bölge | **Grounding DINO** (IDEA-Research, tiny/base, HF transformers) | Apache 2.0 | Metin istemli nesne kutusu; açık kelime dağarcığı | Kan şekilsiz bir "madde", nesne değil: kutular kaba olabilir (H-01) |
| B. Bölge (karşılaştırma) | **OWLv2** (google/owlv2, HF transformers) | Apache 2.0 | Aynı iş, farklı eğitim; ikisi negatif + pozitif sette yarıştırılır, iyisi kalır | Aynı |
| C. Takip | **SAM 2.1** (facebookresearch/sam2, tiny/small) | Apache 2.0 | Kutu isteminden maske, video boyunca yayılım, kapanma sonrası yeniden bulma | Bellek; uzun videoda parça parça çalıştırılmalı (H-03) |

Kullanılmayanlar:
- **YOLO-World / Ultralytics:** hızlı ama AGPL-3.0; MIT projeye karışması lisansı bozar (D-009).
- **Eski Faster R-CNN eğitimi:** elle etiket gerektiriyor, yani birinin kanlı kareleri tek tek kutulaması (FA-01). Daha sonra **damıtma** için geri gelebilir: A-B-C hattının ürettiği otomatik etiketlerle küçük, hızlı bir model eğitilir (tarayıcı eklentisi için, M5). Böylece kimse görüntü görmeden veri seti oluşur.
- **Büyük görsel dil modelleri (VLM):** gerçek zamanlı için ağır. Değerlendirmede ikinci görüş olarak kullanılabilir (pozitif klipleri metinle özetleyip "kaçan var mı" diye sormak); lisans her model için ayrıca kontrol edilecek.

## Donanım

Hedef normal bir dizüstü ekran kartı. Hız ölçülmedi; H-03 bunu ölçer. Ekran kartı yetmezse: daha küçük sürümler (Grounding DINO tiny, SAM 2.1 tiny), daha düşük `sample_fps`, ya da analiz önceden çevrimdışı yapılır ve oynatıcı yalnız izi uygular. Colab, M1-M2 deneyleri için yedek.

**Açık soru:** Emirhan'ın bilgisayarındaki ekran kartı (model ve VRAM). M1 öncesi lazım.

## Kod

Modeller `sensa_ai.detect.Detector` arayüzünün arkasına eklenir (M1). Şimdilik yalnız `ColorProxyDetector` var: modelsiz, kırmızı alan bulan vekil; gerçek tespit değil, boru hattını kansız sınamak için (D-005).
