# RF-DETR-Seg Nano ilk denetimli eğitim (Colab T4, 2026-11-04/05), H-01

Komut: `tools.train_rfdetr prepare --src blood_seg_joost blood_pure_blur --neg data/negatif/coco --aug-per-image 2`, sonra `train --epochs 20 --batch 8 --accum 2 --device cuda` (rfdetr Lightning sürümü, AMP 16). Süre ~3 sa (tur başına ~9 dk). Ağırlık Emirhan'ın Drive'ında, repoda yok.

Bölünme (grup bütün, sızıntısız): train 5630 görüntü (2009 boş), valid 675 (167 boş), test 84 (38 boş).

## Doğrulama eğrisi (EMA, valid)

| tur | segm mAP 50:95 | kutu mAP 50 | F1 | kesinlik | duyarlılık |
|---|---|---|---|---|---|
| 1 | 0.148 | 0.297 | 0.329 | 0.342 | 0.317 |
| 5 | 0.210 | 0.397 | 0.424 | 0.511 | 0.362 |
| 10 | 0.223 | 0.413 | 0.436 | 0.503 | 0.385 |
| 18 (en iyi) | 0.233 | 0.437 | 0.450 | 0.465 | 0.435 |
| 20 | 0.227 | 0.421 | 0.438 | 0.532 | 0.373 |

Eğri 12. turdan sonra düzleşti; daha uzun eğitim az kazandırır.

## Ayrılmış test (`tools.evaluate --pos data/pozitif/egitim/test --empty-as-neg --no-gate`)

| ölçüt | kutu 0.5 | kutu 0.3 | hedef | OWLv2 sıfır atış (karışık, 0.05) |
|---|---|---|---|---|
| pozitif görüntü | 46 | 46 | | 447 |
| yakalama | 0.957 | 0.957 | 0.95 | 0.508 |
| ortalama kapsama | 0.944 | 0.946 | 0.90 | 0.502 |
| kapsama p10 | 0.905 | 0.940 | | |
| fazla bulanıklık | 0.050 | 0.050 | | 0.198 |
| negatif görüntü | 38 | 38 | | 300 |
| yanlış alarm | 0.026 | 0.026 | | 0.310 |
| negatif bulanık alan | 0.003 | 0.003 | | 0.124 |
| sn/görüntü (T4, FP32) | 0.068 | 0.061 | | |

H-01: bu test setinde **DESTEKLENDİ (ön)**. Uyarılar: (1) test küçük (46 pozitif: 0.957 = 44/46, %95 güven aralığı yaklaşık 0.85-0.99); (2) iki küçük Roboflow setinin kendi test bölümü, gerçek film/video karesi değil; (3) negatifler yalnız COCO, kırmızı/ketçap/boya gibi zor negatifler yok. Görüntü düzeyi ölçüt (kapsama ≥ 0.5) örnek düzeyi mAP'den çok daha cömert; mAP düşükken yakalamanın yüksek çıkması bundan.

Sıradaki: valid (675) üzerinde aynı ölçüm (seçimde kullanıldı, hafif iyimser ama 15 kat büyük), zor negatif seti, sonra M5'te hız (bench) ve vekil video oynatıcıda deneme.
