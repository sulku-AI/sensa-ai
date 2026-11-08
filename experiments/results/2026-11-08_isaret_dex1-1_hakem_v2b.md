# Dizide hakem önerisi ve v1 / v2b karşılaştırması (2026-11-08)

Video `dex1-1.mp4` (yasal kopya): 50.9 dk, saniyede 1 örnek, 3054 örnek. Mac M5, `tools.flag_timeline`
(6fe62a7). Hiçbir kare ya da video açılmadı; yalnız metin çıktısı. Önceki tek model ölçümü:
[2026-11-08_isaret_dex1-1_v1.md](2026-11-08_isaret_dex1-1_v1.md).

Modeller: **v1** `models/rfdetr_seg_nano/checkpoint_best_total.pth` (ilk eğitim), **v2b**
`models/rfdetr_v2b/checkpoint_best_ema.pth` (coco_zor zor negatifleriyle, dizi karesi yok).

## 1. Hakemli çalıştırma (eşik 0.3, 433 sn)

```bash
python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4 --min-score 0.3 --judge siglip --judge owlv2_base
```

Hakemler yalnız v1'in işaretlediği saniyelere bakar. `ortak`: üçü birlikte "kan" diyor.

| ölçüt | v1 | hakem:siglip | hakem:owlv2_base | ortak |
|---|---|---|---|---|
| isaretli_ornek | 548 | 382 | 31 | **28** |
| isaretli_oran | 0.179 | 0.125 | 0.010 | 0.009 |

Ortak saniyesi olan dakikalar (diğer bütün dakikalarda ortak 0):

| dakika | v1 | siglip | owlv2 | ortak |
|---|---|---|---|---|
| 5 | 13 | 13 | 5 | **5** |
| 8 | 5 | 3 | 1 | 1 |
| 12 | 3 | 3 | 1 | 1 |
| 15 | 22 | 22 | 1 | 1 |
| 16 | 26 | 26 | 2 | 2 |
| 19 | 3 | 3 | 2 | 2 |
| 20 | 15 | 15 | 1 | 1 |
| 25 | 12 | 11 | 1 | 1 |
| 27 | 14 | 8 | 6 | **5** |
| 28 | 10 | 10 | 2 | 2 |
| 29 | 17 | 10 | 4 | 3 |
| 38 | 22 | 15 | 4 | 3 |
| 49 | 53 | 49 | 1 | 1 |

Gözlemler:

- SigLIP v1'in işaretlerinin %70'ine katılıyor (548'in 382'si): "kan mı, ketçap / boya / kırmızı kumaş mı"
  ayrımı dizide zayıf. Ayırt eden OWLv2 (31 saniye).
- 49-50. dakika büyük olasılıkla kırmızı yanlış alarm: v1'in en yoğun bloğu, OWLv2 1 saniye, v2b de
  neredeyse hiç işaretlemiyor (aşağıda).

**Öneri** (params `dizi`, `tools.flag_timeline` 2.1.0 sonrası iki satırı da kendisi yazar):

- aday kan dakikaları (en az 5 ortak saniye): `5-6,27-28`
- belirsiz (2-4 ortak saniye ve aday dakikaların önü / arkası): `4-5,6-7,16-17,19-20,26-27,28-30,38-39`

Kullanıcı bu öneriyle devam etmeyi seçti (D-022). Model önerisidir, insan bilgisi değil.

## 2. v1 ve v2b aynı karelerde

| eşik | v1 | v2b | süre |
|---|---|---|---|
| 0.2 | 855 (0.280) | 1251 (0.410) | 141 sn |
| 0.3 | 548 (0.179) | 469 (0.154) | 138 sn |

İşaretli örnek (oran). Eşik 0.3'te:

- v2b 49-50 bloğunu neredeyse bırakıyor: 49. dk 11 (v1 53), 50. dk 0 (v1 41).
- v2b 28-33. dakikalarda daha çok işaretliyor, ör. 32. dk 31 (v1 10).

v2b'nin skorları 0.2-0.3 arasında yığılıyor: 0.2'de v1'den çok, 0.3'te az işaretliyor. Eşiği ayrıca
seçilmeli. Gerçek kan dakikaları olmadan "daha iyi" denemez: sıradaki ölçüm `--ranges` ve `--unsure` ile
dakikada yanlış alarm (hedef `requirements.false_alarm_per_min`).

## 3. Sıradaki adım

1. v2b ile eşik 0.2'de autolabel, `relabel_events --ranges "5-6,27-28" --unsure "<belirsiz>"`.
2. Colab'da `DIZI = True` ile eğitim (dizi kareleri yalnız train'e).
3. v1, v2b ve yeni model `flag_timeline --ranges ... --unsure ...` ile aynı bölümde.

Not: 2.1.0 öncesi `autolabel --min-score 0.2` dedektör eşiğinin (0.3) altına inmiyordu; düzeltildi.
7 Kasım'daki autolabel (eşik 0.6) bundan etkilenmedi.
