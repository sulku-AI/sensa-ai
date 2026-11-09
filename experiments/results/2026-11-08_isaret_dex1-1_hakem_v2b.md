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

## 3. Öneri dakikalarıyla ölçüm (b4646dc, her çalıştırma 135 sn)

```bash
python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4 --min-score 0.3 --detector rfdetr:models/rfdetr_seg_nano/checkpoint_best_total.pth --detector rfdetr:models/rfdetr_v2b/checkpoint_best_ema.pth --ranges "5-6,27-28" --unsure "4-5,6-7,16-17,19-20,26-27,28-30,38-39"
```

Kan aralığı dışı 40.8 dk (belirsiz 483 örnek sayılmadı). Hedef `requirements.false_alarm_per_min` 0.5.

| ölçüt | v1 (0.3) | v2b (0.3) | v1 (0.2) | v2b (0.2) |
|---|---|---|---|---|
| yakalanan_aralik | 2/2 | 2/2 | 2/2 | 2/2 |
| ici_isaretli_oran | 0.221 | 0.221 | 0.393 | 0.484 |
| disi_yanlis_alarm_olayi | 137 | 103 | 172 | 155 |
| **disi_dakikada_yanlis_alarm** | 3.356 | **2.523** | 4.214 | 3.797 |
| disi_isaretli_oran | 0.173 | 0.141 | 0.267 | 0.407 |

- Eşik 0.3'te v2b dakikada %25 daha az yanlış alarm veriyor, iki aralığı da yakalıyor. Hedefin hâlâ ~5 katı.
- Eşik 0.2'de (güçlü profil) v2b kan dışındaki saniyelerin %41'ini işaretliyor: v2b ile güçlü profil kullanılmamalı.
- Aralıklar v1 ve hakemlerden türetildi; içerideki sayılar v1 lehine eğilimli olabilir. Dışarıdaki yanlış alarm
  karşılaştırması daha adil.

## 4. Düzeltilmiş dizi eğitim paketi (b4646dc)

```bash
python -m tools.autolabel data/pozitif/videolar/dex1-1.mp4 --out data/pozitif/autolabel --min-score 0.2 --detector rfdetr:models/rfdetr_v2b/checkpoint_best_ema.pth
python -m tools.relabel_events data/pozitif/autolabel --video data/pozitif/videolar/dex1-1.mp4 --ranges "5-6,27-28" --unsure "4-5,6-7,16-17,19-20,26-27,28-30,38-39"
```

- autolabel (83 sn): 3054 kare, 1251 tespitli (v2b'nin 0.2 işaret sayısıyla aynı: eşik düzeltmesi çalışıyor),
  1806 bölge, 194 boş kare tutuldu.
- relabel: **475 zor negatif**, 160 boş negatif, **2 pozitif**; 227 belirsiz kare ve 71 kan aralığı karesi
  (skor < 0.5) atıldı, 510 fazla zor negatif dakika sınırıyla atıldı.
- Paket eğitime 637 kare katar, neredeyse hepsi "kan değil": amaç dizinin kırmızı yanlışlarını öğretmek. Kan
  örnekleri fotoğraf setlerinden gelir. Kan dakikalarının (5, 27) kareleri eğitime girmediği için yeni modelin
  dizideki yakalaması bu iki aralıkta ayrılmış gibi ölçülebilir.
- Pozitif eşiği düşürülmedi: v2b bu dakikalarda 0.2-0.5 arası çok tespit veriyor, hakemler bunların azını
  onaylıyor; onları "kan" diye öğretmek FA-04 hatasını geri getirebilir.

## 5. Sıradaki adım

1. `autolabel.tar` Drive `sensa_ai/`'ye (eskisinin yerine), Colab'da `DIZI = True`, `EGIT = True` ile eğitim
   (çıktı `rfdetr_v2b_dizi`).
2. v1, v2b ve yeni model aynı komutla (§3). Yeni model dışarıda daha az yanlış alarm verip iki aralığı da
   yakalamalı.

Not: 2.1.0 öncesi `autolabel --min-score 0.2` dedektör eşiğinin (0.3) altına inmiyordu; düzeltildi.
7 Kasım'daki autolabel (eşik 0.6) bundan etkilenmedi.
