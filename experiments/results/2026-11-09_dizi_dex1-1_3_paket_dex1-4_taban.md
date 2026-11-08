# Dizi paketi (dex1-1..3) ve dex1-4 taban ölçümü (2026-11-09)

Mac M5, `tools.flag_timeline` / `autolabel` / `relabel_events` (91ec1ce). Hiçbir kare ya da video
açılmadı; yalnız metin çıktısı. Akış: [docs/DIZI_VERISI.md](../../docs/DIZI_VERISI.md), D-022.

Bölümler: dex1-2 54.7 dk, dex1-3 48.8 dk, dex1-4 57.8 dk. Eğitim 1-3, **ayrılmış test 4** (eğitime
girmez; yeni modelin dizide genelleme ölçümü).

## 1. Hakem önerisi (eşik 0.3, birincil v1, hakem SigLIP + OWLv2)

| bölüm | aday kan dakikaları (`--ranges`) | belirsiz (`--unsure`) | süre |
|---|---|---|---|
| dex1-1 | 5-6,27-28 | 4-5,6-7,16-17,19-20,26-27,28-30,38-39 | 433 sn |
| dex1-2 | 3-4,9-10,11-12,27-29 | 2-3,4-5,7-9,10-11,12-13,15-17,23-24,26-27,29-30,41-42 | 389 sn |
| dex1-3 | 0-1,44-46 | 1-3,9-10,30-31,33-34,43-44,46-47 | 473 sn |
| dex1-4 | 2-3,22-23 | 0-2,3-5,21-22,23-24 | 448 sn |

Model önerisidir, insan bilgisi değil.

## 2. Eğitim paketi (autolabel v2b 0.2, relabel pos-min 0.5, dakikada en çok 12 zor negatif)

| bölüm | kare | tespitli | zor negatif | boş negatif | pozitif | atıldı | belirsiz atıldı | fazla negatif atıldı |
|---|---|---|---|---|---|---|---|---|
| dex1-1 | 3054 | 1251 | 475 | 160 | 2 | 71 | 227 | 510 |
| dex1-2 | 3276 | 1191 | 383 | 145 | 13 | 152 | 344 | 376 |
| dex1-3 | 2931 | 1185 | 437 | 156 | 4 | 122 | 217 | 439 |

`autolabel.tar` 365 954 048 bayt: `data/pozitif/autolabel/dex1-{1,2,3}/`, 1775 görüntü, 19 pozitif.
Hepsi relabel geçmişli (`unrelabeled_videos` boş). Kan örnekleri fotoğraf setlerinden gelir; dizi
kareleri kırmızı yanlışları öğretir.

## 3. dex1-4 taban (eşik 0.3, yukarıdaki öneri)

Kan dışı 49.7 dk, 360 belirsiz örnek sayılmadı. Hedef `requirements.false_alarm_per_min`.

| ölçüt | v1 | v2b |
|---|---|---|
| yakalanan_aralik | 2/2 | 2/2 |
| ici_isaretli_oran | 0.450 | 0.325 |
| disi_yanlis_alarm_olayi | 132 | 120 |
| **disi_dakikada_yanlis_alarm** | 2.655 | **2.414** |
| disi_isaretli_oran | 0.100 | 0.097 |

- Görülmemiş bölümde de v2b biraz daha az yanlış alarm veriyor (%9), hedefin ~5 katı.
- Aralıklar v1 ile hakemlerden türetildi; içerideki oran v1 lehine eğilimli olabilir.

## 4. Sıradaki

Linux PC'de (RTX 2060) `rfdetr_v2b_dizi` eğitimi ([docs/PC_EGITIM.md](../../docs/PC_EGITIM.md)),
sonra aynı komutla dex1-4'te üç model. Varsayılan modeli kullanıcı seçer.
