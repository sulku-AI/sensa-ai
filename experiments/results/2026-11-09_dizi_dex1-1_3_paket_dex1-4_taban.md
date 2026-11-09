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

## 5. Ek: dex1-5 ve dex1-6, ikinci tur paketi

Bölümler 53.4 ve 52.9 dk. Mac ısınmasın diye arka plan önceliğinde, 2 çekirdekle çalıştı (bölüm başına
~22 dk). Bu geceki ilk modelin (1-3) görmediği bölümler: ilk model 4, 5 ve 6'da ölçülür.

| bölüm | `--ranges` | `--unsure` |
|---|---|---|
| dex1-5 | 1-2 | 0-1,2-3,50-51 |
| dex1-6 | 2-3,40-41,43-45 | 1-2,3-4,8-9,24-26,39-40,41-43,45-46 |

relabel: dex1-5 zor negatif 499, boş 229, pozitif 7 (atıldı 47, belirsiz 98, fazla negatif 329);
dex1-6 zor negatif 404, boş 201, pozitif 14 (atıldı 129, belirsiz 293, fazla negatif 231).

Taban (eşik 0.3), kan dışı dakikada yanlış alarm:

| bölüm | v1 | v2b | yakalanan |
|---|---|---|---|
| dex1-4 | 2.655 | 2.414 | 2/2, 2/2 |
| dex1-5 | 2.594 | 1.601 | 1/1, 1/1 |
| dex1-6 | 2.232 | 2.258 | 3/3, 3/3 |

`autolabel_tur2.tar` (ikinci eğitim): 578 062 848 bayt, sha256
`a08805e86d987572c1f0917cddd691ec45c42ed183ed45b1bd3b6b71c3592425`; dex1-1/2/3/5/6, 3129 görüntü,
41 pozitif. dex1-4 hiçbir pakette yok.

## 6. Son ikinci tur paketi (dex1-7, dex1-8 eklendi)

`autolabel_tur2.tar`: 803 834 368 bayt, sha256 `9158986c8534a1312e10f4f21053e8f2fe15401c3f650bd10945043f472e3174`;
dex1-1/2/3/5/6/7/8, dex1-4 yok. Bölümler tam hızda işlendi (1289 sn).

| bölüm | `--ranges` | `--unsure` | zor negatif | boş negatif | pozitif |
|---|---|---|---|---|---|
| dex1-7 | (aday yok) | 0-3,4-5,7-9,27-30,38-39,46-47 | 438 | 169 | 0 |
| dex1-8 | 1-2,3-4,8-10,21-22,42-44,48-49 | (rapor edilmedi) | 357 | 136 | 23 |
