# Yanlış alarm araçları (Y1–Y3) Mac ölçümü, 2026-11-10

Yalnız sayı. Mac'te Hannibal 1. bölüm yok; ölçüm doğrulayıcının ayrılmış bölümü **dex3-4**'te yapıldı
(54.5 dk, 6 fps, eşik 0.3). Gerçek kan aralığı insan bilgisi değil, hakem önerisi
(`2026-11-10_isaret_dex3-4_e0.3_hakem_2.md`, v2b_dizi + siglip + owlv2_base):
`--ranges "37-38" --unsure "2-3,9-10,36-37,38-39"`. Tek aralık; kapsama sayıları zayıf kanıt.

Modeller: `rfdetr_v2b_dizi` (Drive, 2026-11-09 01:13; dex3 bölümlerini görmedi), `rfdetr_v2b` (önceki).
Doğrulayıcı: `2026-11-10_dogrulayici.md` (eşik params verify.threshold = 0.30).

| sütun | yakalanan_aralik | ici_isaretli_oran | disi_dakikada_yanlis_alarm |
|---|---|---|---|
| v2b (önceki) | 1/1 | 0.128 | 4.451 |
| v2b_dizi | 1/1 | 0.086 | 0.688 |
| v2b_dizi +dogrula | 1/1 | 0.086 | 0.607 |
| v2b_dizi k=2 | 1/1 | 0.086 | 0.486 |
| v2b_dizi k=3 | 1/1 | 0.086 | 0.405 |
| v2b_dizi +dogrula k=2 | 1/1 | 0.086 | 0.486 |
| v2b_dizi +dogrula k=3 | 1/1 | 0.086 | 0.405 |

Ham tablolar: `2026-11-10_isaret_dex3-4_e0.3.md` (koşu A), `2026-11-10_isaret_dex3-4_e0.3_2.md` (koşu B).

Gözlem: zamansal filtre yanlış alarmı %29 (k=2) / %41 (k=3) düşürüyor, aralık içi kapsama değişmiyor.
Doğrulayıcı tek başına %12; filtrenin üstüne bir şey eklemiyor (elediği olaylar zaten kısa koşular).
params.yaml varsayılanları değiştirilmedi (min_hits 1, verify.enabled false).
