# Pozitif set değerlendirmesi 2026-11-04 (M5, mps), H-01 ilk ölçüm

`python -m tools.evaluate --pos data/pozitif/blood_seg_joost --pos data/pozitif/blood_seg_backup` (Emirhan'ın makinesi; yalnız sayı). **Not:** `blood_seg_backup` indirilemedi (Roboflow: proje yok), 200 görüntünün hepsi `blood_seg_joost`'tan. Kapı SigLIP base (0.30), dedektör OWLv2 base (kutu eşiği 0.25), sıfır atış.

| ölçüt | değer | hedef |
|---|---|---|
| pozitif_goruntu | 200 | |
| yakalama_orani | 0.330 | 0.95 |
| ortalama_kapsama | 0.333 | 0.90 |
| kapsama_p10 | 0.000 | |
| fazla_bulaniklik_ort | 0.075 | |
| saniye_goruntu_basi | 0.654 | |

H-01 (sıfır atış, bu ayar): **ÇÜRÜTÜLDÜ** (0.33 ≪ 0.95). Literatürdeki denetimli YOLOv8s-seg taban çizgisi 0.655 (Sensors 2025).
Sıradaki: kapı mı dedektör mü kaçırıyor (`--no-gate`, `pozitif_kapi_gecis_orani`), düşük kutu eşiği (OWLv2 puanları düşük), sonra bu setlerle denetimli RF-DETR-Seg eğitimi (D-011 damıtma hedefi).

## Tanı (Colab T4, aynı 200 görüntü, 2026-11-04)

| ayar | yakalama | kapsama | fazla bulanıklık | pozitif kapı geçişi |
|---|---|---|---|---|
| kapı yok, kutu 0.25 | 0.335 | 0.337 | 0.078 | — |
| kapı yok, kutu 0.10 | 0.750 | 0.735 | 0.260 | — |
| kapı 0.15, kutu 0.10 | 0.750 | 0.735 | 0.260 | 0.990 |

Sonuç: kaçırmanın sebebi **kapı değil** (pozitiflerin %99'u geçiyor), **dedektörün kutu eşiği**. OWLv2 puanları düşük; 0.25 çok yüksek. 0.10'da yakalama 0.33 → 0.75 (Sensors 2025 denetimli tabanı 0.655'i geçti), bedeli fazla bulanıklık 0.08 → 0.26. Sıradaki: negatif sette 0.10 eşiğin yanlış alarmı, 0.05 denemesi, sonra denetimli eğitim.

## Karışık set ve eşik bedeli (Colab T4, 2026-11-04)

447 pozitif (`blood_seg_joost` 200 + `blood_pure_blur` ilk 300'den etiketlisi; klasör başına `--limit 300`), 300 negatif (COCO), kapı 0.30. `blood_pure_blur` dışa aktarımı 33546 görüntü / 33811 bölge (Universe'te 3229 özgün görüntü: set artırılmış).

| kutu eşiği | yakalama | kapsama | fazla bulanıklık | negatif yanlış alarm | negatif bulanık alan | pozitif kapı | negatif kapı |
|---|---|---|---|---|---|---|---|
| 0.25 (önceki ölçüm, yalnız negatif) | — | — | — | 0.003 | — | — | 0.462 |
| 0.10 | 0.365 | 0.356 | 0.125 | 0.150 | 0.064 | 0.987 | 0.450 |
| 0.05 | 0.508 | 0.502 | 0.198 | 0.310 | 0.124 | 0.987 | 0.450 |

Sonuç: `blood_pure_blur` OWLv2 için çok daha zor (joost'ta 0.10 eşikte 0.75, karışıkta 0.365). Eşiği düşürmek yakalamayı ancak 0.51'e çıkarıyor, bedeli kansız görüntülerin %31'inde yanlış bulanıklık. Sıfır atışla 0.95'e eşik ayarıyla ulaşılamaz; varsayılan kutu eşiği 0.25 kalır. H-01 sıfır atış için **ÇÜRÜTÜLDÜ** (kesin). Sıradaki: denetimli RF-DETR-Seg Nano (`tools.train_rfdetr`, sızıntısız grup bölünmesi), ölçüm yalnız ayrılmış `egitim/test` üzerinde (`--empty-as-neg`).
