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
