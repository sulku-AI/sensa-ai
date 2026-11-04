# Negatif set değerlendirmesi 2026-11-04 (M5, mps)

`python -m tools.evaluate --neg data/negatif/coco --previews` (Emirhan'ın makinesi; metinden aktarıldı). COCO val, kırmızıya yakın 10 sınıf, 381 görüntü. Kapı SigLIP base (eşik 0.30), dedektör OWLv2 base.

| ölçüt | değer |
|---|---|
| negatif_goruntu | 381 |
| negatif_yanlis_alarm_orani | 0.003 |
| negatif_bulanik_alan_ort | 0.003 |
| negatif_kapi_gecis_orani | 0.462 |
| saniye_goruntu_basi | 0.307 |

## Yorum (asistan)

- H-02 için iyi işaret: 381 kansız kırmızı görüntüden yaklaşık 1'inde bulanıklık. Dakika başına ölçüt (`false_alarm_per_min`) video ister, bu görüntü düzeyi.
- Kapı bu sette %46 geçiriyor (bench varsayımı %20). Set bilerek kırmızı ağırlıklı, sıradan film için üst sınır sayılabilir. %46 ile hesap: 2 × (0.080 + 0.46 × 0.490) + 0.46 × 6 × 0.130 = 0.97 s/s → realtime_factor ≈ 1.03 (hedef 1.2'nin altında).
- Kapı eşiğini yükseltmek pozitif sette yakalamayı düşürebilir; eşik ayarı pozitif set ölçülmeden yapılmaz.
