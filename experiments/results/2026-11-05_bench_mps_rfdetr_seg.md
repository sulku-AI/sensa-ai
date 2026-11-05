# Hız: eğitilmiş RF-DETR-Seg Nano, M5 MPS (2026-11-05), H-03

`python -m tools.bench --detector rfdetr_seg --res 720 --frames 20 --no-gate [--no-track]` (Emirhan'ın M5'i, 1280x720 sentetik kare, FP16 isteği açık; rfdetr `inference` izleme uyarıları zararsız).

| ayar | dedektör ms | takip ms/kare | analiz s / video s | realtime_factor |
|---|---|---|---|---|
| kapı yok + EdgeTAM takip (her aralıkta bölge varsayımı, en kötü durum) | 25.6 | 157.6 | 0.997 | 1.00 |
| kapı yok, takip yok | 24.8 | — | 0.050 | 20.18 |

Karşılaştırma: OWLv2 base 490 ms/kare (M5) → RF-DETR-Seg 25 ms, ~20 kat hızlı.

Yorum: darboğaz artık takip. Bench `--no-gate`'te bölge oranını 1.0 sayar; gerçek videoda takip yalnız bölge varken çalışır (ör. sürenin %20'si: 0.05 + 0.2 × 0.95 ≈ 0.24, rf ≈ 4). Başka yol: takip yerine dedektörü daha sık çalıştırmak (6 örnek/sn × 25 ms = 0.15, rf ≈ 6.7; hesap, ölçülmedi), model zaten maske verdiği için. H-03 bu dedektörle **DESTEKLENDİ** (sentetik kare; gerçek video doğrulaması sürüyor).
