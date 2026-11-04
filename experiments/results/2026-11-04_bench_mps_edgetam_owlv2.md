# Hız ölçümü 2026-11-04 (mps), MacBook Air M5: varsayılan hat (D-014)

Emirhan'ın makinesinde `python -m tools.bench --detector owlv2_base` (ekran görüntüsünden aktarıldı). Sentetik kareler, kapı geçiş oranı **varsayım** 0.2.

| ölçüt | değer |
|---|---|
| kare | 1280x720 |
| kapi_model | google/siglip-base-patch16-224 |
| kapi_ms | 80.3 |
| dedektor_model | google/owlv2-base-patch16-ensemble |
| dedektor_ms | 489.7 |
| takip_model | facebook/EdgeTAM@refs/pr/1 |
| takip_ms_kare | 130.3 |
| takip_fps | 6.0 |
| analiz_s_video_s | 0.513 |
| realtime_factor | 1.95 |
| hedef_realtime_factor | 1.2 |
| gecti | True |

## Gün içindeki ölçümler (aynı makine)

| hat | takip | realtime_factor |
|---|---|---|
| LLMDet + SAM 2.1 tiny, her karede takip | 653 ms | 0.28 |
| LLMDet + SAM 2.1 tiny, track.fps 6 | 588 ms | 0.86 |
| OWLv2 + SAM 2.1 tiny, track.fps 6 | 586 ms | 0.94 |
| LLMDet, takipsiz | — | 2.19 |
| **OWLv2 + EdgeTAM, track.fps 6** | **130 ms** | **1.95** |

H-03: **KISMEN** destekleniyor. Kalan şart: kapı geçiş oranını gerçek videoda ölçmek (0.2 varsayım; 0.4'te bu hat ≈ 1.1 olur, hesap).
