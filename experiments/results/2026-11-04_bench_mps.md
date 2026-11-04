# Hız ölçümü 2026-11-04 (mps), MacBook Air M5

Emirhan'ın makinesinde `python -m tools.bench` (ekran görüntüsünden aktarıldı). EdgeTAM yüklenemedi (FA-02), yedek SAM 2.1 tiny çalıştı.

| ölçüt | değer |
|---|---|
| tarih | 2026-11-04 |
| makine | Darwin arm64 arm |
| cihaz | mps |
| kare | 1280x720 |
| kapi_model | google/siglip-base-patch16-224 |
| kapi_ms | 80.3 |
| dedektor_model | iSEE-Laboratory/llmdet_tiny |
| dedektor_ms | 741.9 |
| takip_model | facebook/sam2.1-hiera-tiny |
| takip_ms_kare | 652.9 |
| analiz_s_video_s | 3.591 |
| realtime_factor | 0.28 |
| hedef_realtime_factor | 1.2 |
| gecti | False |

## Yorum (asistan)

Video saniyesi başına yük (sample_fps 2, kapı geçişi 0.2 varsayımı):
- kapı: 2 × 0.080 = 0.16 s
- dedektör: 2 × 0.2 × 0.742 = 0.30 s
- takip: 0.2 × 24 × 0.653 = **3.13 s (%87)**

Darboğaz takip, çünkü her karede çalışıyordu. Değişiklik: `track.fps` (6) eklendi; takip saniyede 6 karede, aradaki kareler iz ara değeriyle. Beklenen yük 0.16 + 0.30 + 0.2 × 6 × 0.653 = 1.24 s → realtime_factor ≈ 0.8 (hâlâ hedefin altında). Takipsiz: 0.46 s → ≈ 2.2. Sıradaki ölçüm: yeni kodla `tools.bench`, `--no-track` ve `--detector owlv2_base`.
