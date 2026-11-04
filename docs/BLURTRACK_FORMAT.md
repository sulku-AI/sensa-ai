# `.blurtrack.json` biçimi, sürüm 1 (taslak)

Videodan ayrı, altyazı gibi bir **bulanıklaştırma izi** (D-003). Kendi oynatıcımız bunu kullanır; uzun vadede bir platform, videoyu kendi tarafında tarayıp bu dosyayı üretebilir ve oynatıcısında uygulayabilir. Dosya görüntü içermez, yalnız zaman ve normalize koordinat taşır.

```json
{
  "format": "sensa-blurtrack",
  "version": 1,
  "video": "film.mp4",
  "duration_s": 5400.0,
  "analyzed_until_s": 5400.0,
  "tracks": [
    {
      "label": "blood",
      "keyframes": [
        {"t": 41.0, "box": [0.12, 0.55, 0.38, 0.91], "score": 0.71,
         "poly": [[0.13, 0.56, 0.37, 0.58, 0.35, 0.90, 0.14, 0.88]]},
        {"t": 41.04, "box": [0.13, 0.55, 0.39, 0.91], "score": 0.71, "poly": null}
      ]
    }
  ]
}
```

| Alan | Anlamı |
|---|---|
| `format`, `version` | Sabit `"sensa-blurtrack"`, `1`. Okuyucu başka değerde reddeder |
| `video` | Kaynak dosya adı (bilgi amaçlı) |
| `duration_s` | Video süresi |
| `analyzed_until_s` | Bu saniyeye kadar analiz edildi. Canlı oynatıcı bunun ötesini güvenli modda gösterir (D-007) |
| `tracks[].label` | Sınıf: şimdilik `blood` (ileride `gore`, `wound`) |
| `keyframes[].t` | Saniye |
| `keyframes[].box` | `[x0, y0, x1, y1]`, 0..1, sol üst köşe başlangıç |
| `keyframes[].score` | Model güveni 0..1 (oynatıcı kullanmaz; eşik analizde uygulanır) |
| `keyframes[].poly` | İsteğe bağlı sadeleştirilmiş poligonlar `[x1, y1, x2, y2, ...]`, 0..1. Yoksa kutu kullanılır |

## Oynatıcı kuralları

1. İki anahtar kare arasında kutu doğrusal ara değerle; poligon en yakın anahtar karenin şekli, kutunun kaymasıyla ötelenir.
2. Bulanıklık izin başlangıcından `blur.ramp_in_s` **önce** başlar, bitişinden `blur.ramp_out_s` sonra söner (güç 0→1→0). İleri bakma (30 sn) bunu mümkün kılar.
3. Bölge `blur.dilate_px` kadar genişletilir ve `blur.feather_px` ile kenarı yumuşatılır; güç `blur.sigma_px` Gauss bulanıklığıdır. Değerler `config/params.yaml`, profil ile değişebilir.
4. `analyzed_until_s - t < lookahead.safe_margin_s` ise (ve video sonu analiz edilmediyse) tam kare hafif bulanıklık.

## Açık konular

- Maskeler için poligon yerine RLE (daha kesin, daha büyük dosya).
- Sınıf başına ayrı güç (kan hafif, ağır yaralanma güçlü).
- Platform tarafı için zaman tabanı (kare numarası mı saniye mi) ve imza.
