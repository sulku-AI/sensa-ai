# Ses ipucu vekil demosu 2026-11-07

`python -m tools.audio_demo`

| Sahne | Olmalı | Sessiz hat | Sesli hat |
|---|---|---|---|
| dizi: zayıf görüntü + çığlık | bulanık | - | bulanık ✓ |
| dizi: orta görüntü, sessiz | bulanık | bulanık | bulanık ✓ |
| yemek: orta görüntü + cızırtı | - | bulanık | - ✓ |
| sessiz: zayıf görüntü, ses yok | - | - | - ✓ |
| dizi: emin görüntü + mutfak sesi | bulanık | bulanık | bulanık ✓ |

| Hat | Olay yakalama | Olay dışı bulanık sn |
|---|---|---|
| sessiz_hat | 0.67 | 6.1 |
| sesli_hat | 1.00 | 3.1 |
