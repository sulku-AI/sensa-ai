# data/ (repoya girmez)

| Klasör | İçerik | Kim bakabilir |
|---|---|---|
| `negatif/` | Kansız ama benzer videolar: ketçap, sos, kırmızı boya, şarap, kırmızı ışık, ruj | **Herkes** (Emirhan dahil) |
| `vekil/` | Vekil hedefli videolar (kırmızı balon vb., D-005) | **Herkes** |
| `pozitif/` | Kan/vahşet içeren değerlendirme videoları | **Yalnız model ve ortak.** Emirhan açmaz, küçük resmine bakmaz |
| `etiket/` | Pozitifler için zaman aralığı etiketleri (JSON, görüntü yok) | Herkes |

Kurallar: `docs/TEST_PLANI.md`. Bu klasörden insan için görüntü üretmenin tek yolu `sensa_ai.safety.write_preview`.
