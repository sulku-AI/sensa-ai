# Ses ipucu (D-019, demo)

Fikir (Emirhan, 2026-11-07): dizide kanlı sahnelerde çoğu zaman çığlık, bağırış, ağlama ya da silah sesi olur. Yemek programında kırmızı et ve sos çoktur ama çığlık olmaz, kızartma ve doğrama sesi olur. Ses, görüntü modelinin emin olamadığı bölgelerde karar vermeye yardım eder.

Ölçüt **kullanım başarısı**: insanlar bunu dizide açacak. Dizide kaçırma azalsın, yemek programında boşuna bulanıklık azalsın. Model metrikleri (AudioSet mAP) ikinci planda.

## Kurallar

Kod: `src/sensa_ai/audio_cue.py` → `decide`. Sayılar: `config/params.yaml` → `audio`.

1. **Ses tek başına bulanıklık açmaz.** Görüntüde aday iz yoksa çığlık bir şey yapmaz (korku filminde, maçta, kavgada kansız çok çığlık var).
2. **Görüntü çok eminse ses kapatamaz** (`sure_threshold`). Kaçırmak yanlış alarmdan pahalı (D-001).
3. Arada: `birleşik = görüntü + distress_boost × sıkıntı − kitchen_penalty × mutfak`, `keep_threshold` üstü tutulur.
4. **Ses nötrse sonuç bugünkü hatla aynı** (test: `test_neutral_audio_equals_visual_only`).

Ses açıkken dedektör daha düşük eşikle (`candidate_threshold`) aday toplar; böylece çığlık, görüntünün tek başına bırakacağı zayıf adayı kurtarabilir. İzin başından `pad_s` önce ve sonundan `pad_s` sonrasına kadar ses sayılır (çığlık kandan önce ya da sonra gelebilir). Sıkıntı aralıktaki en yüksek değer, mutfak aralığın ortalamasıdır (tek tencere sesi değil, sahnenin geneli).

| Durum | Sessiz hat | Sesli hat |
|---|---|---|
| zayıf görüntü + çığlık | kaçırır | bulanık |
| orta görüntü + kızartma sesi (yemek programı) | yanlış alarm | bulanık değil |
| emin görüntü + mutfak sesi | bulanık | bulanık (kural 2) |
| orta görüntü, nötr ses | bulanık | bulanık (kural 4) |
| zayıf görüntü, ses yok | bulanık değil | bulanık değil |

Vekil demo sonucu: `experiments/results/2026-11-07_ses_ipucu_vekil_demo.md` (beş sahnenin beşi doğru; sessiz hat ikisinde yanılıyor). Bu yalnız mantığın sınaması: sentetik ses, sentetik görüntü.

## Model

**AST** (Audio Spectrogram Transformer), AudioSet 527 sınıf: `MIT/ast-finetuned-audioset-10-10-0.4593`, BSD-3-Clause, transformers içinde (yeni bağımlılık yok). ~87M parametre. Grup olasılığı = gruptaki sınıfların en yükseği; gruplar `config/prompts.yaml` → `audio_vocal`, `audio_violence`, `audio_kitchen`.

| Aday | Lisans | Neden şimdi değil |
|---|---|---|
| AST AudioSet | BSD-3 | **seçildi**: transformers, MPS, sınıflar hazır |
| PANNs CNN14 | MIT kod, ağırlık CC BY 4.0 | ayrı paket; AST'den belirgin üstün değil |
| BEATs | MIT | transformers'ta yok, ağırlıklar ayrı indirilir |
| YAMNet | Apache 2.0 | TensorFlow bağımlılığı |
| EfficientAT (MobileNet) | MIT | çok hızlı; AST yavaş kalırsa ilk yedek |
| CLAP (LAION) | Apache 2.0 | metinle sıfır atış ("bir insan acıyla çığlık atıyor"); sınıflar yetmezse denenir |

Hız: saniyede bir pencere (`hop_s`), 3 sn pencere AST'nin 10.24 sn girdisine sıfırla doldurulur. M5'te ölçülmedi; görüntü hattı (6 örnek/sn × 25 ms) yanında ek yük beklenir. Ses bütün dosya için en başta da çıkarılabilir (ses çözmek görüntüden çok ucuz).

## Bilinen sınırlar

- Sınıf adları AudioSet ontolojisinden yazıldı, bulut oturumunda Hugging Face kapalı olduğu için modelin `id2label`'i ile doğrulanmadı. Modelde olmayan ad yüklemede uyarı olarak yazılır.
- Dizi müziği çığlığı bastırabilir; dublajlı içerikte efekt sesi kısık olabilir.
- Gerçek eşikler (`distress_full_at`, `kitchen_full_at`, boost, ceza) tahmin. Ayar H-07 ölçümüyle (yemek klipleri Emirhan, dizi sahneleri ortak).
- Canlı oynatıcıya (`sensa_ai.player`) henüz bağlı değil: demo çevrimdışı hatta (`analyze --audio`) ve onun ürettiği iz ile tarayıcı oynatıcısında çalışır.
- Ses saklama: yalnız sayılar (`.audiocue.json`) yazılır, ses kaydı ya da kesit yazılmaz.
