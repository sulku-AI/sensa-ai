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

Varsayılan **CED-Base** (`mispeech/ced-base`, Xiaomi). 2026-11-07'de güncel açık modeller karşılaştırıldı (Emirhan'ın isteği: "bulabileceğin en iyi model"):

| Model | AudioSet mAP | Parametre | Lisans (kod / ağırlık) | Bizim için |
|---|---|---|---|---|
| **CED-Base** | **50.0** | 86M | HF kodu Apache 2.0 (dosya başlıkları), ağırlık Apache 2.0 | **seçildi**: en yüksek doğrulanmış mAP; değişken uzunluk (3 sn pencere doldurulmaz); 10 sn başına 248 yama (AST 1212): aynı boyda daha hızlı |
| CED-Small | 49.6 | 22M | aynı | M5'te CED-Base yavaş kalırsa: `--audio ced_small` |
| EAT-large / base | 49.5 / 48.9 | 309M / 88M | kod MIT, ağırlık lisansı yazmıyor | lisans belirsiz, şimdilik değil |
| EfficientAT (MN ensemble / dymn20) | 49.8 / 49.1 | toplam / 40M | kod MIT, ağırlık lisansı yazmıyor; transformers dışı | lisans belirsiz |
| BEATs iter3 | 48.0 | 90M | sayfa okunamadı | doğrulanamadı |
| AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) | 45.9 (tek model) | 87M | BSD-3 | seçenek: `--audio ast_audioset`; özel kod gerektirmez |
| Dasheng | (sınıflandırıcı yok) | 86M-1.2B | Apache 2.0 | yalnız gömme, 527 sınıf başı yok |

Kaynaklar: model kartları ve depolar (huggingface.co/mispeech/ced-base, ced-small, MIT/ast-…, github.com/cwx-worst-one/EAT, github.com/fschmid56/EfficientAT), codesota.com AudioSet tablosu. mAP'ler farklı makalelerden; ±0.005 fark eşit sayılmalı. Hiçbiri bizim tarafımızdan ölçülmedi.

Notlar:
- CED'in GitHub deposu (RicherMans/CED) **GPL-3.0**; biz onu kullanmıyoruz. Hugging Face deposundaki `modeling_ced.py` ve `feature_extraction_ced.py` Apache 2.0 başlıklı; yalnız bunlar `trust_remote_code` ile yüklenir ve bu yalnız `mispeech/` kimlikleri için açılır (`audio_cue.TRUSTED_REMOTE_CODE`). Ek bağımlılık: `torchaudio`.
- CED sınıf adları kısa ("Crying", "Dishes"), AST tam ("Crying, sobbing"). Eşleştirme ilk virgüle kadar yapılır; prompts.yaml tam adları tutar.
- CED başı sigmoid'i kendisi uygulayabiliyor (doğrulanmadı); çıktı zaten 0..1 ise tekrar uygulanmaz.

Grup olasılığı = gruptaki sınıfların en yükseği; gruplar `config/prompts.yaml` → `audio_vocal`, `audio_violence`, `audio_kitchen`.

Hız: saniyede bir pencere (`hop_s`), 3 sn pencere. M5'te ölçülmedi.

## Bilinen sınırlar

- Sınıf adları AST'nin `id2label`'i ile web üzerinden karşılaştırıldı (hepsi var); CED'de 0-386 arası doğrulandı, üstü (silah, kırılma, Sizzle) yerel yüklemede kontrol edilecek. Modelde olmayan ad yüklemede uyarı olarak yazılır.
- Dizi müziği çığlığı bastırabilir; dublajlı içerikte efekt sesi kısık olabilir.
- Gerçek eşikler (`distress_full_at`, `kitchen_full_at`, boost, ceza) tahmin. Ayar H-07 ölçümüyle (yemek klipleri Emirhan, dizi sahneleri ortak).
- Canlı oynatıcıya (`sensa_ai.player`) henüz bağlı değil: demo çevrimdışı hatta (`analyze --audio`) ve onun ürettiği iz ile tarayıcı oynatıcısında çalışır.
- Ses saklama: yalnız sayılar (`.audiocue.json`) yazılır, ses kaydı ya da kesit yazılmaz.
