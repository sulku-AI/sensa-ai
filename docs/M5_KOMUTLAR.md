# M5'te çalıştırma (MacBook Air M5, okulda)

Bu sayfa sırayla izlenir. Her adımın çıktısı **yalnız sayı ya da kansız görüntüdür**; pozitif (kanlı) hiçbir şey ekrana gelmez. Ana hedef M5 (D-012). Bütün komutlar repo kökünde.

## 0. Güvenlik kuralları (önce oku)

- `data/pozitif/` klasörünü **Finder'da açma**, Quick Look (boşluk tuşu) kullanma. Finder küçük resim gösterir. Araçlar klasörü kilitler (yalnız sen ve betikler) ve Spotlight dizinlemesini kapatır.
- Roboflow Universe ve Hugging Face'in **veri seti sayfalarını tarayıcıda açma**, örnek görüntü gösterirler. API anahtarı için yalnız roboflow.com → Settings → API Keys sayfası (görüntü yok).
- `previews/` altındaki her şey `safe_preview`'dan geçmiş, gri ve pikselleştirilmiş küçük resimdir; yine de yalnız negatif ve vekil içerik için üretilir.

## 1. Kurulum (bir kez, ~10 dk, internet gerekir)

Komutları tek tek yapıştır. Satır sonuna `#` açıklaması koyma: zsh onu komutun parçası sanar.

```bash
brew install python@3.12 ffmpeg
cd ~
git clone https://github.com/sulku-AI/sensa-ai.git
cd sensa-ai
git checkout claude/repo-yapisi-ve-plan
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python --version
pip install -U pip
pip install -e ".[dev,models,player,data,train]"
python -m pytest
```

`python --version` **3.12.x** demeli (3.14 değil; torch ve bazı paketler 3.14'te henüz sorunlu). `pytest` 40+ test çalıştırır, hepsi sentetik. Bütün sonraki komutlar `sensa-ai` klasöründe, `.venv` açıkken.

İlk model çalıştırmasında ağırlıklar Hugging Face'ten iner (LLMDet tiny, SigLIP base, EdgeTAM: toplam ~1.5 GB, tahmin).

## 2. Hız ölçümü (H-03), ~5 dk

```bash
python -m tools.bench
python -m tools.bench --detector owlv2_base
python -m tools.bench --no-track --res 480
```

- `python -m tools.bench`: SigLIP + OWLv2 + EdgeTAM, MPS (D-014)
- `python -m tools.bench --detector owlv2_base`: karşılaştırma
- `python -m tools.bench --no-track --res 480`: daha hafif ayar

Sonuç: `experiments/results/<tarih>_bench_mps.md`. `realtime_factor` ≥ 1.2 olmalı. Altındaysa sırayla: `--res 480`, `config/params.yaml` → `lookahead.sample_fps` 1.0, `--detector owlv2_base`.

## 3. Vekil video ile oynatıcı (sen izleyebilirsin)

```bash
python -m tools.make_proxy_video data/vekil/balon.mp4
python -m sensa_ai.player data/vekil/balon.mp4 --proxy
python -m sensa_ai.pipeline analyze data/vekil/balon.mp4 --prompts proxy
python -m sensa_ai.player data/vekil/balon.mp4 --track data/vekil/balon.blurtrack.json
```

- `python -m tools.make_proxy_video data/vekil/balon.mp4`: 20 sn, kırmızı balon hareket eder
- `python -m sensa_ai.player data/vekil/balon.mp4 --proxy`: modelsiz, anında
- `python -m sensa_ai.pipeline analyze data/vekil/balon.mp4 --prompts proxy`: gerçek modeller, vekil istem

Hassasiyet profili: `--profile hafif`, `orta` (varsayılan) ya da `guclu` (daha çok ve daha erken bulanıklık); değerler `config/params.yaml` → `profiles`. Ses PyAV ile çalınır ve oynatıcının saatidir; ses aygıtı yoksa sessiz oynar.

Kendi çektiğin kırmızı balon / kırmızı top videosu da olur. Bakılacak: bulanıklık bölge görünmeden önce başlıyor mu, kenarlar yumuşak mı, ileri sarınca "analiz yetişiyor" bulanıklığı makul sürede kalkıyor mu (H-04).

## 3b. Tarayıcı oynatıcısı (sen izleyebilirsin)

```bash
python -m tools.make_proxy_video data/vekil/balon.webm
python -m sensa_ai.pipeline analyze data/vekil/balon.webm --proxy
python -m tools.web_player data/vekil/balon.webm
python -m tools.web_player --profile guclu
```

- `python -m tools.make_proxy_video data/vekil/balon.webm`: Chrome mp4v çalmaz; .webm VP8 yazar
- `python -m tools.web_player data/vekil/balon.webm`: tarayıcı açılır: http://127.0.0.1:8765/
- `python -m tools.web_player --profile guclu`: dosyaları sayfada seç

Sayfa yalnız bulanıklaştırılmış tuvali gösterir; iz yüklenmeden ekran siyahtır. Boşluk tuşu oynat/duraklat.

## 4. Negatif set (H-02, sen izleyebilirsin)

```bash
python -m tools.fetch_negatives --per-class 40
python -m tools.evaluate --neg data/negatif/coco --previews
```

- `python -m tools.fetch_negatives --per-class 40`: COCO: domates, şarap, kırmızı ışık, pizza...

Sonuç: yanlış alarm oranı. `--previews` yanlış alarm veren negatiflerin güvenli küçük resmini `previews/negatif/` altına yazar.

## 5. Pozitif set (H-01). Sen görmeden, yalnız sayı

```bash
export ROBOFLOW_API_KEY=...
python -m tools.fetch_datasets --list
python -m tools.fetch_datasets blood_seg_joost blood_pure_blur
python -m tools.evaluate --pos data/pozitif/blood_seg_joost --pos data/pozitif/blood_pure_blur \
    --neg data/negatif/coco --detector llmdet_tiny
python -m tools.evaluate --pos data/pozitif/blood_seg_joost --neg data/negatif/coco --detector owlv2_base
```

- `export ROBOFLOW_API_KEY=...`: roboflow.com > Settings > API Keys

Sonuç: `experiments/results/<tarih>_eval_*.md`. Bakılacak satırlar: `yakalama_orani` (hedef 0.95), `ortalama_kapsama` (hedef 0.90), `negatif_yanlis_alarm_orani`. Bu dosyayı bana (Claude'a) da verebilirsin, eşikleri ben ayarlarım.

## 5b. Video düzeyinde değerlendirme (ortak etiketledikten sonra)

Ortak pozitif videoları izleyip `data/etiket/<video>.json` yazar (biçim `tools/eval_video.py` başında; yalnız zaman aralığı, görüntü yok).

```bash
python -m tools.eval_video data/pozitif/videolar --labels data/etiket
python -m tools.eval_video data/vekil --labels data/etiket --proxy
```

- `python -m tools.eval_video data/vekil --labels data/etiket --proxy`: vekil videolarla deneme

Sonuç: olay yakalama, kare kapsama, geç başlama (saniye), olay dışı bulanıklık. Yalnız sayı.

## 5c. Tanıtım klibi (Instagram, kansız)

```bash
python -m tools.before_after data/vekil/balon.mp4 previews/balon_once_sonra.mp4 --proxy --vertical
```

Pozitif klasörden video verilirse araç reddeder.

## 6. Eğitilmiş model (RF-DETR-Seg) Mac'te

Eğitim Colab'da yapılır (`notebooks/colab_sayi.ipynb`); Mac ısınıyor. Model dosyası Drive'da: `MyDrive/sensa_ai/rfdetr/checkpoint_best_total.pth`. Bu bir ağırlık dosyası, görüntü değil.

1. Drive'dan `checkpoint_best_total.pth` dosyasını indir.
2. Terminalde (dosya İndirilenler'deyse):

```bash
cd ~/sensa-ai
source .venv/bin/activate
git pull
pip install "rfdetr>=1.11"
mkdir -p models/rfdetr_seg_nano
mv ~/Downloads/checkpoint_best_total.pth models/rfdetr_seg_nano/
python -m tools.bench --detector rfdetr_seg --res 720 --frames 20 --no-gate
python -m tools.bench --detector rfdetr_seg --res 720 --frames 20 --no-gate --no-track
```

Çıkan `realtime_factor` iki satırını gönder (hedef 1.2). İkincisi takipsiz: model kendi maskesini verdiği için takip belki gerekmez.

Not: eğitilmiş model kan için eğitildi; kırmızı balonlu vekil videoda bölge bulmaması normal. Vekil oynatıcı testi OWLv2 ile sürer.

## 7. Bir sorun çıkarsa

- `No module named 'cv2'`: eğitim paketi `opencv-python-headless` da kurmuş olabilir, iki paket aynı `cv2` klasörünü paylaşır. Düzeltme:
  `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python` ve ardından `pip install --force-reinstall --no-deps opencv-python`.

Hata metnini (yalnız terminal metni, görüntü değil) thread'e yapıştır. `experiments/results/` dosyaları da olur.
