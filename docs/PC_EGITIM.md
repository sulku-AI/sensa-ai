# Linux PC'de eğitim (RTX 2060)

Ücretsiz Colab sık koptuğu için (8 Kas 2026: toplam ~1 saat çalıştı) dizi modeli PC'de eğitilir.
PC: Ryzen 5 5600X, 16 GB RAM, RTX 2060 6 GB, Windows/Linux dual boot.

Bölümler (videolar) Mac'te kalır. Etiketleme ve ayrılmış bölüm ölçümü Mac'te yapılır
([DIZI_VERISI.md](DIZI_VERISI.md)). PC'ye yalnız iki arşiv gider:

- `veri.tar`: fotoğraf setleri ve negatifler (Drive `sensa_ai/`)
- `autolabel.tar`: düzeltilmiş dizi kareleri (Mac `~/sensa-ai/`, USB ya da Drive ile)

Komutlar tek tek, açıklamasız yazılır.

## 1. Kurulum (bir kez)

Ekran kartı sürücüsü: `nvidia-smi` tabloyu göstermeli. Göstermezse (Ubuntu):

```bash
sudo ubuntu-drivers install
sudo reboot
```

Repo ve ortam:

```bash
sudo apt install git python3-venv tmux
git clone https://github.com/sulku-AI/sensa-ai.git ~/sensa-ai
cd ~/sensa-ai
git checkout claude/repo-yapisi-ve-plan
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,models,train]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Son satır `True NVIDIA GeForce RTX 2060` yazmalı.

## 2. Veri

`veri.tar` ve `autolabel.tar` `~/sensa-ai/` içine konur. Sonra:

```bash
tar -xf veri.tar
rm -rf data/pozitif/autolabel
tar --warning=no-unknown-keyword -xf autolabel.tar
python -m tools.train_rfdetr prepare --src data/pozitif/blood_seg_joost data/pozitif/blood_pure_blur --train-only data/pozitif/autolabel --neg data/negatif/coco data/negatif/coco_zor --aug-per-image 2
```

prepare relabel'den geçmemiş bir bölüm görürse durur (FA-04). Bu bir hata değil, korumadır:
Mac'te paket yeniden kurulur.

## 3. Deneme (1 tur)

```bash
bash tools/pc_egit.sh 1
```

Bir turun süresine bakılır ve 30 turun gece boyu sığıp sığmadığı hesaplanır. Bellek yetmezse
(`CUDA out of memory`) `BATCH=2 ACCUM=8` ile denenir. Deneme çıktısı silinir:

```bash
rm -rf models/rfdetr_v2b_dizi
```

## 4. Gece eğitimi

Masaüstünü kapatıp yalnız metin ekranında çalıştırmak ekran kartı belleğini ve işlemciyi eğitime
bırakır. Uyku kapatılır:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl isolate multi-user.target
```

Metin ekranında kullanıcıyla girilir. Sonra:

```bash
cd ~/sensa-ai
source .venv/bin/activate
tmux new -s egitim
KAPAT=1 bash tools/pc_egit.sh
```

`KAPAT=1` bitince bilgisayarı kapatır; açık kalsın istenirse yazılmaz. Betik kopmada en çok 5 kez
`last.ckpt`'ten devam eder. Kayıt: `models/rfdetr_v2b_dizi/egitim.log`.

Ertesi gün masaüstü ve uyku geri:

```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl isolate graphical.target
```

## 5. Sonuç

`models/rfdetr_v2b_dizi/checkpoint_best_ema.pth` (ya da `checkpoint_best_total.pth`) Drive
`sensa_ai/rfdetr_v2b_dizi/` klasörüne yüklenir. Mac bunu indirip v1 ve v2b ile aynı komutla ölçer
(ayrılmış bölümde ve dex1-1'de). Varsayılan modeli kullanıcı seçer.

## Elektrik

2060 eğitimde ~160 W, bütün PC ~250 W (tahmin, ölçülmedi). Gece 8 saat ≈ 2 kWh.
