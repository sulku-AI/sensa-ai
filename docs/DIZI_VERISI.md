# Dizi verisiyle iyileştirme (2.1): görmeden ölç, görmeden düzelt

Sorun (FA-04): fotoğraf setinde yanlış alarmı düşük olan model, gerçek bir dizi bölümünde "neredeyse
her kırmızıya" kan diyor. Fotoğraf test seti dizi sahnesi değil, bu yüzden bu hatayı göstermedi.
Çözüm iki parça: diziyi **ölçmek** ve modeli dizinin kendi hatalarıyla **eğitmek**. İkisinde de
kimse kanlı kareye bakmaz; insan yalnız "hangi dakikalarda gerçekten kan vardı" bilgisini verir.

## 1. Ölç: işaret zaman çizelgesi

```bash
python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4
```

Saniyede bir kare alır, modelin "kan" dediği saniyeleri dakika dakika sayar. Çıktı bir tablo:
hangi dakikada kaç saniye işaretli. Görüntü yazmaz, göstermez. Sonuç
`experiments/results/<tarih>_isaret_dex1-1.md`.

İki modeli aynı karelerde karşılaştırmak için `--detector` iki kez verilir:

```bash
python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4 --detector rfdetr:models/rfdetr_seg_nano/checkpoint_best_total.pth --detector rfdetr:models/rfdetr_v2b/checkpoint_best_total.pth
```

Bölümü izleyen kişi gerçek kan dakikalarını kabaca yazarsa (`--ranges "12-14,38:10-41"`), araç iki
sayı daha verir:

| Sayı | Anlamı | Hedef |
|---|---|---|
| `disi_dakikada_yanlis_alarm` | gerçek kan dışında dakikada kaç ayrı yanlış bulanıklık olayı | `requirements.false_alarm_per_min` (params.yaml) |
| `ici_isaretli_oran`, `yakalanan_aralik` | gerçek kan aralıklarında işaretli saniye oranı, en az bir kez yakalanan aralık | yüksek |

Aralıklar kaba olduğu için bu sayılar kesin değil, ama iki modeli karşılaştırmaya yeter.

## 2. Düzelt: etiketleri gerçek dakikalarla temizle

```bash
python -m tools.autolabel data/pozitif/videolar --out data/pozitif/autolabel --min-score 0.2
python -m tools.relabel_events data/pozitif/autolabel --video data/pozitif/videolar/dex1-1.mp4 --ranges "12-14,38:10-41"
```

`--min-score 0.2`: oynatıcının güçlü profilde gördüğü yanlışların hepsi yakalansın diye düşük.
`relabel_events` aralık dışındaki her tespiti siler, o kare "bu kan değil" örneği olur (zor negatif).
Aralık içinde yalnız güçlü tespitler (skor ≥ 0.5) kan olarak kalır.

Sonra `autolabel.tar` yeniden yapılır, Colab defterinde `DIZI = True` ile eğitime girer (yalnız
train'e; ölçüme girmez).

## 3. Kansız video varsa: zor negatif madenciliği

Kan içermediği bilinen bir dizi ya da film (yasal kopya) en temiz veridir: modelin orada "kan" dediği
her kare yanlıştır, insan hiçbir şey işaretlemez.

```bash
python -m tools.autolabel data/kansiz_videolar --out data/pozitif/kansiz_zor --negative --min-score 0.2
```

Kareler kilitli `data/pozitif/` altına yazılır: video "kansız" denmiş olsa da modelin kan sandığı
kareler bunlar, kimse açmaz. Eğitimde `--neg data/pozitif/kansiz_zor` ile verilir. Bir videonun bütün kareleri aynı bölüme düşer
(sızıntı yok) ve kopya sınırı video karelerine uygulanmaz.

## Neden bu sıra

Önce ölçüm: hangi modelin daha iyi olduğunu tahminle değil, aynı bölümde dakikada yanlış alarm sayısıyla
söyleyebilmek için. Sonra eğitim: model tam yanıldığı sahnelerden öğrenir. Ses ipucu (D-019) bunun
üstüne ikinci bir süzgeçtir, veri düzeltmesinin yerini tutmaz.
