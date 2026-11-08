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

Aralıklar kaba olduğu için bu sayılar kesin değil, ama iki modeli karşılaştırmaya yeter. Emin olunmayan
dakikalar `--unsure` ile sayımdan çıkarılır.

### Gerçek dakikalar bilinmiyorsa: hakem önerisi

```bash
python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4 --min-score 0.3 --judge siglip --judge owlv2_base
```

İlk modelin işaretlediği her saniyeye iki bağımsız model daha bakar: SigLIP ("kan mı, ketçap / boya /
kırmızı kumaş mı") ve OWLv2 (metinle "blood"). Üçünün birlikte "kan" dediği saniyeler `ortak` sütunu.
Araç iki öneri yazar (eşikler params `dizi`):

- **aday kan dakikaları**: en az `judge_min_s` ortak saniyesi olan dakikalar → `--ranges`
- **belirsiz dakikalar**: daha az ortak saniyesi olan dakikalar ve aday dakikaların önü / arkası → `--unsure`

Belirsiz dakikalar ne kan ne "kan değil" sayılır: eğitimden de yanlış alarm sayımından da çıkar. Bu bir
model önerisi; hakemlerin kaçırdığı gerçek kan "kan değil" diye öğretilebilir (D-022). İzleyen biri
dakikaları kabaca biliyorsa onun bilgisi her zaman önce gelir. dex1-1 sonucu:
[2026-11-08_isaret_dex1-1_hakem_v2b.md](../experiments/results/2026-11-08_isaret_dex1-1_hakem_v2b.md).

## 2. Düzelt: etiketleri gerçek dakikalarla temizle

Mac'te, repo klasöründe (`~/sensa-ai`):

```bash
python -m tools.autolabel data/pozitif/videolar/dex1-1.mp4 --out data/pozitif/autolabel --min-score 0.2 --detector rfdetr:models/rfdetr_v2b/checkpoint_best_ema.pth
python -m tools.relabel_events data/pozitif/autolabel --video data/pozitif/videolar/dex1-1.mp4 --ranges "5-6,27-28" --unsure "4-5,6-7,16-17,19-20,26-27,28-30,38-39"
COPYFILE_DISABLE=1 tar -cf autolabel.tar data/pozitif/autolabel
```

- `--min-score 0.2`: oynatıcının güçlü profilde gördüğü yanlışların hepsi yakalansın diye düşük; dedektörün
  kendi eşiği de 0.2'ye iner (2.1.0'dan önce 0.3'te kalıyordu). Zor negatifler eğitilen modelin kendi
  yanlışları olsun diye en yeni model (v2b) kullanılır.
- `relabel_events` aralık dışındaki her tespiti siler, o kare "bu kan değil" örneği olur (zor negatif).
  Bir dakikada en çok `dizi.neg_per_min` tanesi kalır (eşit aralıklı): art arda saniyeler neredeyse aynı
  kare, tek bölüm eğitimi boğmasın. Aralık içinde yalnız güçlü tespitler (skor ≥ 0.5) kan olarak kalır.
  Belirsiz aralıktaki kareler atılır. Tekrar çalıştırmak güvenli: hep özgün dosyadan uygulanır.

`autolabel.tar` Drive'da `sensa_ai/` klasörüne konur (eski paketin yerine; Drive boyutu MiB gösterir, 119 MB'lık
dosya ~113 MB görünür), Colab defterinde `DIZI = True` ile eğitime girer (yalnız train'e; ölçüme girmez).
Defter açtığı paketin boyutunu ve hangi videoların düzeltildiğini yazar. Mac tar'ının eklediği
`LIBARCHIVE.xattr` notları zararsızdır. Eğitim hazırlığı (`prepare --train-only`) `relabel_events`'ten geçmemiş
bir videonun karelerini reddeder: ham etiketler modelin kırmızı yanlışlarını içerir (FA-04).

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
