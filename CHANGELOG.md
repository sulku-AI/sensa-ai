# Değişiklik günlüğü

## 2.1.0 (2026-11-08)

İlk gerçek dizi denemesinden (FA-04) sonra: dizide ölçmek, dizinin kendi hatalarından öğrenmek ve
hatta yük olan yerleri hızlandırmak. Sürüm numarası kullanıcının isteğiyle (D-021).

**Yeni**
- `tools/flag_timeline`: modelin bir videoda nerede "kan" dediğini dakika dakika sayar, `--judge` ile
  bağımsız ikinci görüş (SigLIP, OWLv2) alıp aday kan dakikaları önerir, birden çok
  modeli aynı karelerde karşılaştırır; gerçek kan dakikaları verilirse dizide dakikada yanlış alarm
  ölçer. Görüntü yazmaz ([docs/DIZI_VERISI.md](docs/DIZI_VERISI.md)).
- `tools/relabel_events`: autolabel etiketlerini kabaca bilinen gerçek kan dakikalarıyla düzeltir;
  aralık dışındaki tespitler zor negatif olur, dakikada en çok `dizi.neg_per_min`. `--unsure`: emin
  olunmayan dakikalar eğitimden çıkar (gerçek dakikalar yerine hakem önerisi kullanıldığında, D-022).
- `flag_timeline --judge` iki öneri yazar: aday kan dakikaları ve belirsiz dakikalar (params `dizi`);
  `--unsure` yanlış alarm sayımında da geçerli. Sonuç dosyası eşik ve hakem bilgisini taşır, aynı gün
  yapılan önceki ölçümün üstüne yazmaz; model sütunları kısa ad (`rfdetr_v2b/best_ema`).
- `tools/autolabel --negative`: kansız videodan zor negatif madenciliği. Her karede `time_s`, dakika
  başına işaretli saniye özeti, uzun videoda ilerleme satırı, `--profile`.
- Web oynatıcı: Esc / b ile anında siyah ekran (masaüstü oynatıcıdaki gibi), "Karart" düğmesi.
- Colab: dizi kareleri isteğe bağlı (`DIZI`), yeni modeli ilk modelle aynı test setinde karşılaştıran
  kutu (3b), paylaşılan Drive klasörünü kısayoldan bulma.

**Hız**
- `config.load` ayrıştırılmış YAML'ı dosya değişene kadar saklar: çağrı başına ~6 ms → ~0,1 ms.
  Dedektörler bunu her karede çağırıyordu (RF-DETR karesi ~25 ms, yani ~%20 ek yük).
- Canlı oynatıcı analizi her örnekte video araması yapmıyor; yakın ilerideki kareye çözücüyle ilerliyor
  (H.264'te her arama önceki anahtar kareden yeniden çözmek demek).
- Çevrimdışı `analyze` (takipsiz) ve `autolabel` örneklenmeyen kareleri yalnız atlıyor (`grab`).

**Düzeltme**
- Eğitim hazırlığı: negatif video kareleri video başına tek grup (önceden tek tek bölünüyordu, aynı
  sahne hem train'de hem test'te olabiliyordu); video karelerine artırma kopyası sınırı uygulanmıyor
  (önceden bir videonun yalnız 2 karesi kalabiliyordu).
- Model yüklenirken torch.jit / izleme uyarıları gizleniyor (zararsızdı ama ekranı dolduruyordu).
- `autolabel --min-score` dedektörün kendi eşiğinin altına inmiyordu: 0.2 verilse de RF-DETR 0.3'ün
  altını döndürmüyordu. Artık dedektör eşiği de iner. Klasör yerine tek video dosyası da kabul.
- Eğitim hazırlığı `relabel_events`'ten geçmemiş dizi karelerini (`--train-only`) reddeder (FA-04).

**Bilinen sınırlar**
- Dizide gerçek kan dakikaları bilinmiyor; dex1-1 için hakem önerisi kullanılıyor (5. ve 27. dk, D-022).
  Bu dakikalarla eşik 0.3'te kan dışında dakikada yanlış alarm: v1 3.36, v2b 2.52 (hedef 0.5); ikisi de iki
  aralığı yakalıyor ([sonuç](experiments/results/2026-11-08_isaret_dex1-1_hakem_v2b.md)). Dizi
  kareleriyle eğitim (v2b_dizi) bekleniyor.

## 0.1.0 (2026-11-03 – 2026-11-07)

M0–M5: yapı ve plan, kansız test hattı, model sarmalayıcıları, 30 sn ileri bakan oynatıcı, web
oynatıcı, RF-DETR-Seg Nano eğitimi (ayrılmış testte yakalama 0,957, yanlış alarm 0,026, küçük set),
veri denetimi ve kontrollü görüntüleyici. Ayrıntı: [docs/DECISIONS.md](docs/DECISIONS.md) D-001…D-020.
