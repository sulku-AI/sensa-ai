# Değişiklik günlüğü

## 2.1.0 (2026-11-08)

İlk gerçek dizi denemesinden (FA-04) sonra: dizide ölçmek, dizinin kendi hatalarından öğrenmek ve
hatta yük olan yerleri hızlandırmak. Sürüm numarası kullanıcının isteğiyle (D-021).

**Yeni**
- `tools/flag_timeline`: modelin bir videoda nerede "kan" dediğini dakika dakika sayar, birden çok
  modeli aynı karelerde karşılaştırır; gerçek kan dakikaları verilirse dizide dakikada yanlış alarm
  ölçer. Görüntü yazmaz ([docs/DIZI_VERISI.md](docs/DIZI_VERISI.md)).
- `tools/relabel_events`: autolabel etiketlerini kabaca bilinen gerçek kan dakikalarıyla düzeltir;
  aralık dışındaki tespitler zor negatif olur.
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

**Bilinen sınırlar**
- Dizideki yanlış alarm henüz ölçülmedi; ilk ölçüm `flag_timeline` ile, gerçek kan dakikaları
  gelince. v2b modelinin (zor negatifli) sonuçları bekleniyor.

## 0.1.0 (2026-11-03 – 2026-11-07)

M0–M5: yapı ve plan, kansız test hattı, model sarmalayıcıları, 30 sn ileri bakan oynatıcı, web
oynatıcı, RF-DETR-Seg Nano eğitimi (ayrılmış testte yakalama 0,957, yanlış alarm 0,026, küçük set),
veri denetimi ve kontrollü görüntüleyici. Ayrıntı: [docs/DECISIONS.md](docs/DECISIONS.md) D-001…D-020.
