# Olumsuz sonuçlar ve bırakılan yollar

Her kayıt: yaklaşım, neden umut vericiydi, neden bırakıldı, ders.

## FA-01: Elle etiketli veriyle Faster R-CNN eğitimi (2026-10-17 → 2026-11-03)

- **Yaklaşım:** torchvision Faster R-CNN (ResNet-50 FPN), 3 sınıf (arka plan, blood, gore), MakeSense ile Pascal VOC XML kutuları, Colab'da eğitim. Kod değiştirilmeden `archive/2026-10-17_faster_rcnn/` altında.
- **Neden umut vericiydi:** Standart, iyi belgelenmiş yöntem; kutu tespiti doğrudan bölgesel bulanıklık verir.
- **Neden bırakıldı:** Veri seti için birinin yüzlerce kanlı kareyi açıp kutulaması gerekiyordu. Emirhan için bu mümkün değil (D-001); projenin eski engeli tam olarak buydu. Ayrıca kutu, kan gibi şekilsiz bölgeler için kaba; ve eğitilmiş model yalnız etiketlenen türü bilir.
- **Durum:** Eğitim hiç çalıştırılmadı (veri yoktu). `setup_project.py` sonunda Python dışı `git` satırları vardı (çalışmazdı).
- **Ders:** Önce görüntüyü kimsenin görmesi gerekmeyen yolu seç: metin istemli modeller + SAM 2 (D-006). Eğitim, hattın otomatik etiketleriyle damıtma olarak geri gelebilir (M5).
