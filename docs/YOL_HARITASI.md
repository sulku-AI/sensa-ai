# Yol haritası: demo adımları (2026-11-03)

Her adımın bir **çıkış kapısı** var: kapı geçilmeden sonrakine geçilmez. Süreler tahmin; çalışma saatleri ~13.30-16.00 ve ~23.00-03.00, isoshift günde 2-3 saat alıyor.

| Adım | İş | Çıkış kapısı | Kim bakar |
|---|---|---|---|
| **M0** (bu PR) | Repo yapısı: params.yaml, karar defteri, güvenli önizleme, iz biçimi, bölgesel bulanıklık, vekil uçtan uca test, CI | CI yeşil; `pytest` vekil videoda izi bulur ve bulanıklaştırır | CI |
| **M1** | Çevrimdışı gerçek hat: SigLIP + Grounding DINO + SAM 2.1 `Detector` arkasında; `analyze` → `.blurtrack.json`, `render` → bulanık mp4 (ses ffmpeg ile kopyalanır); maske RLE izde | Vekil istemle ("red balloon") gerçek bir videoda doğru bölge bulanıklaşıyor | **Emirhan izler** (vekil) |
| **M2** | Gerçek istemler; negatif set (H-02) ve pozitif set metrikleri (H-01, H-05); Grounding DINO ve OWLv2 karşılaştırması; eşik ayarı | H-01 ve H-02 ölçütleri; sonuç `experiments/results/` | Negatif: Emirhan. Pozitif: model + ortak |
| **M3** | Canlı oynatıcı: PyAV ile çözme, ses saati, PySide6 ekran; analiz iş parçacığı oynatma kafasının 30 sn önünde (`lookahead.next_window`); tampon kısalınca güvenli mod (D-007); ileri sarmada yeniden tarama | H-03: gerçek zamandan hızlı, tampon düşmüyor; vekil videoda canlı bulanıklık | Emirhan (vekil) |
| **M4** | Hassasiyet ve bulanıklık gücü ayarı (kişi profili), H-04 ve H-06; Instagram için kansız önce/sonra klipleri (vekil ve negatif) | Ortağın kontrol listesinde 0 sızan olay; Emirhan bulanıklığı "rahatsız etmiyor" der | Ortak + Emirhan (vekil/negatif) |
| **M5** | Damıtma: hattın otomatik etiketleriyle küçük hızlı model; tarayıcı eklentisi; platformlar için iz biçimi belgesi | Ayrı plan | — |

## M0 sonrası hemen yapılabilecekler

1. Emirhan'ın ekran kartını öğrenmek (model boyutu buna bağlı).
2. Vekil ve negatif klip toplamak: 5-10 kısa video (kırmızı balon, ketçap, kırmızı boya, şarap, kırmızı ışıklı sahne).
3. Ortak bulmak (pozitif seti etiketleyecek ve son kontrolü yapacak kişi).

## Açık sorular

- Yeni isim (D-008).
- Güvenli mod: bulanıklık mı, duraklatma mı (D-007).
- TEKNOFEST Engelsiz Yaşam şu an isoshift'e ayrılmış; tek başvuru kuralı kontrol edilmedi.
- Açık kaynak destek programları (AI şirketleri) doğrulanmadı.
