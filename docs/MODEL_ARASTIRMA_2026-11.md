# OWLv2 ve EdgeTAM M5'i gerçek zamana taşır

Kasım 2026 itibarıyla metin istemiyle maske üretip videoda takip eden en güçlü açık ağırlıklı aile **SAM 3 / SAM 3.1**. Ancak bu ailenin lisansı özel "SAM License", HF erişimi kapılı ve boyutu yaklaşık 0,9B parametre. Bu yüzden MIT lisanslı sensa-ai'nin **varsayılan hattına girmemeli**, en fazla isteğe bağlı bir deney ya da çevrimdışı öğretmen olarak kalmalı. Apache-2.0 tarafında M5 için en mantıklı yeni varsayılan şu: **SigLIP base kapı → OWLv2 base dedektör → EdgeTAM takip** (`facebook/EdgeTAM`, `revision="refs/pr/1"`). OWLv2 hem M5'te ölçülen hızda (490 ms ve 742 ms) hem de SAM 3 makalesindeki kavram-maske kıyaslamasında (SA-Co/Gold cgF1 17,3 ve 6,5) LLMDet'in önünde. Projenin kendi bench formülüne göre OWLv2 ile hedef olan `realtime_factor ≥ 1,2`'ye ulaşmak için takipçinin 6 fps'te **kare başına en fazla ~398 ms** sürmesi gerekiyor. EdgeTAM M5'te henüz ölçülmedi. Ölçüm bu eşiği aşarsa, mevcut SAM 2.1 tiny ile `track.fps` 3'e indirmek aritmetik olarak 1,41 veriyor. Bu, EdgeTAM'den bağımsız, bugün uygulanabilir bir yedek. Bütün bu hesaplar "kareler kapıyı %20 oranında geçer" varsayımına dayanıyor. Oran %40'a çıkarsa takipli hiçbir mevcut kombinasyon hedefi tutturmuyor. Bu yüzden ölçülmesi gereken ilk büyüklük model hızından önce gerçek videodaki kapı geçiş oranı. "Kan" gibi amorf bölgelerde metin istemli modellerin başarısını doğrudan ölçen bir birincil kaynak yok. En yakın kanıt, SAM 3'ün cerrahi sahnelerde dil istemleriyle "currently suboptimal" kalması. Kalite sorusu ancak projenin kendi sayısal değerlendirmesiyle cevaplanabilir.

> Güvenlik notu: Bu raporda yalnız model kartı, makale özet sayfası (arXiv `abs`), repo ve dokümantasyon bağlantıları var. Veri seti sayfası, örnek galerisi ve figürlü makale tam metni bağlantısı bilerek verilmedi. Görsel betimleme yok.

## M5'te darboğaz takip; dedektör seçimi ikinci sırada

Projenin bench aracı video saniyesi başına analiz yükünü şöyle hesaplıyor: `sample_fps × (kapı + p × dedektör) + p × track.fps × takip`. Burada p, kapıyı geçen kare oranı (`tools/bench.py`). 2026-11-04 M5 ölçümlerine (1280×720, MPS, FP32) göre `sample_fps` 2 ve p 0,2 iken kapı ile dedektör birlikte LLMDet'te 0,457 s, OWLv2'de 0,356 s tutuyor. 6 fps'teki SAM 2.1 tiny takibi ise tek başına **0,704 s** ekliyor. Ölçülen 0,86 (LLMDet) ve 0,94 (OWLv2) değerlerinin nedeni bu. Takipsiz hat 2,19'a çıkıyor. Hedef 1,2 için toplam yükün ≤ 0,833 s olması gerekiyor. Buna göre takipçinin kare başı bütçesi:

| Dedektör | track.fps 6 | track.fps 4 | track.fps 3 |
|---|---|---|---|
| LLMDet tiny (742 ms) | ≤ 314 ms | ≤ 471 ms | ≤ 628 ms |
| OWLv2 base (490 ms) | ≤ 398 ms | ≤ 597 ms | ≤ 796 ms |

Tablo ölçülmüş bileşenlerden hesaplandı, kendisi bir ölçüm değil. İki sonuç çıkıyor. Birincisi, **SAM 2.1 tiny (587 ms) 3 fps takipte iki dedektörle de hedefi geçiyor**: LLMDet ile 1,24, OWLv2 ile 1,41. 6 fps'te ise ikisiyle de geçmiyor. İkincisi, EdgeTAM M5'te ~400 ms'nin altına inerse 6 fps takip korunabiliyor. 300 ms'de OWLv2 ile 1,40, 150 ms'de 1,87 çıkıyor. Bu iki değer varsayımsal örnek, ölçüm değil.

Hesabın en kırılgan girdisi p. Kanlı sahneler art arda geldiğinde p 0,4'e çıkarsa OWLv2 + SAM 2.1 tiny 3 fps'te 0,80'e düşüyor. Hedefi geçen tek senaryolar şunlar: takipsiz hat (OWLv2 ile 1,81, hesap) ve çok hızlı bir takipçi (150 ms ve 4 fps'te 1,26, varsayım). 30 sn'lik ileri bakma tamponu kısa yoğun sahneleri emer. Tampon tükenirse D-007 güvenli modu devreye girer. Yine de p'nin gerçek video dağılımında ölçülmesi, model değiştirmekten daha çok bilgi verir.

## EdgeTAM M5'in en olası çözümü, ama henüz ölçülmedi

EdgeTAM (Meta, CVPR 2025) SAM 2'nin cihaz üstü sürümü. **Kod ve checkpoint Apache-2.0**. iPhone 15 Pro Max'te 15,7 FPS, A100'de torch.compile ile 150,9 FPS ölçülmüş ve SAM 2'den "22× daha hızlı" deniyor ([facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)). J&F değerleri DAVIS17 87,7, SA-V test 71,7 ([transformers EdgeTAM Video](https://huggingface.co/docs/transformers/main/en/model_doc/edgetam_video)). Bu, SAM 2.1 tiny sınıfı doğruluk demek. transformers ağırlıkları `facebook/EdgeTAM` reposunun ana dalında değil, birleştirilmemiş **PR #1**'de duruyor (`model.safetensors` 55,9 MB) ([HF tartışma #1](https://huggingface.co/facebook/EdgeTAM/discussions/1), [refs/pr/1 dosya listesi](https://huggingface.co/facebook/EdgeTAM/tree/refs%2Fpr%2F1)). Koddaki `refs/pr/1` düzeltmesi bu nedenle doğru. Sınıflar `EdgeTamVideoModel` ve `Sam2VideoProcessor`. Dokümandaki `yonigozlan/edgetam-video-1` kimliği M5'te 401 vermişti (FA-02). Tekrarlanabilirlik için PR revizyonunu commit'e sabitlemek iyi olur (notlarda `f3a0979`).

Dikkat edilmesi gereken bir kod ayrıntısı var: `tracking.Tracker.from_pretrained` EdgeTAM yüklenemezse **sessizce SAM 2.1 tiny'ye düşüyor**. Bench çıktısındaki `takip_model` satırı `facebook/EdgeTAM` göstermiyorsa, ölçülen değer EdgeTAM'e ait değildir.

Hız beklentisi bir tahmin. iPhone'da ~16 FPS ve SAM 2.1 tiny'nin kabaca üçte biri boyutundaki ağırlık (56 MB, yaklaşık 14M parametre, tahmin), M5 MPS'te 587 ms'nin belirgin altını düşündürüyor. Ancak hiçbir kaynakta Apple Silicon MPS ölçümü yok. Takipte diğer seçenekler daha zayıf. EfficientTAM (Apache-2.0) 512 girişle iPhone'da 80,6 ms gösteriyor ama transformers formatı yok ([arXiv 2411.18933](https://arxiv.org/abs/2411.18933), [model kartı](https://huggingface.co/yunyangx/efficient-track-anything/blob/main/README.md)). SAM 2.1 tiny'nin MLX portu M2 Max'te video yayılımında Torch/MPS'ten 1,4–1,75× hızlı ([avbiswas/sam2.1-hiera-tiny-mlx](https://huggingface.co/avbiswas/sam2.1-hiera-tiny-mlx)). Bu 587 ms'yi kabaca 340–420 ms'ye indirebilir; bu da tahmin ve eşiğin hemen üstü ya da altı demek. SAMURAI, SAM2Long ve DAM4SAM doğruluğa ekleme yapan eğitimsiz eklentiler, hızı artırmıyor. Son 6 ayda açık ağırlıklı, cihaz üstü yeni bir takipçi bulunamadı. SAM-MT (Temmuz 2026) ve SAM2Dual (Ağustos 2026) yalnız özet düzeyinde biliniyor; ağırlık ve lisans durumları belirsiz ([SAM-MT](https://arxiv.org/abs/2607.08688v1), [SAM2Dual](https://arxiv.org/abs/2608.18640)).

## OWLv2 kavram kıyaslamasında LLMDet'i geçiyor; SAM 3 ikisini de geçiyor

SAM 3 makalesi, metinle verilen bir kavramın bütün örneklerini maskeleme görevinde (SA-Co/Gold, cgF1) şu skorları veriyor: **SAM 3 54,1**, OWLv2* 24,6, OWLv2 17,3, LLMDet-L 6,5, Grounding DINO-T 3,3, insan 72,8 ([SAM 3, arXiv 2511.16719](https://arxiv.org/abs/2511.16719)). Kıyaslama LLMDet'in en büyük sürümünü kullanıyor. Tiny sürümün daha da geride olması beklenir. Bu skorlar, mevcut varsayılan dedektörün (LLMDet tiny, D-011) seçilirken dayandığı LVIS kutu AP üstünlüğünün kavram düzeyindeki ve olumsuz istemli sorgulara taşınmadığını gösteriyor. M5'te OWLv2'nin %34 daha hızlı ölçülmesiyle birlikte düşünüldüğünde, **dedektör varsayılanını OWLv2 base'e çevirmek** hem hız hem kalite açısından tutarlı bir adım. Bunun bir kullanıcı kararı olarak `docs/DECISIONS.md`'ye yazılması gerekiyor.

SAM 3 ailesi kalite olarak açık ara önde. transformers'ta `Sam3Model`, `Sam3TrackerVideoModel` ve SAM3-LiteText (`sam3_lite_text`) sınıfları 2025-11-19'dan beri var. Varsayılan giriş 1008 px. `image_size=560` hızlandırıyor ama doğruluğu düşürebiliyor ([transformers SAM3](https://huggingface.co/docs/transformers/en/model_doc/sam3), [SAM3 tracker video](https://huggingface.co/docs/transformers/model_doc/sam3_tracker_video)). SAM 3.1 (27 Mart 2026) Object Multiplex ile çoklu nesne takibini hızlandırıyor ama transformers entegrasyonu yok ([facebook/sam3.1](https://huggingface.co/facebook/sam3.1)). MLX dönüşümü M3 Max'te 3 nesne takibinde **227 ms/kare** (SAM 3'te 547 ms), 224 px "gerçek zamanlı modda" ~40 ms/kare bildiriyor ([mlx-community/sam3.1-bf16](https://huggingface.co/mlx-community/sam3.1-bf16), ayna üzerinden okundu). Notlardaki tahmin, M5 Air'de aynı ayarın 60–120 ms süreceği. Bu doğrulanmadı ve 224 px küçük bölgeleri kaçırabilir. Engel lisansta. SAM License royalty-free ve ticari kullanımı yasaklamıyor. Ama dağıtımda lisans metninin eklenmesini istiyor, tersine mühendisliği ve askeri/silah kullanımlarını yasaklıyor, Meta'ya fesih ve tazminat hakkı veriyor ([facebook/sam3 LICENSE](https://huggingface.co/facebook/sam3/blob/main/LICENSE)). Model ayrıca kapılı ve F32 ağırlıkları 0,9B ([facebook/sam3](https://huggingface.co/facebook/sam3)). D-011 bu yüzden SAM 3'ü kod bağımlılığı olarak dışlamıştı. Yeni kanıt bu kararı değiştirmiyor. Ancak "kullanıcı kendi indirir, isteğe bağlı eklenti" seçeneği Emirhan'ın vereceği bir karar olarak açık.

**EfficientSAM3 / SAM3-LiteText** en ilginç ara yol. SAM 3'ü 0,68M–20,6M'lik görüntü kodlayıcılarına ve MobileCLIP metin kodlayıcılarına damıtıyor. LiteText, SA-Co/Gold'da SAM 3'ün 54,1'ine karşı 53,6–53,95 alıyor ve lisansı "Apache 2.0" olarak beyan ediliyor ([EfficientSAM3 README](https://huggingface.co/Simon7108528/EfficientSAM3/commit/47b2e1fee0fb635c78d248c7ebed3d8a6438a25f)). Üç belirsizlik var: SAM 3'ten damıtılmış ağırlıkların SAM License türevi sayılıp sayılmayacağı, kare başı gecikmenin hiç yayımlanmamış olması ve video takibinin henüz "planlı" olması.

Diğer adaylar ya lisans ya görev uyumu yüzünden eleniyor. **YOLOE-26-seg** metin istemli maske veriyor, 3,9M–55,2M parametre aralığında, LVIS 24,7–40,6 mAP alıyor ve CoreML'e aktarılabiliyor. Ancak lisansı **AGPL-3.0** ve D-006 ile dışlanmış durumda ([Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe), [YOLO26, arXiv 2606.03748](https://arxiv.org/abs/2606.03748)). **RF-DETR-Seg** bütün seg boyutlarında Apache-2.0 ve T4 TensorRT'de 3,4–21,8 ms sürüyor, ama kapalı küme bir model ve ince ayar gerektiriyor ([roboflow/rf-detr](https://github.com/roboflow/rf-detr)). Damıtma hedefi olarak D-011'deki yeri korunuyor. **Rex-Omni** ~3B parametreli bir MLLM, maske üretmiyor ve lisansı görünmüyor ([arXiv 2510.12798](https://arxiv.org/abs/2510.12798), [Rex-Omni-AWQ](https://huggingface.co/IDEA-Research/Rex-Omni-AWQ)).

### Model tablosu

Hız sütununda "M5" yazan değerler **projenin ölçümü**. Diğerleri kaynakta bildirilen değerler ve başka donanımda alınmış; tahmin olanlar ayrıca işaretli. ⚠ işareti bayraklı lisansı gösteriyor.

| Model | Lisans | Tarih | Boyut | HF id | transformers / MLX | Bilinen hız | Amorf bölge kanıtı |
|---|---|---|---|---|---|---|---|
| SAM 3 | ⚠ SAM License, kapılı | 2025-11 | 0,9B (F32) | `facebook/sam3` | `Sam3Model`, `Sam3TrackerVideoModel` / mlx-vlm | H200 ~30 ms (100+ nesne); M3 Max MLX 547 ms (3 nesne takip) | ADE-847 13,8 mIoU; cerrahide dil istemi "suboptimal" |
| SAM 3.1 | ⚠ SAM License, kapılı | 2026-03-27 | ~873M | `facebook/sam3.1` | yok / `mlx-community/sam3.1-bf16` | M3 Max 227 ms (3 nesne), 224 px ~40 ms | doğrudan yok |
| EfficientSAM3 / SAM3-LiteText | Apache-2.0 beyanı ⚠ (damıtma belirsiz) | 2026-01 / 2026-02 | kodlayıcı 0,68–20,6M + metin 42,6–123,6M | `Simon7108528/EfficientSAM3` | `sam3_lite_text` sayfası var (checkpoint eşleşmesi doğrulanmadı) / yok | kare başı yok | yok |
| OWLv2 base | Apache-2.0 | 2023 (doğrulanmadı) | ViT-B | `google/owlv2-base-patch16-ensemble` | `Owlv2ForObjectDetection` / doğrulanmadı | **M5 490 ms** | SA-Co/Gold 17,3 (kutu tabanlı) |
| LLMDet tiny | Apache-2.0 | CVPR 2025 | tiny | `iSEE-Laboratory/llmdet_tiny` | Grounding DINO sınıfları / doğrulanmadı | **M5 742 ms** | SA-Co/Gold 6,5 (L sürümü) |
| Grounding DINO tiny | Apache-2.0 | 2023 | tiny | `IDEA-Research/grounding-dino-tiny` | `GroundingDinoForObjectDetection` / doğrulanmadı | M5 ölçülmedi | SA-Co/Gold 3,3 |
| YOLOE-26-seg | ⚠ AGPL-3.0 (D-006 dışı) | 2026 | 3,9–55,2M | Ultralytics (HF değil) | yok / CoreML aktarımı var | belgede yok | yok |
| RF-DETR-Seg | Apache-2.0 (seg) | ICLR 2026 | Nano–2XL | GitHub `roboflow/rf-detr` | kendi paketi / yok | T4 TRT 3,4–21,8 ms | açık sözlük değil, ince ayar gerekli |
| Rex-Omni | ⚠ lisans görünmüyor | 2025-10 | ~3B | `IDEA-Research/Rex-Omni-AWQ` | qwen2_5_vl / doğrulanmadı | raporlanmamış | maske yok |
| **EdgeTAM** | Apache-2.0 | CVPR 2025; transformers 2025-09-29 | 56 MB (~14M, tahmin) | `facebook/EdgeTAM` @ `refs/pr/1` | `EdgeTamVideoModel` / CoreML encoder betiği | iPhone 15 Pro Max ~16 FPS; A100 150,9 FPS; **M5 ölçülmedi** | takipçi; istem kutudan gelir |
| SAM 2.1 tiny | Apache-2.0 | 2024 | ~39M | `facebook/sam2.1-hiera-tiny` | var / `avbiswas/sam2.1-hiera-tiny-mlx` | **M5 587 ms** | takipçi |
| EfficientTAM-Ti/S | Apache-2.0 | 2024-11 | 18M / 34M | `yunyangx/efficient-track-anything` | yok / yok | S@512: iPhone 80,6 ms | takipçi |

Lisans, sınıf adı ve hız değerlerinin kaynakları önceki bölümlerde verildi. Tarih sütunundaki "doğrulanmadı" işareti, bu turda birincil kaynağı açılmayan değerleri gösteriyor.

## Kapı 80 ms'den ~20 ms'ye inebilir ama kazancı sınırlı

Kapının mevcut payı video saniyesi başına 0,16 s. Bu, OWLv2'li hattın takipsiz yükünün %45'i. **PE-Core-B/16** (Meta, Nisan 2025) Apache-2.0, 0,09B görüntü parametreli ve IN-1k sıfır atışta %78,4 alıyor. timm sürümü hazır ([timm kartı](https://huggingface.co/facebook/pe_core_base_patch16_224_timm/blob/main/README.md), [arXiv 2504.13181](https://arxiv.org/abs/2504.13181)). **MobileCLIP2-B** %79,4 ile biraz daha iyi ve görüntü kodlayıcısı için 10,4 ms bildiriyor. Ancak lisansı ⚠ `apple-amlr` (araştırma lisansı) ve gecikmenin hangi cihazda ölçüldüğü belirtilmemiş ([MobileCLIP2-B](https://huggingface.co/apple/MobileCLIP2-B/blob/main/README.md)). Önceden hesaplanmış metin gömmeleriyle kapının ~20 ms'ye inmesi bir tahmin. Gerçekleşirse OWLv2 + 300 ms takip senaryosu 1,40'tan 1,68'e çıkar. Kazanç gerçek, ama takip ve p kadar belirleyici değil.

**ShieldGemma 2** (4B, ⚠ Gemma lisansı, kapılı) violence/gore politikasında F1 85,0 alıyor. LlavaGuard'ın skoru 57,0 ([arXiv 2504.01081](https://arxiv.org/abs/2504.01081), [model kartı](https://huggingface.co/google/shieldgemma-2-4b-it)). 4B'lik bir VLM gerçek zamanlı kapı olamaz. Yalnız çevrimdışı etiketleyici ya da kapı değerlendirmesinde referans olabilir. Kana özgü, iyi değerlendirilmiş ve 100M'den küçük açık bir sınıflandırıcı bulunamadı.

Apple tarafında, M5 GPU'sundaki Neural Accelerator'lara erişimin kanıtlanmış yolu **macOS 26.2+ üzerinde MLX**. M4'e göre TTFT'de 3,3–4,1× kazanç bildiriliyor, ama vision encoder ölçümü verilmiyor ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). PyTorch MPS'in bu birimleri kullandığına dair kanıt bulunamadı. Proje MPS'te FP32 çalışıyor (`device.dtype_for`). FP16 MPS'te denenmemiş bir kaldıraç.

### Doğrulanmış ile tahmin olanın ayrımı

| İddia | Durum (`params.yaml` sözlüğü) |
|---|---|
| SigLIP 80 ms, LLMDet tiny 742 ms, OWLv2 490 ms, SAM 2.1 tiny 587 ms, rf 0,86 / 0,94 / 2,19 | measured (M5, 2026-11-04) |
| Bütçe tablosu, rf 1,24 / 1,41 (3 fps), OWLv2 takipsiz 2,81 | sim (ölçümlerden formülle hesap) |
| SA-Co/Gold skorları, EdgeTAM iPhone/A100 hızı, SAM 3.1 M3 Max MLX hızı | literature |
| EdgeTAM M5'te < 400 ms; SAM 3.1 M5'te 60–120 ms; kapı ~20 ms; MLX portuyla SAM 2.1 tiny ~340–420 ms | estimate |
| p = 0,2 | estimate (gerçek videoda ölçülmedi) |
| Metin istemli modellerin "kan" gibi amorf bölgede başarısı | bilinmiyor; yalnız dolaylı kanıt (cerrahi, ADE-847) |

Amorf bölge sorusunda dolaylı kanıt iki yönlü. SAM 3 cerrahi sahnede nokta ve kutu istemleriyle SAM 2'yi geçiyor, ama dil istemlerinde "currently suboptimal" kalıyor ([arXiv 2512.07596](https://arxiv.org/abs/2512.07596v1)). Kutu tabanlı dedektörlerin şekilsiz bölgede kötü kutu üretmesi ve SAM tipi takipçinin bunu maskeyle telafi etmesi bir hipotez, ölçülmedi. Aynı fikrin 2025'teki denetimli taban çizgisi (Son, Cha ve Park, *Sensors* 25(7):2093; figür içerebileceği için bağlantı verilmedi) YOLOv8s-seg ile mask mAP50 0,718'de ve yakalama 0,655'te kalıyor. Projenin olay yakalama hedefi 0,95.

## Önerilen yeni M5 varsayılan hattı

**Kapı:** SigLIP base (`google/siglip-base-patch16-224`, ölçüldü 80 ms). PE-Core-B/16 ölçülüp daha hızlı ve en az eşit ayırıcı çıkarsa onunla değiştirilir.

**Dedektör:** **OWLv2 base** (`google/owlv2-base-patch16-ensemble`, Apache-2.0). `config/models.yaml` → `default.detect: owlv2_base`. LLMDet tiny karşılaştırma olarak kalır. Değişiklik yeni bir DECISIONS kaydı ister ve kullanıcı onayı gerekir.

**Takip:** **EdgeTAM** (`facebook/EdgeTAM@refs/pr/1`), `track.fps` 6. Bunun koşulu M5'te `takip_ms_kare ≤ ~350 ms` (398 ms eşiği ve pay). EdgeTAM 350–600 ms arası çıkarsa `track.fps` 4. 600 ms'yi aşarsa ya da yüklenemezse **SAM 2.1 tiny ile `track.fps` 3** (hesapla 1,41). Takip fps'i düşünce `frame_coverage` (hedef 0,90) hızlı harekette düşebilir. Bu yüzden vekil videoda sayısal kontrol gerekir.

**Dışarıda kalanlar:** SAM 3 / 3.1 (lisans ve kapı), YOLOE-26 (AGPL), Rex-Omni ve ShieldGemma 2 (gerçek zamanlı değil). EfficientSAM3/LiteText yalnız deney dalında, lisans kararı verildikten sonra.

**Ölçülecek sırayla.** Komutlar repo kökünde, `.venv` açıkken çalışır. Çıktılar yalnız sayı.

1. EdgeTAM'in gerçekten yüklendiğini ve kare başı süresini görmek için: `takip_model` satırı `facebook/EdgeTAM` olmalı, `[tracking] ... yüklenemedi` satırı çıkmamalı.
   ```
   python -m tools.bench
   ```
2. Önerilen dedektörle aynı ölçüm:
   ```
   python -m tools.bench --detector owlv2_base
   ```
3. Kapı geçiş oranına duyarlılık (p 0,2 → 0,4):
   ```
   python -m tools.bench --detector owlv2_base --gate-pass 0.4
   ```
4. Çözünürlüğün etkisi (480p):
   ```
   python -m tools.bench --detector owlv2_base --res 480
   ```
5. EdgeTAM sonucu 350 ms'yi aşarsa `config/params.yaml` → `track.fps` değerini 3.0 yap, yeniden ölç:
   ```
   python -m tools.bench --detector owlv2_base
   ```
6. Kurulu transformers'ın SAM 3 ve EdgeTAM sınıflarını içerdiğini doğrula:
   ```
   python -c "import transformers; print(transformers.__version__)"
   python -c "from transformers import EdgeTamVideoModel, Sam3Model; print('ok')"
   ```
7. MLX ve M5 hızlandırıcıları için macOS sürümü (26.2 ve üstü gerekli):
   ```
   sw_vers -productVersion
   ```
8. PE-Core-B/16 kapı adayının MPS gecikmesi (yükleme satırı karttakinden farklıysa kartı izle; doğrulanmadı):
   ```
   pip install -U timm
   python -c "import timm,torch,time; torch.set_grad_enabled(False); m=timm.create_model('hf-hub:facebook/pe_core_base_patch16_224_timm',pretrained=True).eval().to('mps'); x=torch.randn(1,3,224,224,device='mps'); [m(x) for _ in range(3)]; torch.mps.synchronize(); t=time.time(); [m(x) for _ in range(20)]; torch.mps.synchronize(); print(round((time.time()-t)/20*1000,1),'ms')"
   ```
9. OWLv2'nin negatif setteki yanlış alarm oranı (H-02; içerik izlenebilir):
   ```
   python -m tools.evaluate --neg data/negatif/coco --detector owlv2_base
   ```
10. OWLv2 ile pozitif sette yalnız sayısal yakalama ve kapsama (görüntü açılmaz):
    ```
    python -m tools.evaluate --pos data/pozitif/blood_seg_joost --neg data/negatif/coco --detector owlv2_base
    python -m tools.evaluate --pos data/pozitif/blood_seg_joost --neg data/negatif/coco --detector llmdet_tiny
    ```
11. Vekil videoda uçtan uca süre ve kapsama (izlenebilir):
    ```
    time python -m sensa_ai.pipeline analyze data/vekil/balon.mp4 --prompts proxy
    python -m tools.eval_video data/vekil --labels data/etiket --proxy
    ```
12. Sonuçları `config/params.yaml`'a `status: measured` ile (dosya ve tarih yazarak) işle. Dedektör değişikliğini ve SAM 3 eklenti sorusunu `docs/DECISIONS.md`'ye karar veren kişinin adıyla yaz. EdgeTAM başarısız olursa `docs/FAILED_APPROACHES.md`'ye ekle.

## Sonuç

Bu turun asıl dersi, gerçek zaman sorununun daha iyi bir dedektörle değil **takip bütçesi ve kapı geçiş oranıyla** çözüldüğü. Mevcut ağırlıklarla bile `track.fps` 3'e inmek hedefi aritmetik olarak geçiyor. EdgeTAM'in değeri, bunu 6 fps takipte ve yoğun sahnelerde pay bırakarak yapabilmesinde. Son 12 ayın büyük atılımı olan SAM 3 ailesi, kalitenin nereye gidebileceğini gösteriyor: kavram düzeyinde maskede OWLv2'nin üç katından fazla. Ama lisans ve boyut nedeniyle bugün yalnız bir tavan ölçüsü. Apache beyanlı damıtılmış türevlerin (EfficientSAM3/LiteText) olgunlaşması, bu tavanın MIT bir projeye taşınmasının en olası yolu.

Belirsizliğin merkezi kalite tarafında. Amorf bölgelerde hiçbir açık modelin metin istemiyle 0,95 olay yakalamaya yaklaştığına dair kanıt yok. Tek denetimli taban çizgisi 0,655'te kalıyor. Bu yüzden 9–11 numaralı sayısal değerlendirmeler, hız ölçümleri kadar öncelikli. Bir sonraki model kararı "hangisi hızlı" sorusuyla değil, OWLv2 ile LLMDet arasındaki yakalama farkıyla verilmeli.
