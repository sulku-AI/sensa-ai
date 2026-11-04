# Test planı: kan görmeden test (2026-11-03)

**Kural (D-001):** Emirhan'a hiçbir zaman kanlı görüntü gösterilmez ve izlemesi istenmez. Bu plan, her testin Emirhan'ın gözü olmadan yapılabildiği şekilde kuruldu. Eşik sayıları `config/params.yaml` → `requirements`.

## 1. Kim neye bakar

| Malzeme | Emirhan | Ortak (sinestezisi olmayan) | Model / Claude |
|---|---|---|---|
| Vekil videolar (kırmızı balon) ve sonuçları | **İzler** | İzler | Çalıştırır |
| Negatif set (ketçap, sos, boya, şarap, kırmızı ışık, ruj, kırmızı kumaş, çiğ et) ve sonuçları | **İzler** | İzler | Çalıştırır |
| Pozitif set (kan/vahşet) ham hali | **Asla** | Etiketler | Çalıştırır, sayar |
| Pozitif setin bulanıklaştırılmış çıktısı | **Asla** (iyi bulanıklaşmış olsa bile kaçan kare olabilir) | İzler, kaçanları işaretler | Kontrol eder |
| Sayılar, grafikler, `safe_preview` küçük resimleri | Bakar | Bakar | Üretir |
| Metin özetleri | Okur | Yazar | Yazar |

Metin özetleri **nötr şablonla** yazılır, betimleme yok: `klip 07 · 00:41-00:46 · sınıf: kan · bölge: sol alt · kapsama %82 · kaçan kare: 3`. "Korkunç", "derin", "parçalanmış" gibi sıfatlar ve ne olduğunun anlatımı yazılmaz.

## 2. Kodda güvence (D-004)

- İnsan için görüntü yazmanın tek yolu `safety.write_preview`: 160 px, gri ton, 16 px bloklar. Kırmızı renk ve ayrıntı tanınmaz. `tests/test_safety.py` bunu ve başka modülün `imwrite` kullanmadığını sınar.
- `data/pozitif/` yerelde, repoya girmez. CI `no-media` işi repoda video/görüntü olursa kırmızı yanar.
- Testler yalnız sentetik renkli şekiller üretir (`tests/conftest.py`).
- Hata ayıklama araçları ham kare göstermez. Pencere açan araçlar: oynatıcı (Emirhan'ın kendi videoları, bulanıklık açık) ve `tools/label_events` (pozitif klasörde yalnız `--ortak` bayrağıyla açılır).
- `safety.write_dataset_frame` ham kareyi yalnız kilitli `data/pozitif/` altına yazar (klasör 700, Spotlight kapalı).
- `tools/before_after` (tanıtım klibi) pozitif klasörden video kabul etmez.
- Tarayıcı oynatıcısında ham `<video>` görünmez (opaklık 0); iz yüklenmeden tuval siyah. `tests/test_web.py` gerçek tarayıcıda (Playwright varsa) bölge içinin bulanık, dışının keskin olduğunu sentetik dama deseniyle ölçer.

## 2b. Araçlar ve hangi hipotezi ölçtükleri

| Araç | Hipotez | Çıktı |
|---|---|---|
| `tools/bench` | H-03 | ms/kare, realtime_factor (sentetik kareler) |
| `tools/evaluate` | H-01, H-02 (görüntü) | yakalama, kapsama, yanlış alarm oranı |
| `tools/label_events` | (ortak) | `data/etiket/*.json`, yalnız zaman |
| `tools/eval_video` | H-01, H-05, H-06 (video) | olay yakalama, kare kapsama, geç başlama, olay dışı bulanıklık |
| `tools/before_after` | H-04 (vekil/negatif) | önce/sonra klibi |
| `tools/web_player` | H-04 (tarayıcıda) | yerel tarayıcı oynatıcısı; ham video gizli, yalnız bulanık tuval (D-013) |

## 3. Hipotezler

**Durum değerleri:** `AÇIK` · `DESTEKLENDİ` · `ÇÜRÜTÜLDÜ` · `KISMEN`. 3 Kasım 2026 itibarıyla hepsi `AÇIK`. Sonuçlar `experiments/results/` altına tarihli yazılır.

| ID | İddia | Test | Geçme ölçütü | Kim bakar |
|---|---|---|---|---|
| H-01 | Metin istemli modeller kanlı bölgeleri yakalar | Pozitif sette olay bazında yakalama ve kapsama; etiket = ortağın zaman aralıkları | `requirements.event_recall`, `frame_coverage` | Model + ortak |
| H-02 | Kansız kırmızı şeylerde az yanlış alarm | Negatif sette dakikada yanlış bulanıklık | `requirements.false_alarm_per_min` | **Emirhan izleyebilir** |
| H-03 (KISMEN, 2026-11-04: M5'te 1.95, sentetik kare, kapı oranı varsayım; `experiments/results/2026-11-04_bench_mps_edgetam_owlv2.md`) | Dizüstü ekran kartında gerçek zamandan hızlı | Analiz süresi / video süresi, tampon hiç `safe_margin_s` altına inmiyor | `requirements.realtime_factor` | Sayı |
| H-04 | Bulanıklık izlemeyi bozmuyor | Vekil ve negatif klipler; bulanıklık doğal mı, rampa yeterince önce mi | Emirhan + ortak "rahatsız etmiyor" der | **Emirhan izleyebilir** |
| H-05 | Karanlık sahnede de çalışır | Pozitif setin karanlık alt kümesi | H-01 ölçütü karanlıkta da | Model + ortak |
| H-06 | Bölgeler bulanıklıktan sızmıyor (kenar, hızlı hareket) | Ortak bulanık çıktıyı izler, sızan kareyi zaman damgasıyla yazar | 0 sızan olay | Ortak |

Kaçırma (yakalanmayan kan) yanlış alarmdan daha kötü: Emirhan için kaçırma fiziksel acı, yanlış alarm yalnız hafif bulanıklık. Bu yüzden eşikler önce yakalamaya göre ayarlanır, sonra yanlış alarm düşürülür.

## 4. Veri

- **Vekil ve negatif:** kendi çekimimiz ya da serbest lisanslı stok video. Emirhan seçip indirebilir.
- **Pozitif:** Emirhan aramaz ve indirmez. Ortak toplar ya da kamuya açık, etiketli bir veri seti kullanılır (aday listesi doğrulanmadı; ortak ya da asistan bulup yalnız adını ve lisansını yazar). Film sahneleri yerine efekt/makyaj kliplerinden başlamak hem daha yumuşak hem lisansı daha açık.
- Etiketler (`data/etiket/*.json`) yalnız zaman aralığı ve kaba bölge içerir, görüntü içermez.

## 5. Emirhan hiç pozitif izlemeden ürünün son kontrolü

1. Ortak pozitif setin bulanık çıktısını izler ve kontrol listesini doldurur (kaçan, sızan, geç başlayan).
2. Hata sayıları Emirhan'a metin ve sayı olarak gelir.
3. Emirhan'ın gerçek kullanımı (bir film izlemek) H-01 ve H-06 ortağın kontrolünden geçtikten sonra ve kendi kararıyla olur. Plan bunu istemez.
