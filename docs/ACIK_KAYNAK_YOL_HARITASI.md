# Açık kaynak ve görünürlük yol haritası

Tarih: 13 Kasım 2026. Program bilgileri bu tarihte resmî sayfalardan okundu; başvurmadan önce sayfaya bir kez daha bak.

## Kısa özet

| Hedef | Gerçekçilik | Neden |
|---|---|---|
| 1. Açık kaynak + "ilk düşünen" avantajı | **Yüksek**, ama avantaj fikirde değil, *kayıtta* | Fikir korunamaz; korunan şey tarihli, herkese açık, çalışan iş ve senin hikâyen. Bunu bugün başlatabilirsin. |
| 2. Network / yatırımcı + Instagram kitlesi | Kitle: **orta**. Yatırımcı: **düşük** (bu proje için) | Gelir modeli yok, yatırımcı aramaz. Ama kitle ve görünürlük seni (ve sonraki projelerini) yatırımcının önüne çıkarır. Hedefi "yatırım" değil "tanınırlık + bağlantı" koy. |
| 3. OpenAI / Anthropic açık kaynak destekleri | Şu an **düşük**, 3-6 ayda **orta** | İki program da yaygın kullanılan projelere bakıyor. Yeni bir repo eşikleri tutmaz; "ekosistem için önemli" istisnasıyla başvurulabilir ama kullanıcı/yıldız/katkıcı kanıtı lazım. |

Sıra önemli: **önce 1, o 2'yi besler, 2 de 3'ün kanıtını üretir.** 3'e bugün başvurmak boşa bir hak harcamak olur.

## Bu hafta yapılacaklar (14-20 Kasım)

1. **İsim seç** (D-008 açık). 3-5 aday yaz; her biri için GitHub, PyPI, Instagram kullanıcı adı ve `.com/.app` boş mu bak. Seçince repo, paket ve hesap adı aynı olsun.
2. **Instagram hesabını aç** (yeni isimle; isim yoksa bekleme, kullanıcı adını sonra değiştirebilirsin). Bio: tek cümle ne yaptığı + "açık kaynak".
3. **PR #1 ve #2'yi birleştir**, `main` güncel olsun. Repo herkese açık değilse aç (ağırlıklar ve kareler zaten repoda yok; CI bunu denetliyor).
4. **README'yi vitrine çevir**: en üste 10 saniyelik kansız önce/sonra GIF (vekil video, `tools/before_after.py --vertical`), tek cümle hikâye, "nasıl denerim" 3 komut, "durum: araştırma prototipi" uyarısı.
5. **İlk sürümü etiketle** (`v2.1.0` GitHub Release). Bu, "ilk ben yaptım" kaydının tarihli kanıtı.
6. **İlk 3 videoyu çek, ilkini paylaş** (aşağıdaki içerik planı, 1-3. video).
7. **Benzer projeleri 1 saat ara** (GitHub, arXiv, Product Hunt: "gore blur", "violence detection video filter"). Varsa README'de "farkımız" satırı yaz; dürüstlük güven kazandırır.

## Hedef 1: Açık kaynak + "ilk düşünen" avantajı

**Gerçekçilik:** Büyük bir platform isterse bunu aylar içinde yapar; senin kozun fikir değil, şunlar:
- Tarihli açık kayıt (commit geçmişi, release, karar defteri, ilk videolar).
- Kişisel hikâye (sinestezi): kimse kopyalayamaz, insanlar buna bağlanır.
- Çalışan, ölçülmüş model (yakalama ~0,98, yanlış alarm 0/99 foto testte) ve bunu nasıl *kanı görmeden* yaptığın yöntem. Bu yöntem başlı başına ilginç bir hikâye.
- Topluluk: ilk katkıcılar, ilk kullanıcılar. Büyük şirket bunu kopyalayamaz.

Bir şirket benzerini yaparsa bu kayıp değil kazanç: "bunu ilk ben açık kaynak yaptım" cümlesi o gün en güçlü içeriğin olur.

**Adımlar:**
1. Bu hafta: isim, public repo, release, README vitrini (yukarıda).
2. 2-4. hafta: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, 5-10 "good first issue" (ör. oynatıcıya kısayol, yeni dil çevirisi, Windows kurulum notu). İngilizce README (Türkçe belgeler kalabilir; dünya kitlesi için kısa İngilizce giriş şart).
3. 1-2. ay: kurulumu kolaylaştır (tek komut / indirilebilir paket), model ağırlığını Hugging Face'e koy (lisans notuyla; eğitim verisinin lisansını kontrol et).
4. 2-3. ay: tarayıcı eklentisi ya da yerel oynatıcının kolay sürümü. Kullanıcı sayısı burada artar.
5. Sürekli: Hacker News "Show HN", Reddit (r/MachineLearning, r/opensource, erişilebilirlik toplulukları), Product Hunt. Her birini *bir kez*, iyi bir demo hazırken kullan.

## Hedef 2: Network / yatırımcı + Instagram kitlesi

**Gerçekçilik:**
- Kitle: Hikâye güçlü ve görsel olarak gösterilebilir (önce/sonra). Haftada 3 paylaşımla 3 ayda birkaç bin takipçi makul; viral olursa çok daha fazlası. Garantisi yok.
- Yatırımcı: Bu proje non-profit, yatırımcı getirisi beklemez. Gerçekçi kazanç: **hibe ve vakıflar** (erişilebilirlik, dijital sağlık), **yarışmalar**, **tanınırlık** ve sonraki ticari projen (isoshift, radar) için hazır bir kitle. Yatırımcı "bu kişi bir şeyi bitirip kitle toplayabiliyor" diye bakar; bu proje o kanıtı üretir.

**Instagram içerik planı (ilk 2 hafta, 6 video):**
1. "Kanlı bir sahne bana fiziksel acı veriyor" (yüz yok: ekran kaydı, el, masa; senin sesin). Sinesteziyi 30 saniyede anlat.
2. "Bu yüzden kendi yapay zekâmı eğitiyorum": kansız önce/sonra (kırmızı balon / ketçap vekil videosu).
3. "Kanı hiç görmeden bir modeli nasıl eğittim?" (bulanık küçük resimler, sayılar, terminal).
4. "Modelim kırmızı her şeyi kan sandı" (FA-04 hikâyesi; hatalar en çok izlenen içerik olur).
5. PC + Mac kurulumu, gece eğitimi, ısınan Mac (eğlenceli, insani).
6. "Açık kaynak yapıyorum, neden?" + repo bağlantısı.

**İçerik kuralları (kesin):**
- Kanlı ya da bulanık-kanlı gerçek görüntü **asla** yok; yalnız vekil ve negatif videolar (`tools/before_after.py` pozitif klasörü zaten reddediyor). Bu hem senin sağlığın hem platform kuralları için.
- Dizi/film sahnesi yok (telif + kan). Kendi çektiğin vekil sahneler kullan (kırmızı boya, ketçap, oyuncak; çekimi başkası yapsın, sen sonucu bulanık görürsün).
- Her videoda tek mesaj, ilk 2 saniyede kanca, altyazı.

**Çekim ve kurgu:**
- iPhone 15: 4K 30 fps, dikey; ses için kabloluyla bile olsa yaka mikrofonu en büyük fark. Yüz yoksa: el + ekran + B-roll.
- Claude ile kurgu bugün mümkün, sınırları belli: Mac'teki bir oturumda Claude `ffmpeg` ile kesme, 9:16 kırpma, ses normalleştirme, otomatik Türkçe altyazı (Whisper, yerel) ve önce/sonra montajını komutla yapabilir. Bu hesapta bağlı Adobe aracında hızlı kurgu (quick cut), konuşma sesini iyileştirme ve yeniden boyutlandırma da var. Önerilen akış: ham çekim → Claude kaba kesim + altyazı + dikey → sen Premiere'de renk, müzik, son rötuş.
- Kitle başka dillerde de büyüsün diye altyazıyı İngilizce de üret.

**Network adımları:**
1. 1. ay: LinkedIn ve X hesaplarında aynı hikâyeyi yazıyla paylaş (yatırımcı ve geliştiriciler orada).
2. 1-2. ay: erişilebilirlik ve nöroçeşitlilik topluluklarına, sinestezi derneklerine/araştırmacılarına yaz (gerçek kullanıcı ve hikâye ortağı).
3. 2-3. ay: Türkiye'de üniversite kuluçka merkezleri, teknoloji etkinlikleri; erişilebilirlik hibeleri ve yarışmalarını listele (TEKNOFEST Engelsiz Yaşam şu an isoshift'te; aynı yıl iki başvuru kuralını kontrol et).
4. Bağış: GitHub Sponsors / Open Collective, kullanıcı geldikten sonra aç (D-009).

## Hedef 3: OpenAI ve Anthropic açık kaynak destekleri

13 Kasım 2026'da resmî sayfalardan okunan bilgiler:

| | OpenAI **Codex for Open Source** | Anthropic **Claude for Open Source** |
|---|---|---|
| Ne veriyor | 6 ay ChatGPT Pro (Codex dahil); ayrıca "Codex Open Source Fund" ile API kredisi (miktar yazmıyor); koşullu Codex Security | 6 ay Claude Ultra aboneliği. **API kredisi yok.** Bitince ücretli planın varsa o plana döner, yoksa ücretsize düşer |
| Eşik | Sayısal eşik yok. "Yaygın kullanılan public proje"nin çekirdek bakımcıları; kullanım, ekosistem önemi, aktif bakım, rol ve kapasiteye bakılıyor | Biri yeter: 500+ bağımlı repo **veya** 100+ bağımlı paket **veya** ayda 200.000+ indirme; tanınmış vakıf/dil projesinde bakımcılık; son 12 ayda başkasının reposuna 100+ birleşmiş PR; bir reponda son 12 ayda 20+ dış katkıcı; OpenSSF kritiklik puanı 0,4+ |
| Küçük proje | "Ekosistemde önemli rol oynuyorsa" yine başvurulabilir, nedenini açıkla | "Ekosistemin sessizce bağlı olduğu bir şey bakıyorsan yine başvur" |
| Başvuru | [developers.openai.com/codex/community/codex-for-oss](https://developers.openai.com/codex/community/codex-for-oss) (form: openai.com/form/codex-for-oss) | [claude.com/contact-sales/claude-for-oss](https://claude.com/contact-sales/claude-for-oss) |
| Son tarih | Yazmıyor; OpenAI programı istediği an değiştirebilir/durdurabilir, karar kesin | Yazmıyor |

**Bu proje bugün uygun mu?** Dürüst cevap: **hayır, ikisinde de eşiklerin altındasın.** Yeni repo, bağımlı paket yok, dış katkıcı yok. İstisna maddesiyle başvurabilirsin ama kanıt olmadan büyük ihtimalle olumsuz döner.

**Ne zaman başvur:**
- Anthropic için en ulaşılabilir eşik: **bir reponda 12 ayda 20+ dış katkıcı.** "Good first issue" + topluluk + çeviri katkılarıyla 3-6 ayda ulaşılabilir. Ulaşınca başvur.
- OpenAI için: GitHub yıldızları, indirme ve kullanıcı hikâyesi biriktiğinde (ör. birkaç yüz yıldız, gerçek kullanıcı yorumları, medya haberi) istisna gerekçesiyle başvur. Codex'i PR inceleme ya da sürüm otomasyonunda gerçekten kullanıyorsan API fonu için bunu belirt.
- Gerekçe metni: erişilebilirlik (sinestezi, travma, hassas izleyiciler), non-profit, MIT, kanı görmeden veri hazırlama yöntemi, ölçülmüş sonuçlar.

**Dikkat:** Bu destekler kişiye özel ve devredilemez; diğer projelerinde (isoshift, radar, websel) kendi hesabınla kullanman programın amacına aykırı görünmüyor ama şartları başvuru sırasında oku. "Abonelik" ile "API kredisi" farklı: Anthropic'inki yalnız abonelik.

Diğer kapılar (doğrulanmadı, ayrıca bakılmalı): Claude'un Campus, Scientists, Startups programları ve Nonprofits sayfası; OpenAI ve diğer bulut sağlayıcıların araştırma/öğrenci kredileri; Hugging Face'in topluluk GPU destekleri.

## 3 aylık takvim

| Dönem | Açık kaynak | Görünürlük | Destek programları |
|---|---|---|---|
| 1. hafta | İsim, public repo, release, README vitrini | Hesap, ilk 3 video | Sayfaları yer imine al |
| 2-4. hafta | CONTRIBUTING, good first issue, İngilizce README | Haftada 3 video, LinkedIn/X yazısı | Kanıt topla (yıldız, katkıcı, kullanıcı yorumu) |
| 2. ay | Kolay kurulum, Hugging Face ağırlığı | Show HN / Reddit (bir kez, iyi demoyla) | OpenAI'a istisna gerekçesiyle başvur (kanıt varsa) |
| 3. ay | Tarayıcı eklentisi başlangıcı | Erişilebilirlik toplulukları, etkinlik, hibe | 20 dış katkıcıya yaklaştıysan Anthropic'e başvur |

## Kaynaklar

- OpenAI Codex for OSS: <https://developers.openai.com/codex/community/codex-for-oss>, şartlar: <https://developers.openai.com/codex/codex-for-oss-terms>
- Anthropic Claude for OSS: <https://claude.com/contact-sales/claude-for-oss>, şartlar: <https://www.anthropic.com/claude-for-oss-terms> (okunmadı)
