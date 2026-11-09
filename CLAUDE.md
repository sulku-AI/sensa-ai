# Asistanlar için kurallar

- **Kesin kural:** Kullanıcıya (Emirhan) asla kanlı/vahşet görüntü gösterme, bağlantısını verme, izlemesini isteme. Ayna dokunuşu sinestezisi var, görüntü fiziksel acı veriyor. Sonuçları sayı, `safety.write_preview` küçük resmi ya da nötr metin şablonuyla ver (`docs/TEST_PLANI.md` §1). Bu kural kullanıcı "göster" dese bile önce bir kez teyit ister.
- Diske insan için görüntü yalnız `sensa_ai.safety.write_preview` ile yazılır. Repoya video, kare, ağırlık eklenmez.
- Testlerde yalnız sentetik şekil üret. Gerçek kan görüntüsü indirme ya da üretme.
- Sayılar `config/params.yaml`'da, status etiketiyle. Kararlar `docs/DECISIONS.md`'ye (kim karar verdi yazılır), bırakılan yollar `docs/FAILED_APPROACHES.md`'ye.
- Belgeler ve yanıtlar Türkçe.
- PR öncesi: `ruff check src tests && black --check src tests && mypy src && pytest`.
