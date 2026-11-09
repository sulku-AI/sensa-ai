#!/usr/bin/env bash
# Linux PC'de (RTX 2060) gece boyu eğitim. Kopma ya da çökme olursa last.ckpt'ten devam eder.
# Ayrıntı: docs/PC_EGITIM.md
#
#   bash tools/pc_egit.sh            30 tur
#   bash tools/pc_egit.sh 1          deneme: 1 tur, süreyi gösterir
#   KAPAT=1 bash tools/pc_egit.sh    bitince bilgisayarı kapatır
set -uo pipefail
cd "$(dirname "$0")/.."

TUR=${1:-30}
OUT=${OUT:-models/rfdetr_v2b_dizi}
DENEME=${DENEME:-5}
mkdir -p "$OUT"
LOG="$OUT/egitim.log"

bitti=0
for i in $(seq 1 "$DENEME"); do
  devam=()
  if [ -f "$OUT/last.ckpt" ]; then
    devam=(--resume "$OUT/last.ckpt")
  fi
  echo "== $(date '+%F %T') deneme $i/$DENEME, devam: ${devam[*]:-yok, sıfırdan}" | tee -a "$LOG"
  if python -m tools.train_rfdetr train --epochs "$TUR" --batch "${BATCH:-4}" --accum "${ACCUM:-4}" \
      --workers "${WORKERS:-6}" --device cuda --out "$OUT" "${devam[@]}" 2>&1 | tee -a "$LOG"; then
    bitti=1
    break
  fi
  echo "== $(date '+%F %T') eğitim durdu, 60 sn sonra yeniden" | tee -a "$LOG"
  sleep 60
done

if [ "$bitti" = 1 ]; then
  echo "== $(date '+%F %T') BİTTİ: $OUT" | tee -a "$LOG"
else
  echo "== $(date '+%F %T') $DENEME denemede bitmedi, $LOG dosyasının sonuna bak" | tee -a "$LOG"
fi
ls -la "$OUT" | tee -a "$LOG"

if [ "${KAPAT:-0}" = 1 ]; then
  systemctl poweroff
fi
