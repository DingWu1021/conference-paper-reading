#!/bin/bash
# 把 conference-paper/<TIMESTAMP>/ 里的论文 md 同步到飞书 conference-paper-reader/<TIMESTAMP>/
# 用法: feishu_sync.sh <时间戳文件夹名，如 iclr2026>
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

PAPER_READING_REPO="/Users/zhengxianwu/Documents/paper_reading"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="${1:?需要时间戳文件夹名，如 iclr2026}"
LOCALDIR="$REPO/$TIMESTAMP"
[ -d "$LOCALDIR" ] || { echo "ERR: 本地无此文件夹 $LOCALDIR"; exit 1; }

source "$PAPER_READING_REPO/scripts/feishu_lib.sh"

LOG="$REPO/scripts/feishu_sync.log"
mkdir -p "$REPO/scripts"

echo "===== $(date '+%F %T') feishu_sync $TIMESTAMP =====" | tee -a "$LOG"

TK=$(feishu_token)
[ -z "$TK" ] && { echo "ERR no token" | tee -a "$LOG"; exit 1; }
echo "Token OK"

# 找或建 conference-paper-reader 根文件夹（新建时按 open_id 共享给用户，container 继承）
ROOT_CACHE="$REPO/scripts/.feishu_conf_root"
USER_OID="ou_5f7242101b602b04c6d5e0a0a69b5660"
ROOT=$(cat "$ROOT_CACHE" 2>/dev/null || true)
if [ -z "${ROOT:-}" ]; then
  echo "Creating conference-paper-reader folder..."
  ROOT=$(feishu_create_folder "$TK" "conference-paper-reader" "")
  case "$ROOT" in ERR*|"") echo "ERR create root: $ROOT" | tee -a "$LOG"; exit 1;; esac
  echo "$ROOT" > "$ROOT_CACHE"
  # 一次性共享给用户（edit+container，子内容自动继承）
  python3 -c "import json; print(json.dumps({'member_type':'openid','member_id':'$USER_OID','perm':'edit','perm_type':'container','type':'user'}))" > /tmp/_fs_member.json
  share_result=$(fscurl -X POST "$FS_BASE/drive/v1/permissions/$ROOT/members?type=folder" \
    -H "Authorization: Bearer $TK" -H "Content-Type: application/json" -d @/tmp/_fs_member.json \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('shared OK' if d.get('code')==0 else 'share WARN:'+str(d))")
  echo "folder share: $share_result" | tee -a "$LOG"
fi
echo "Root folder: $ROOT" | tee -a "$LOG"

# 建时间子文件夹
echo "Creating $TIMESTAMP subfolder..."
SUBF=$(feishu_create_folder "$TK" "$TIMESTAMP" "$ROOT")
case "$SUBF" in ERR*|"") echo "ERR create subfolder: $SUBF" | tee -a "$LOG"; exit 1;; esac
echo "Subfolder: $SUBF" | tee -a "$LOG"

MAP="$REPO/scripts/_work/feishu_${TIMESTAMP}.map"; mkdir -p "$(dirname "$MAP")"; : > "$MAP"

# 逐篇导入（00-summary 排最后，先按名排序）
for md in $(ls "$LOCALDIR"/*.md | sort); do
  base=$(basename "$md")
  case "$base" in
    00-summary-1.md|"00-今日论文总结-1.md") name="★今日论文总结（一）" ;;
    00-summary-2.md|"00-今日论文总结-2.md") name="★今日论文总结（二）" ;;
    00-summary-3.md|"00-今日论文总结-3.md") name="★今日论文总结（三）" ;;
    *) name=$(python3 "$PAPER_READING_REPO/scripts/_derive_docname.py" "$md") ;;
  esac

  # 转飞书公式语法（行内 $...$ → $$...$$，块级 $$...$$ 不变）
  fsmd="/tmp/_feishu_${TIMESTAMP}_${base}"
  python3 "$PAPER_READING_REPO/scripts/md_to_feishu.py" "$md" > "$fsmd"

  echo "Importing '$name' ..." | tee -a "$LOG"
  # grep for URL: feishu_import_md may emit polling status lines to stdout
  url=$(feishu_import_md "$TK" "$fsmd" "$name" "$SUBF" | grep "^https://" | tail -1)
  if [ -n "$url" ]; then
    tok=${url##*/docx/}
    feishu_set_public "$TK" "$tok" >/dev/null
    printf '%s\t%s\n' "$name" "$url" >> "$MAP"
    echo "OK  $name -> $url" | tee -a "$LOG"
    # 上传本地图片到飞书 CDN 替换外链占位（需本地 figs/ 目录存在）
    feishu_fix_images_for_md "$TK" "$tok" "$md" 2>/dev/null | tee -a "$LOG"
  else
    echo "FAIL $base" | tee -a "$LOG"
  fi
done

# 生成索引文档
python3 - "$TIMESTAMP" "$MAP" > /tmp/_fs_index.md <<'PY'
import sys
folder, mapf = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(mapf) if l.strip()]
out = [f"# {folder} 索引\n", f"> 共 {len(rows)} 篇（飞书云文档，公式/图已渲染，点击进入）。\n"]
for i, (name, url) in enumerate(rows, 1):
    out.append(f"{i}. [{name}]({url})")
open("/tmp/_fs_index.md", "w").write("\n".join(out))
PY
IDXURL=$(feishu_import_md "$TK" "/tmp/_fs_index.md" "★ ${TIMESTAMP} 索引" "$SUBF" | grep "^https://" | tail -1)
if [ -n "$IDXURL" ]; then
  tok=${IDXURL##*/docx/}; feishu_set_public "$TK" "$tok" >/dev/null
  echo "INDEX $IDXURL" | tee -a "$LOG"
else
  echo "FAIL index" | tee -a "$LOG"
fi

echo "===== $(date '+%F %T') feishu_sync $TIMESTAMP done =====" | tee -a "$LOG"
