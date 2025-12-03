#!/bin/bash

echo "=== Files larger than 500MB with locations ==="
find . -type f -size +500M -printf "%s\t%p\n" | \
  numfmt --field=1 --to=iec-i --suffix=B --padding=7 | \
  sort -hr | \
  head -30

echo ""
echo "=== Extensions of files > 500MB (with counts) ==="
find . -type f -size +500M -exec basename {} \; | \
  sed 's/.*\(\.[^.]*\)$/\1/' | \
  sort | uniq -c | sort -rn

echo ""
echo "=== Suggested .dockerignore entries ==="
find . -type f -size +500M -exec basename {} \; | \
  sed 's/.*\(\.[^.]*\)$/\1/' | \
  sort -u | \
  awk '{print "*" $0}'

echo ""
echo "=== Top directories by total size ==="
du -h --max-depth=1 . 2>/dev/null | sort -hr | head -20
