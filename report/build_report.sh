#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MD="${REPORT_DIR}/technical_report.md"
OUTPUT_DOCX="${REPORT_DIR}/强化学习增强蛋白粘附与力学代理性能技术报告.docx"

if [[ -x /home/jianquanzhao/anaconda24/bin/python ]]; then
  PYTHON_BIN=/home/jianquanzhao/anaconda24/bin/python
else
  PYTHON_BIN=python3
fi

MPLCONFIGDIR=/tmp/mprl-report-mpl "${PYTHON_BIN}" "${REPORT_DIR}/generate_assets.py"

pandoc "${SOURCE_MD}" \
  --from=markdown+raw_attribute \
  --to=docx \
  --number-sections \
  --resource-path="${REPORT_DIR}" \
  --output="${OUTPUT_DOCX}"

"${PYTHON_BIN}" "${REPORT_DIR}/postprocess_docx.py" "${OUTPUT_DOCX}"
printf '%s\n' "${OUTPUT_DOCX}"
