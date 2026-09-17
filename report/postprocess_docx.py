#!/usr/bin/env python3
"""Apply Chinese technical-report styles and page numbering to a Pandoc DOCX."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"

for prefix, uri in (("w", W), ("r", R), ("cp", CP), ("dc", DC)):
    ET.register_namespace(prefix, uri)


def q(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def child(parent: ET.Element, tag: str) -> ET.Element:
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    return node


def set_attr_element(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    node = child(parent, tag)
    for key, value in attrs.items():
        node.set(q(W, key), value)
    return node


def style_by_id(root: ET.Element, style_id: str) -> ET.Element | None:
    return root.find(f".//w:style[@w:styleId='{style_id}']", {"w": W})


def set_run_style(style: ET.Element, east_asia: str, latin: str, size: int, bold: bool = False) -> None:
    rpr = child(style, q(W, "rPr"))
    set_attr_element(
        rpr,
        q(W, "rFonts"),
        ascii=latin,
        hAnsi=latin,
        eastAsia=east_asia,
        cs=latin,
    )
    set_attr_element(rpr, q(W, "sz"), val=str(size))
    set_attr_element(rpr, q(W, "szCs"), val=str(size))
    bold_node = rpr.find(q(W, "b"))
    if bold and bold_node is None:
        ET.SubElement(rpr, q(W, "b"))
    if not bold and bold_node is not None:
        rpr.remove(bold_node)


def set_paragraph_style(
    style: ET.Element,
    *,
    before: int = 0,
    after: int = 120,
    line: int = 360,
    first_line: int | None = None,
    keep_next: bool = False,
    page_break: bool = False,
    align: str | None = None,
) -> None:
    ppr = child(style, q(W, "pPr"))
    set_attr_element(ppr, q(W, "spacing"), before=str(before), after=str(after), line=str(line), lineRule="auto")
    if first_line is not None:
        set_attr_element(ppr, q(W, "ind"), firstLine=str(first_line))
    if keep_next and ppr.find(q(W, "keepNext")) is None:
        ET.SubElement(ppr, q(W, "keepNext"))
    page_break_node = ppr.find(q(W, "pageBreakBefore"))
    if page_break and page_break_node is None:
        ET.SubElement(ppr, q(W, "pageBreakBefore"))
    elif not page_break and page_break_node is not None:
        ppr.remove(page_break_node)
    if align is not None:
        set_attr_element(ppr, q(W, "jc"), val=align)


def process_styles(data: bytes) -> bytes:
    root = ET.fromstring(data)
    for style_id in ("Normal", "BodyText", "FirstParagraph", "Compact"):
        style = style_by_id(root, style_id)
        if style is not None:
            set_run_style(style, "Noto Serif CJK SC", "Times New Roman", 22)
            set_paragraph_style(style, after=100, line=360, first_line=420)

    specifications = {
        "Title": (40, 0, 280, True, True, "center"),
        "Subtitle": (26, 0, 180, False, False, "center"),
        "Author": (22, 3000, 120, False, False, "center"),
        "Date": (22, 120, 120, False, False, "center"),
        "Heading1": (32, 240, 160, True, True, None),
        "Heading2": (28, 200, 120, True, False, None),
        "Heading3": (24, 160, 100, True, False, None),
        "Heading4": (22, 140, 80, True, False, None),
    }
    for style_id, (size, before, after, bold, page_break, align) in specifications.items():
        style = style_by_id(root, style_id)
        if style is not None:
            if style_id in {"Subtitle", "Author", "Date"}:
                based_on = child(style, q(W, "basedOn"))
                based_on.set(q(W, "val"), "Normal")
            set_run_style(style, "Noto Sans CJK SC", "Arial", size, bold=bold)
            set_paragraph_style(
                style,
                before=before,
                after=after,
                line=300,
                keep_next=True,
                page_break=page_break,
                align=align,
            )

    toc_heading = style_by_id(root, "TOCHeading")
    if toc_heading is not None:
        set_run_style(toc_heading, "Noto Sans CJK SC", "Arial", 32, bold=True)
        set_paragraph_style(
            toc_heading,
            before=240,
            after=160,
            line=300,
            keep_next=True,
            page_break=True,
            align="center",
        )

    for style_id in ("ImageCaption", "Caption"):
        style = style_by_id(root, style_id)
        if style is not None:
            set_run_style(style, "Noto Sans CJK SC", "Arial", 20)
            set_paragraph_style(style, after=100, line=300, align="center")

    for style_id in ("Table", "TableText", "VerbatimChar"):
        style = style_by_id(root, style_id)
        if style is not None:
            set_run_style(style, "Noto Serif CJK SC", "Times New Roman", 19)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def process_document(data: bytes, footer_rid: str) -> bytes:
    root = ET.fromstring(data)
    for text_node in root.findall(".//w:t", {"w": W}):
        if text_node.text == "Table of Contents":
            text_node.text = "目　录"
    for sect in root.findall(".//w:sectPr", {"w": W}):
        set_attr_element(sect, q(W, "pgSz"), w="11906", h="16838")
        set_attr_element(
            sect,
            q(W, "pgMar"),
            top="1417",
            right="1701",
            bottom="1417",
            left="1701",
            header="720",
            footer="720",
            gutter="0",
        )
        for ref in list(sect.findall(q(W, "footerReference"))):
            sect.remove(ref)
        footer_ref = ET.Element(q(W, "footerReference"))
        footer_ref.set(q(W, "type"), "default")
        footer_ref.set(q(R, "id"), footer_rid)
        sect.insert(0, footer_ref)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def process_settings(data: bytes) -> bytes:
    root = ET.fromstring(data)
    update = root.find(q(W, "updateFields"))
    if update is None:
        update = ET.SubElement(root, q(W, "updateFields"))
    update.set(q(W, "val"), "true")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_footer_relationship(data: bytes) -> tuple[bytes, str]:
    ET.register_namespace("", REL)
    root = ET.fromstring(data)
    used = {rel.get("Id", "") for rel in root}
    index = 1
    while f"rId{index}" in used:
        index += 1
    rid = f"rId{index}"
    rel = ET.SubElement(root, q(REL, "Relationship"))
    rel.set("Id", rid)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer")
    rel.set("Target", "footer_report.xml")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), rid


def process_content_types(data: bytes) -> bytes:
    ET.register_namespace("", CT)
    root = ET.fromstring(data)
    part_name = "/word/footer_report.xml"
    if not any(node.get("PartName") == part_name for node in root):
        override = ET.SubElement(root, q(CT, "Override"))
        override.set("PartName", part_name)
        override.set("ContentType", "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def footer_xml() -> bytes:
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W}" xmlns:r="{R}">
  <w:p>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Noto Sans CJK SC"/><w:sz w:val="18"/></w:rPr><w:t>第 </w:t></w:r>
    <w:fldSimple w:instr=" PAGE "><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>
    <w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Noto Sans CJK SC"/><w:sz w:val="18"/></w:rPr><w:t> 页</w:t></w:r>
  </w:p>
</w:ftr>'''
    return xml.encode("utf-8")


def process_core(data: bytes) -> bytes:
    root = ET.fromstring(data)
    title = root.find(q(DC, "title"))
    if title is not None:
        title.text = "强化学习增强蛋白氢键介导粘附/内聚潜力技术报告"
    subject = root.find(q(DC, "subject"))
    if subject is None:
        subject = ET.SubElement(root, q(DC, "subject"))
    subject.text = "模型实现、训练探索、阶段结果与后续技术决策"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def transform(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}

    rel_name = "word/_rels/document.xml.rels"
    payload[rel_name], footer_rid = add_footer_relationship(payload[rel_name])
    payload["[Content_Types].xml"] = process_content_types(payload["[Content_Types].xml"])
    payload["word/styles.xml"] = process_styles(payload["word/styles.xml"])
    payload["word/document.xml"] = process_document(payload["word/document.xml"], footer_rid)
    payload["word/settings.xml"] = process_settings(payload["word/settings.xml"])
    payload["word/footer_report.xml"] = footer_xml()
    if "docProps/core.xml" in payload:
        payload["docProps/core.xml"] = process_core(payload["docProps/core.xml"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="mprl_report_", suffix=".docx", delete=False, dir=destination.parent) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in payload.items():
                archive.writestr(name, data)
        shutil.move(str(temporary), str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    transform(args.source, args.destination or args.source)


if __name__ == "__main__":
    main()
