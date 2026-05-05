from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

from .config import project_root, resolve_project_path


def ensure_sample_assets(root: Path | None = None) -> dict[str, str]:
    root = root or project_root()
    data_dir = root / "data"
    docs_dir = root / "sample_docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "sample_diet.db"
    ensure_sample_database(db_path)
    ensure_sample_docs(docs_dir)
    return {"database": str(db_path), "docs": str(docs_dir)}


def ensure_sample_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table if not exists foods (
                id integer primary key,
                name text not null unique,
                category text not null,
                calories integer not null,
                protein_g real not null,
                carbs_g real not null,
                fat_g real not null,
                fiber_g real not null
            );
            create table if not exists meals (
                id integer primary key,
                meal_date text not null,
                meal_type text not null,
                note text not null default ''
            );
            create table if not exists meal_items (
                id integer primary key,
                meal_id integer not null references meals(id),
                food_id integer not null references foods(id),
                servings real not null
            );
            create table if not exists water_logs (
                id integer primary key,
                log_date text not null unique,
                ounces integer not null
            );
            create table if not exists goals (
                id integer primary key,
                name text not null unique,
                target_value real not null,
                unit text not null,
                note text not null default ''
            );
            """
        )
        food_count = conn.execute("select count(*) from foods").fetchone()[0]
        if food_count:
            return
        conn.executemany(
            """
            insert into foods
                (id, name, category, calories, protein_g, carbs_g, fat_g, fiber_g)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "oatmeal", "grain", 150, 5.0, 27.0, 3.0, 4.0),
                (2, "plain yogurt", "dairy", 120, 12.0, 9.0, 3.0, 0.0),
                (3, "blueberries", "fruit", 84, 1.1, 21.0, 0.5, 3.6),
                (4, "scrambled eggs", "protein", 180, 12.0, 2.0, 14.0, 0.0),
                (5, "lentil soup", "legume", 230, 14.0, 36.0, 4.0, 11.0),
                (6, "green salad", "vegetable", 80, 3.0, 11.0, 3.0, 5.0),
                (7, "grilled tofu", "protein", 190, 20.0, 6.0, 11.0, 2.0),
                (8, "brown rice", "grain", 215, 5.0, 45.0, 2.0, 3.5),
                (9, "apple", "fruit", 95, 0.5, 25.0, 0.3, 4.4),
                (10, "trail mix", "snack", 260, 8.0, 22.0, 17.0, 3.0),
                (11, "carrots and hummus", "snack", 160, 5.0, 20.0, 7.0, 6.0),
            ],
        )
        conn.executemany(
            "insert into meals (id, meal_date, meal_type, note) values (?, ?, ?, ?)",
            [
                (1, "2026-04-27", "breakfast", "warm bowl before a walk"),
                (2, "2026-04-27", "lunch", "quick soup and salad"),
                (3, "2026-04-27", "dinner", "rice bowl"),
                (4, "2026-04-28", "breakfast", "eggs and fruit"),
                (5, "2026-04-28", "snack", "afternoon snack"),
                (6, "2026-04-29", "breakfast", "yogurt bowl"),
                (7, "2026-04-29", "lunch", "lentil leftovers"),
                (8, "2026-04-29", "snack", "crunchy snack"),
            ],
        )
        conn.executemany(
            "insert into meal_items (meal_id, food_id, servings) values (?, ?, ?)",
            [
                (1, 1, 1.0),
                (1, 3, 0.5),
                (2, 5, 1.0),
                (2, 6, 1.0),
                (3, 7, 1.0),
                (3, 8, 1.0),
                (4, 4, 1.0),
                (4, 9, 1.0),
                (5, 10, 1.0),
                (6, 2, 1.0),
                (6, 3, 1.0),
                (7, 5, 1.0),
                (8, 11, 1.0),
            ],
        )
        conn.executemany(
            "insert into water_logs (log_date, ounces) values (?, ?)",
            [
                ("2026-04-27", 68),
                ("2026-04-28", 76),
                ("2026-04-29", 70),
                ("2026-04-30", 82),
            ],
        )
        conn.executemany(
            "insert into goals (name, target_value, unit, note) values (?, ?, ?, ?)",
            [
                ("daily_water", 72, "ounces", "fictional demo hydration target"),
                ("daily_fiber", 25, "grams", "general sample goal, not medical advice"),
                ("protein_per_meal", 12, "grams", "demo target for comparing meals"),
            ],
        )


def ensure_sample_docs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    markdown = path / "meal_notes.md"
    if not markdown.exists():
        markdown.write_text(
            "# Meal Notes\n\nBreakfast examples include oatmeal, yogurt, fruit, and eggs.\n",
            encoding="utf-8",
        )
    html = path / "hydration_guide.html"
    if not html.exists():
        html.write_text(
            "<html><body><h1>Hydration Guide</h1><p>The sample goal is 72 ounces daily.</p></body></html>",
            encoding="utf-8",
        )
    pdf = path / "pantry_guide.pdf"
    if not pdf.exists():
        pdf.write_bytes(_simple_pdf_bytes())
    docx = path / "fiber_notes.docx"
    if not docx.exists():
        _write_simple_docx(docx)


def _simple_pdf_bytes() -> bytes:
    lines = [
        "Pantry Guide",
        "The demo pantry includes grains, legumes, fruit, vegetables, and protein foods.",
        "Use database queries for exact nutrition totals.",
    ]
    content_lines = ["BT", "/F1 12 Tf", "72 740 Td"]
    for index, line in enumerate(lines):
        if index:
            content_lines.append("0 -18 Td")
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def _write_simple_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Fiber Notes</w:t></w:r></w:p>
    <w:p><w:r><w:t>The sample diet log includes fiber values for foods such as oatmeal, lentil soup, salad, apples, and carrots with hummus.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Use this document with SQL queries when comparing tracked foods to the sample daily fiber goal.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def main() -> None:
    assets = ensure_sample_assets(project_root())
    print(f"Sample database: {assets['database']}")
    print(f"Sample docs: {assets['docs']}")


if __name__ == "__main__":
    main()
