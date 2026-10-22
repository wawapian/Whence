import pymupdf


def make_table_pdf(path, text=("course", "credit", "PE", "1.0")):
    """画 2×2 表格线 + ASCII 文字（默认字体渲染不了 CJK）。"""
    doc = pymupdf.open()
    page = doc.new_page()
    for x in (100, 200, 300):
        page.draw_line((x, 100), (x, 160))
    for y in (100, 130, 160):
        page.draw_line((100, y), (300, y))
    page.insert_text((110, 122), text[0]); page.insert_text((210, 122), text[1])
    page.insert_text((110, 152), text[2]); page.insert_text((210, 152), text[3])
    doc.save(str(path)); doc.close()
