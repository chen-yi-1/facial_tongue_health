"""将 chapter5.md 转换为格式化的 .docx 文件"""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

INPUT_MD = "docs/superpowers/thesis/chapter5.md"
OUTPUT_DOCX = "docs/superpowers/thesis/第五章_系统设计与实现.docx"

doc = Document()

# 设置默认字体
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# 设置页边距
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

def add_heading_styled(text, level):
    """添加标题，使用黑体"""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 2:
            run.font.size = Pt(16)
        elif level == 3:
            run.font.size = Pt(14)
        elif level == 4:
            run.font.size = Pt(13)
    return h

def add_paragraph_styled(text, bold=False, font_size=12, first_line_indent=True):
    """添加正文段落，首行缩进2字符"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    if first_line_indent:
        pf.first_line_indent = Pt(24)  # 约2个字符

    # 处理文本中的粗体标记 **text**
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
        else:
            run = p.add_run(part)
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(font_size)
    return p

def add_plain_paragraph(text, bold=False):
    """添加段落（无首行缩进，用于代码块、标注等）"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    run = p.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(12)
    run.bold = bold
    return p

def add_code_paragraph(text):
    """添加代码段落（等宽字体、小一号）"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.2
    pf.left_indent = Cm(1)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(9)
    return p

def add_image_note(text):
    """添加图片插入标注（蓝色、居中）"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.5
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x0D, 0x94, 0x88)
    return p

def add_list_item(text, ordered=False, index=1):
    """添加列表项"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    pf.left_indent = Cm(1)
    pf.first_line_indent = Cm(-0.5)

    prefix = f"{index}. " if ordered else "• "

    # 处理粗体
    parts = re.split(r'(\*\*.*?\*\*)', text)
    first = True
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run((prefix if first else '') + part[2:-2])
            run.bold = True
        else:
            run = p.add_run((prefix if first else '') + part)
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(12)
        first = False
    return p

def parse_table_line(line):
    """解析 markdown 表格行"""
    line = line.strip().strip('|')
    return [cell.strip() for cell in line.split('|')]

def add_table(headers, rows):
    """添加 Word 表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'

    # 表头
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(10)
        # 表头背景色
        shading = cell._element.get_or_add_tcPr()
        shading_elem = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'F0FDFA',
            qn('w:val'): 'clear',
        })
        shading.append(shading_elem)

    # 数据行
    for r, row in enumerate(rows):
        for c, cell_text in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = ''
            run = cell.paragraphs[0].add_run(cell_text)
            run.font.name = '宋体'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(10)

    # 表后空行
    doc.add_paragraph()

# 读取并解析 markdown
with open(INPUT_MD, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
in_code_block = False
in_table = False
table_lines = []
in_quote = False
ordered_list_idx = 0
in_ordered_list = False

while i < len(lines):
    line = lines[i].rstrip()

    # 跳过使用说明中的引用块（第一段 > 开头的）
    if i == 0 and line.startswith('> '):
        in_quote = True

    if in_quote:
        if line.startswith('> ') or line == '>' or (line == '' and i < 10):
            i += 1
            if i >= len(lines) or (line == '' and not lines[i].strip().startswith('>')):
                in_quote = False
            continue
        in_quote = False

    # 代码块
    if line.startswith('```'):
        if in_code_block:
            in_code_block = False
            i += 1
            ordered_list_idx = 0
            in_ordered_list = False
            continue
        else:
            in_code_block = True
            i += 1
            ordered_list_idx = 0
            in_ordered_list = False
            continue

    if in_code_block:
        add_code_paragraph(line)
        i += 1
        continue

    # 表格（在非代码块中）
    if line.startswith('|') and line.endswith('|'):
        if not in_table:
            in_table = True
            table_lines = [line]
        else:
            table_lines.append(line)
        i += 1
        # 检查下一行是否还是表格
        if i < len(lines) and not (lines[i].strip().startswith('|') and lines[i].strip().endswith('|')):
            # 表格结束，解析并输出
            if len(table_lines) >= 2:
                headers = parse_table_line(table_lines[0])
                # 跳过分隔行
                rows = [parse_table_line(l) for l in table_lines[2:]]
                add_table(headers, rows)
            in_table = False
            table_lines = []
            ordered_list_idx = 0
            in_ordered_list = False
        continue

    # 空行
    if line == '':
        doc.add_paragraph()  # 空行
        ordered_list_idx = 0
        in_ordered_list = False
        i += 1
        continue

    # 分隔线
    if line.strip() == '---':
        doc.add_paragraph()
        i += 1
        ordered_list_idx = 0
        in_ordered_list = False
        continue

    # 标题
    if line.startswith('### '):
        add_heading_styled(line[4:], level=3)
        ordered_list_idx = 0
        in_ordered_list = False
        i += 1
        continue
    if line.startswith('#### '):
        add_heading_styled(line[5:], level=4)
        ordered_list_idx = 0
        in_ordered_list = False
        i += 1
        continue

    # 图片标注 [图5-X: ...]
    if line.startswith('[图5-') and ']' in line:
        add_image_note(line)
        i += 1
        continue

    # 表格标注 [表5-1: ...]（单独处理标注行，表格本体已在上面的表格逻辑中处理）
    if line.startswith('[表5-') and ']' in line:
        add_image_note(line)
        i += 1
        continue

    # 有序列表 (数字. 开头)
    ordered_match = re.match(r'^(\d+)\.\s+(.+)$', line)
    if ordered_match:
        idx, text = ordered_match.groups()
        add_list_item(text, ordered=True, index=int(idx))
        in_ordered_list = True
        ordered_list_idx = int(idx)
        i += 1
        continue

    # 无序列表 (- 或 * 开头)
    list_match = re.match(r'^[-*]\s+(.+)$', line)
    if list_match:
        add_list_item(list_match.group(1))
        in_ordered_list = False
        ordered_list_idx = 0
        i += 1
        continue

    # 加粗标题行（**xxx** 独占一行）
    bold_line_match = re.match(r'^\*\*(.+)\*\*$', line)
    if bold_line_match:
        add_plain_paragraph(bold_line_match.group(1), bold=True)
        i += 1
        continue

    # 引用块（论文一级标题）
    if line.startswith('## '):
        add_heading_styled(line[3:], level=2)
        i += 1
        continue

    # 普通段落
    add_paragraph_styled(line)
    i += 1

# 保存
doc.save(OUTPUT_DOCX)
print(f"已生成: {OUTPUT_DOCX}")
