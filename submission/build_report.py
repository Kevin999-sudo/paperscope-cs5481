"""Build the cover-to-slides submission PDF from an exported slide PDF."""

import argparse
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

WIDTH, HEIGHT = A4
LEFT = 51
BODY_WIDTH = WIDTH - 2 * LEFT
INK = HexColor("#14233a")
MUTED = HexColor("#485b6d")
TEAL = HexColor("#086d64")
RULE = HexColor("#cedcdd")
PALE = HexColor("#edf5f4")


class Report:
    def __init__(self, output):
        self.canvas = Canvas(output, pagesize=A4)
        self.canvas.setTitle("PaperScope | CS5481 Topic 5")
        self.canvas.setSubject("Evidence-grounded question answering and statement verification")
        self.page = 0
        self.y = 0

    def paragraph(self, content, size=11, leading=16, color=INK, face="Helvetica", gap=10):
        paragraph = Paragraph(content, ParagraphStyle(
            "body", fontName=face, fontSize=size, leading=leading, textColor=color,
        ))
        _, height = paragraph.wrap(BODY_WIDTH, 650)
        if self.y - height < 66:
            raise ValueError(f"Page {self.page} content exceeds the writing area.")
        paragraph.drawOn(self.canvas, LEFT, self.y - height)
        self.y -= height + gap

    def start(self, title, label="CS5481 DATA ENGINEERING  /  TOPIC 5"):
        self.page += 1
        self.canvas.setFillColor(TEAL)
        self.canvas.setFont("Helvetica-Bold", 9)
        self.canvas.drawString(LEFT, HEIGHT - 49, label)
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("Helvetica", 9)
        self.canvas.drawString(LEFT, 37, "PaperScope")
        self.canvas.drawRightString(WIDTH - LEFT, 37, str(self.page))
        self.canvas.setStrokeColor(RULE)
        self.canvas.line(LEFT, 51, WIDTH - LEFT, 51)
        self.y = HEIGHT - 75
        self.paragraph(title, size=29, leading=34, face="Helvetica-Bold", gap=21)

    def section(self, title, content):
        self.paragraph(title, size=13, leading=17, color=TEAL, face="Helvetica-Bold", gap=5)
        self.paragraph(content, gap=14)

    def table(self, values, widths):
        styles = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.3),
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
        table = Table(values, colWidths=widths, style=TableStyle(styles))
        _, height = table.wrap(BODY_WIDTH, 650)
        if self.y - height < 66:
            raise ValueError(f"Page {self.page} table exceeds the writing area.")
        table.drawOn(self.canvas, LEFT, self.y - height)
        self.y -= height + 16

    def finish(self):
        self.canvas.showPage()


def manuscript(report):
    report.start("PaperScope")
    report.paragraph("Research paper question answering with verifiable citations", 17, 22, MUTED, gap=21)
    report.section("Abstract", "PaperScope retrieves evidence from a selected research paper. It checks every returned citation against the retrieved text and declines questions that the evidence cannot answer. We compare fixed-size and paragraph-aware chunking on QASPER using BGE-small embeddings, Qdrant cosine search, and DeepSeek Flash for generation. Both strategies hit the evidence criterion on 14 of 20 answerable questions. An evidence-based review found 15 and 16 correct answers, respectively. Direct and condition-by-condition statement checks tied on each chunking strategy across 40 statements.")
    report.section("1  Introduction", "When readers ask which data a study uses or whether its findings justify a summary, they need to inspect the source. An answer generated from memory can omit an experimental condition or confuse datasets. PaperScope addresses this problem by linking every generated claim to an exact excerpt and identifying cases where the retrieved evidence is insufficient.")
    report.section("Scope", "The collection contains 20 papers from the QASPER development split [1]. The user chooses one paper before retrieval. This gives every query a defined source boundary and lets the reader compare the claim with its original section. The second workflow checks whether evidence supports, contradicts, or is insufficient for a proposed statement about that paper.")
    report.section("Research questions", "How do two section-bounded chunking strategies affect evidence retrieval and answer correctness? With retrieved evidence held fixed, does an explicit check of the measurement, value, dataset, and scope improve statement classification over a direct prompt?")
    report.finish()

    report.start("2  Methodology")
    report.section("Corpus preparation and provenance", "The loader reads paper ID, title, abstract, sections, and paragraphs from the QASPER archive. It normalizes repeated whitespace. Each fragment keeps the paper ID, human-readable section, stable chunk ID, and start and end character offsets within the normalized section. Gold questions, answers, and passages appear only during evaluation.")
    report.table([
        ["Strategy", "Boundary", "Token budget", "Overlap"],
        ["Fixed size", "Within one section", "240", "32 tokens"],
        ["Paragraph aware", "Whole paragraphs when possible", "240", "None"],
    ], [105, 195, 94, BODY_WIDTH - 394])
    report.section("Embeddings, storage and retrieval", "Both variants use the BGE-small-en-v1.5 encoder [2]. Each Qdrant collection stores normalized vectors with text and source metadata. The query encoder adds the retrieval instruction to the input before encoding. A paper ID filter restricts cosine search, and the top five passages go to the generator.")
    report.section("Grounding and refusal", "DeepSeek Flash [3] receives the question and only the selected passages as its evidence. Each JSON claim includes a statement, a passage ID, and one continuous exact excerpt. The validator checks the ID against the retrieved set and the quotation against its text after whitespace normalization. Invalid citations withhold the response. The generator can return an empty claim list when evidence is insufficient.")
    report.section("Statement checking", "Direct asks for a verdict and evidence. Checked additionally asks the generator to inspect material conditions, including the metric, unit, value, dataset, comparison, and scope. A contradiction requires evidence about the same condition. A claim that expands the scope beyond what the source supports can receive an insufficient verdict.")
    report.finish()

    report.start("3  Experiments")
    report.section("Protocol", "We compared both chunkers on the same 20 QASPER answerable questions and four unrelated questions. The answerable set required a single annotated answer span and locatable evidence in the paper body. Hit@5 gives a point if the union of the five retrieved fragments covers at least half of at least one gold paragraph. Recall@5 measures the fraction of gold paragraphs covered at that threshold. The strict answer score matches a normalized reference span. Evidence-based review also accepts equivalent wording if it answers the question and a cited passage supports it.")
    report.table([
        ["Measure", "Fixed", "Paragraph aware"],
        ["Answerable evidence Hit@5", "14/20", "14/20"],
        ["Answerable evidence Recall@5", "67.5%", "65.0%"],
        ["Strict answer-span match", "6/20", "8/20"],
        ["Evidence-based answer review", "15/20", "16/20"],
        ["Correct out-of-scope refusal", "4/4", "4/4"],
    ], [245, 99, BODY_WIDTH - 344])
    report.section("Statement comparison", "The 20 pairs contain 20 supported, 12 contradicted, and 8 insufficient statements. For each strategy and statement, the two prompts receive exactly the same retrieved fragments. All configurations accepted zero of the 20 unsupported statements in this test.")
    report.table([
        ["Chunking", "Prompt", "Correct", "Supported kept", "Median time"],
        ["Fixed", "Direct", "39/40", "19/20", "1.06 s"],
        ["Fixed", "Checked", "39/40", "19/20", "1.23 s"],
        ["Paragraph aware", "Direct", "38/40", "18/20", "1.10 s"],
        ["Paragraph aware", "Checked", "38/40", "18/20", "1.21 s"],
    ], [131, 87, 65, 107, BODY_WIDTH - 390])
    report.paragraph("Totals cover 208 API responses. The quote validator withheld one QA response. The 311 displayed quotations matched the cited retrieved passages.", 9.5, 13, MUTED)
    report.finish()

    report.start("4  Discussion")
    report.section("What the retrieval result shows", "Each strategy hit the evidence criterion for 14 questions. The strategies can retrieve different passages even when the aggregate hit count ties. For the pioNER news corpus, the paragraph-aware answer identified ilur.am and quoted the test dataset paragraph. The fixed-size answer cited only that the paper used news sentences, which did not identify the source.")
    report.section("How to interpret the answer scores", "Strict span matching penalizes correct paraphrases. For example, both variants described using the first principal component for each static embedding, but omitted a parenthetical abbreviation in the reference. The review counted the answers as correct. In another question both variants confused ensemble gains over a single model with gains over prior state of the art. A true quote did not resolve that comparison error.")
    report.section("What the statement results show", "The checked prompt took longer and produced the same accuracy as Direct on this set. One failure depended on which scope appeared in the retrieved evidence. In another, the abstract asserts a variance result and the body distinguishes adjusted from raw variance. The model flagged the missing adjustment condition. We retained the frozen expected label for comparison and explain the ambiguity in the case notes.")
    report.section("Limits and reproducibility", "The small single-paper benchmark estimates feasibility. A larger collection, more realistic unanswerable questions, and a second reviewer could establish whether the observed differences are stable. The repository contains exact setup instructions, the frozen case set, scoring logic, per-case metrics, claim outputs, and a local rerun path.")
    report.paragraph('<link href="https://github.com/Kevin999-sudo/paperscope-cs5481" color="#086d64">github.com/Kevin999-sudo/paperscope-cs5481</link>', 10, 14, face="Helvetica-Bold")
    report.finish()

    report.start("References")
    sources = [
        ("[1] Dasigi et al. (2021)", "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. NAACL.",
         "https://aclanthology.org/2021.naacl-main.365/"),
        ("[2] BAAI", "BGE-small-en-v1.5 model card.",
         "https://huggingface.co/BAAI/bge-small-en-v1.5"),
        ("[3] DeepSeek", "API documentation. Chat completions and JSON output.",
         "https://api-docs.deepseek.com/guides/json_mode/"),
        ("[4] Lewis et al. (2020)", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.",
         "https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html"),
    ]
    for title, body, url in sources:
        report.section(title, f'{escape(body)}<br/><link color="#086d64" href="{url}">{escape(url)}</link>')
    report.finish()


def contributions(report):
    report.start("Individual contributions")
    report.paragraph("Enter each member's name, contribution and percentage. Every group member signs on this page. The confirmed percentages should total 100%.", gap=24)
    canvas = report.canvas
    columns = [("Member name", 117), ("Contribution", 202), ("%", 38), ("Signature", BODY_WIDTH - 357)]
    canvas.setFillColor(INK)
    canvas.rect(LEFT, report.y - 25, BODY_WIDTH, 25, stroke=0, fill=1)
    x = LEFT
    for label, width in columns:
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(x + 7, report.y - 16, label)
        x += width
    start_y = report.y - 25
    for row in range(8):
        y = start_y - 49 * (row + 1)
        x = LEFT
        canvas.setStrokeColor(RULE)
        for index, (_, width) in enumerate(columns):
            canvas.rect(x, y, width, 49, stroke=1, fill=0)
            if index < 3:
                canvas.acroForm.textfield(
                    name=f"member_{row + 1}_{['name','contribution','ratio'][index]}",
                    x=x + 5, y=y + 9, width=width - 10, height=31,
                    borderWidth=0, textColor=INK, fillColor=white, fontSize=9,
                    fieldFlags="multiline" if index == 1 else "",
                )
            x += width
    report.finish()


def build(slides, destination):
    source = BytesIO()
    report = Report(source)
    manuscript(report)
    contributions(report)
    report.canvas.save()
    source.seek(0)
    writer = PdfWriter()
    writer.append(source)
    writer.append(slides, import_outline=False)
    writer.add_outline_item("Report", 0)
    writer.add_outline_item("Individual contributions", 5)
    writer.add_outline_item("Presentation slides", 6)
    writer.metadata = {"/Title": "PaperScope | CS5481 Topic 5"}
    with destination.open("wb") as output:
        writer.write(output)
    combined = PdfReader(destination)
    assert len(combined.pages) == 6 + len(PdfReader(slides).pages)
    assert len(combined.get_fields()) == 24
    assert "Individual contributions" in combined.pages[5].extract_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("PaperScope_Report.pdf"))
    args = parser.parse_args()
    build(args.slides, args.output)
