PHASE 5.4 — REPORT BUILDER (HTML + PDF)
==========================================

AGENT ROLE: Report Designer
DEPENDS ON: Phase 5.2 (coaching text), Phase 5.3 (charts)
DELIVERS TO: Phase 5.5 (final pipeline)
RUNS ON: Local M1
ESTIMATED TIME: 25 minutes (agent), 5 minutes (user — install wkhtmltopdf)

OBJECTIVE:
Build src/report/report_builder.py that assembles all coaching text,
charts, and raw data into a polished HTML report (with optional PDF export).

REPORT STRUCTURE (matches the mockup shown earlier):

```
Page 1:  Executive Summary (coaching text)
Page 2:  Radar Chart (6 scores) + score explanations
Page 3:  Timeline Heatmap (full speech at a glance)
Pages 4-10: Per-segment coaching feedback (with embedded mini-charts)
Page 11: Data Appendix (emotion arc, filler map, raw metrics table)
Page 12: Practice Plan (3 prioritized exercises)
```

IMPLEMENTATION:

1. Build HTML template at src/report/templates/report_template.html:
   - Clean, professional design (white background, good typography)
   - Responsive layout (looks good on screen and print)
   - Chart images embedded as base64 data URIs (self-contained HTML)
   - CSS in src/report/templates/styles.css

2. Build ReportBuilder class:
```python
class ReportBuilder:
    def __init__(self, template_dir="src/report/templates"):
        self.template_dir = template_dir

    def build_html(self, coaching_data, scores, charts, metadata) -> str:
        """Assemble the full HTML report."""
        # Load template
        # Inject: executive_summary, scores, chart base64 images,
        #         per-segment coaching, data tables, practice plan
        # Return complete HTML string

    def save_html(self, html, output_path):
        with open(output_path, "w") as f:
            f.write(html)

    def save_pdf(self, html, output_path):
        """Convert HTML to PDF using wkhtmltopdf or weasyprint."""
        try:
            import pdfkit
            pdfkit.from_string(html, output_path)
        except ImportError:
            # Fallback: weasyprint
            from weasyprint import HTML
            HTML(string=html).write_pdf(output_path)

    def build_and_save(self, coaching_data, scores, charts,
                        metadata, output_dir) -> dict:
        html = self.build_html(coaching_data, scores, charts, metadata)
        html_path = os.path.join(output_dir, "speech_report.html")
        pdf_path = os.path.join(output_dir, "speech_report.pdf")

        self.save_html(html, html_path)
        try:
            self.save_pdf(html, pdf_path)
        except Exception as e:
            print(f"PDF generation skipped: {e}")
            pdf_path = None

        return {"html": html_path, "pdf": pdf_path}
```

═══════════════════════════════════════════════════════════════
BRIDGE AGENT INTERVENTION
═══════════════════════════════════════════════════════════════

🔔 USER INPUT REQUIRED — Phase 5.4: Report Builder

For PDF generation, install one of these:
```bash
# Option A (recommended)
pip install weasyprint

# Option B
brew install wkhtmltopdf
pip install pdfkit
```

If neither installs cleanly, HTML reports still work perfectly.
PDF is optional.

COMPLETION: ✓ when HTML report generates with all sections populated.


═══════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════