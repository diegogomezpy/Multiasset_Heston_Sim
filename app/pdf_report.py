"""
app/pdf_report.py
-----------------
Institutional-grade PDF report generator for the Structured Note Simulator.

Visual language modelled on sell-side QIS / wealth-management publications:
  - Cover page: brand wordmark, series title, accent subtitle, executive
    summary bullets, right-hand info sidebar panel.
  - Inner pages: small-caps eyebrow header + hairline, accent section
    headings, centred "Figure N" captions with source lines, thin-rule
    key-value tables, filled-header data tables, callout boxes.
  - Footer: italic compliance line + "Page X of Y".

Public API (unchanged)
----------------------
generate_pdf_report(terms, results, asset_names, figures, lang,
                    bt_summary, bt_figures, live_data, live_figure) -> bytes
"""

from __future__ import annotations

import io
import datetime
import numpy as np
from fpdf import FPDF

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
_BRAND        = (20,  82,  20)    # dark green — wordmark, table headers
_ACCENT       = (26, 107,  26)    # section titles, figure captions
_TEXT         = (40,  40,  40)
_TEXT_SOFT    = (105, 105, 105)
_HAIRLINE     = (200, 200, 200)
_RULE_LIGHT   = (225, 225, 225)
_PANEL        = (238, 244, 238)   # sidebar / callout fill
_ROW_ALT      = (247, 249, 247)
_WHITE        = (255, 255, 255)

# ──────────────────────────────────────────────────────────────────────────────
# Translations
# ──────────────────────────────────────────────────────────────────────────────
_LABELS: dict[str, dict[str, str]] = {
    "series_title":          {"en": "Structured Note Analytics",         "es": "Análisis de Nota Estructurada"},
    "report_eyebrow":        {"en": "STRUCTURED NOTE ANALYTICS",         "es": "ANÁLISIS DE NOTA ESTRUCTURADA"},
    "generated":             {"en": "Publication date",                  "es": "Fecha de publicación"},
    "underlyings":           {"en": "UNDERLYINGS",                       "es": "SUBYACENTES"},
    "key_terms":             {"en": "KEY TERMS",                         "es": "TÉRMINOS CLAVE"},
    "exec_summary":          {"en": "Executive Summary",                 "es": "Resumen Ejecutivo"},
    "note_terms":            {"en": "Note Terms",                        "es": "Términos de la Nota"},
    "obs_schedule":          {"en": "Observation Schedule",              "es": "Calendario de Observaciones"},
    "sim_summary":           {"en": "Monte Carlo Simulation",            "es": "Simulación Monte Carlo"},
    "model_box_title":       {"en": "Model & Methodology",               "es": "Modelo y Metodología"},
    "model_box_body":        {
        "en": "Multi-asset Heston stochastic-volatility model calibrated on "
              "dividend-adjusted (total-return) closes; drift, variance and "
              "correlation blocks estimated by method of moments. Simulation "
              "runs on a real trading-day grid; predictable dividend ex-date "
              "drops are applied as deterministic jumps so barrier levels are "
              "observed on price (not total-return) paths. Antithetic "
              "variates; Student-t copula for joint tail dependence. The "
              "payoff engine is shared between simulation and historical "
              "backtest.",
        "es": "Modelo Heston multi-activo de volatilidad estocástica calibrado "
              "sobre cierres ajustados por dividendos (retorno total); deriva, "
              "varianza y correlaciones estimadas por método de momentos. La "
              "simulación corre sobre un calendario real de días hábiles; las "
              "caídas previsibles por ex-dividendo se aplican como saltos "
              "deterministas, de modo que las barreras se observan sobre "
              "precios (no retorno total). Variables antitéticas; cópula "
              "t-Student para dependencia de colas. El motor de pagos es "
              "compartido entre simulación y backtest histórico.",
    },
    "calibration":           {"en": "Model Calibration",                 "es": "Calibración del Modelo"},
    "backtest":              {"en": "Historical Backtest",               "es": "Backtest Histórico"},
    "live":                  {"en": "Current Performance",               "es": "Rendimiento Actual"},
    "disclaimer_title":      {"en": "Important Information",             "es": "Información Importante"},
    # Term labels
    "maturity":              {"en": "Maturity",                          "es": "Vencimiento"},
    "freq":                  {"en": "Payment frequency",                 "es": "Frecuencia de pago"},
    "coupon_pa":             {"en": "Coupon p.a.",                       "es": "Cupón anual"},
    "coupon_barrier":        {"en": "Coupon barrier",                    "es": "Barrera de cupón"},
    "autocall_barrier":      {"en": "Autocall barrier",                  "es": "Barrera de autocall"},
    "autocall_start":        {"en": "First autocall observation",        "es": "Primera observación autocall"},
    "ki_barrier":            {"en": "Knock-in barrier (European)",       "es": "Barrera knock-in (europea)"},
    "memory":                {"en": "Memory coupon",                     "es": "Cupón con memoria"},
    "coupon_basket":         {"en": "Coupon basket",                     "es": "Cesta de cupón"},
    "autocall_basket":       {"en": "Autocall basket",                   "es": "Cesta de autocall"},
    "final_basket":          {"en": "Final redemption basket",           "es": "Cesta de redención final"},
    "rescue_barrier":        {"en": "Final redemption barrier",          "es": "Barrera de redención final"},
    "ac_step_down":          {"en": "Autocall step-down / period",       "es": "Reducción de barrera / período"},
    "ac_floor":              {"en": "Autocall barrier floor",            "es": "Suelo de barrera autocall"},
    "premium_at_call":       {"en": "Premium paid only at autocall",     "es": "Prima pagada solo al autocall"},
    "issue_date":            {"en": "Issue date",                        "es": "Fecha de emisión"},
    "issuer":                {"en": "Issuer",                            "es": "Emisor"},
    # Simulation labels
    "expected_irr":          {"en": "Expected IRR p.a.",                 "es": "TIR esperada anual"},
    "expected_total_return": {"en": "Expected total return",             "es": "Retorno total esperado"},
    "total_return_short":    {"en": "Total return",                      "es": "Retorno total"},
    "in_this_report":        {"en": "In this report",                    "es": "En este informe"},
    "expected_coupon":       {"en": "Expected coupon income",            "es": "Cupón total esperado"},
    "prob_autocall":         {"en": "P(autocall)",                       "es": "P(autocall)"},
    "prob_knock_in":         {"en": "P(capital loss)",                   "es": "P(pérdida de capital)"},
    "n_paths":               {"en": "Simulated paths",                   "es": "Caminos simulados"},
    # Autocall table
    "autocall_by_period":    {"en": "Autocall Probability by Period",    "es": "Probabilidad de Autocall por Período"},
    "period":                {"en": "Period",                            "es": "Período"},
    "time_y":                {"en": "Time (yrs)",                        "es": "Tiempo (años)"},
    "p_autocall":            {"en": "P(autocall)",                       "es": "P(autocall)"},
    "ac_level":              {"en": "Barrier",                           "es": "Barrera"},
    "eligible":              {"en": "Eligible",                          "es": "Elegible"},
    "yes":                   {"en": "Yes",                               "es": "Sí"},
    "no":                    {"en": "No",                                "es": "No"},
    # Figures
    "fig_irr":               {"en": "Distribution of simple annualised IRR across simulated paths",
                              "es": "Distribución de TIR anual simple en los caminos simulados"},
    "fig_wof":               {"en": "Worst-of basket performance fan with barrier levels",
                              "es": "Abanico de la cesta worst-of con niveles de barrera"},
    "fig_corr":              {"en": "Calibrated return correlation matrix",
                              "es": "Matriz de correlaciones de retorno calibrada"},
    "fig_bt_outcome":        {"en": "Distribution of historical outcomes by issue date",
                              "es": "Distribución de resultados históricos por fecha de emisión"},
    "fig_bt_irr":            {"en": "Realised simple annualised IRR by historical issue date",
                              "es": "TIR anual simple realizada por fecha de emisión histórica"},
    "fig_live":              {"en": "Underlying performance since issue date with observation outcomes",
                              "es": "Rendimiento de los subyacentes desde emisión con resultados de observación"},
    "src_mc":                {"en": "Source: Heston Monte Carlo simulation",
                              "es": "Fuente: simulación Monte Carlo Heston"},
    "src_hist":              {"en": "Source: Yahoo Finance daily closing prices",
                              "es": "Fuente: precios de cierre diarios de Yahoo Finance"},
    # Heston table
    "asset":                 {"en": "Asset",                             "es": "Activo"},
    "feller":                {"en": "Feller",                            "es": "Feller"},
    # Backtest section
    "bt_n_issues":           {"en": "Issue dates tested",                "es": "Fechas de emisión probadas"},
    "bt_mean_irr":           {"en": "Mean IRR p.a.",                     "es": "TIR media anual"},
    "bt_median_irr":         {"en": "Median IRR p.a.",                   "es": "TIR mediana anual"},
    "bt_knock_in_pct":       {"en": "Knock-in rate",                     "es": "Tasa de knock-in"},
    "bt_autocalled_pct":     {"en": "Autocall rate",                     "es": "Tasa de autocall"},
    # Live performance section
    "live_wof_today":        {"en": "Worst-of today",                    "es": "Worst-of hoy"},
    "live_worst_asset":      {"en": "Worst asset",                       "es": "Peor activo"},
    "live_irr_to_date":      {"en": "Coupon IRR to date (ann.)",         "es": "TIR de cupones a fecha (anual)"},
    "live_elapsed":          {"en": "Elapsed (years)",                   "es": "Transcurrido (años)"},
    "live_asset_perf":       {"en": "Current Asset Performance",         "es": "Rendimiento Actual por Activo"},
    "live_obs_history":      {"en": "Observation History",               "es": "Historial de Observaciones"},
    "performance":           {"en": "Performance",                       "es": "Rendimiento"},
    # Compliance
    "footer_line": {
        "en": "For information only. Output of an automated quantitative simulation — not investment advice, an offer, or a solicitation.",
        "es": "Solo a título informativo. Resultado de una simulación cuantitativa automatizada — no es asesoramiento ni oferta de inversión.",
    },
    "cover_topline": {
        "en": "This document was generated by an automated quantitative simulation tool and has not been reviewed by any research department. "
              "Refer to the Important Information section at the end of this document.",
        "es": "Este documento fue generado por una herramienta automatizada de simulación cuantitativa y no ha sido revisado por ningún departamento de análisis. "
              "Consulte la sección de Información Importante al final del documento.",
    },
    "disclaimer_body": {
        "en": "This report is the output of an automated quantitative simulation tool and is provided for information purposes only. "
              "It does not constitute investment research, investment advice, a recommendation, an offer to sell or a solicitation of an "
              "offer to buy any security or financial instrument.\n\n"
              "Simulated performance is based on a Heston stochastic-volatility model calibrated to historical market data. Model "
              "parameters, correlations and dividend forecasts are estimates and may differ materially from realised market behaviour. "
              "Simulated and backtested results are hypothetical, do not reflect actual trading, and are not a reliable indicator of "
              "future results. Historical backtest windows overlap and the resulting statistics are autocorrelated.\n\n"
              "Structured notes are complex instruments that may result in the loss of some or all of the capital invested. Payments "
              "depend on the creditworthiness of the issuer. Barrier observation levels, dates and payoff mechanics are simplified "
              "representations of the relevant term sheet; in case of any discrepancy the official offering documentation prevails.\n\n"
              "Market data sourced from Yahoo Finance and may be delayed, incomplete or inaccurate. No representation or warranty, "
              "express or implied, is made as to the accuracy or completeness of the information contained herein.",
        "es": "Este informe es el resultado de una herramienta automatizada de simulación cuantitativa y se proporciona únicamente con "
              "fines informativos. No constituye análisis financiero, asesoramiento de inversión, una recomendación, una oferta de venta "
              "ni una solicitud de compra de ningún valor o instrumento financiero.\n\n"
              "El rendimiento simulado se basa en un modelo de volatilidad estocástica de Heston calibrado con datos históricos de "
              "mercado. Los parámetros del modelo, las correlaciones y las previsiones de dividendos son estimaciones y pueden diferir "
              "materialmente del comportamiento realizado del mercado. Los resultados simulados y de backtest son hipotéticos, no "
              "reflejan operaciones reales y no son un indicador fiable de resultados futuros. Las ventanas del backtest histórico se "
              "solapan y las estadísticas resultantes están autocorrelacionadas.\n\n"
              "Las notas estructuradas son instrumentos complejos que pueden conllevar la pérdida parcial o total del capital invertido. "
              "Los pagos dependen de la solvencia del emisor. Los niveles de barrera, fechas y mecánica de pagos son representaciones "
              "simplificadas del term sheet correspondiente; en caso de discrepancia prevalece la documentación oficial de la emisión.\n\n"
              "Datos de mercado procedentes de Yahoo Finance, que pueden estar retrasados, incompletos o ser inexactos. No se ofrece "
              "ninguna garantía, expresa o implícita, sobre la exactitud o integridad de la información aquí contenida.",
    },
}


def _t(key: str, lang: str) -> str:
    return _LABELS.get(key, {}).get(lang, _LABELS.get(key, {}).get("en", key))


# ──────────────────────────────────────────────────────────────────────────────
# Text sanitisation — core Helvetica is Latin-1 only
# ──────────────────────────────────────────────────────────────────────────────
_TRANSLITERATE = {
    "—": "-", "–": "-", "−": "-",
    "·": "-", "•": "-",
    "→": "->", "←": "<-",
    "≥": ">=", "≤": "<=",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "×": "x", "÷": "/",
    "€": "EUR", "£": "GBP",
    "κ": "kappa", "θ": "theta", "ξ": "xi", "ρ": "rho", "σ": "sigma", "μ": "mu", "ν": "nu",
    "₀": "0", "√": "sqrt ",
    "✅": "OK", "⚠️": "!", "❌": "x", "🚀": ">>", "⏳": "...",
}


def _safe(text) -> str:
    s = str(text)
    for bad, good in _TRANSLITERATE.items():
        s = s.replace(bad, good)
    # Drop any remaining non-Latin-1 characters (keeps accented Spanish letters).
    return s.encode("latin-1", "ignore").decode("latin-1")


# ──────────────────────────────────────────────────────────────────────────────
# FPDF subclass — institutional page chrome and building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _NotePDF(FPDF):
    """A4 portrait document with QIS-publication styling."""

    def __init__(self, lang: str = "en", issuer: str = "", doc_ref: str = ""):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.lang     = lang
        self.issuer   = issuer
        self.doc_ref  = doc_ref
        self._is_cover = False
        self._fig_no   = 0
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=26)
        self.alias_nb_pages()

    # ------------------------------------------------------------------
    # Sanitise every string written to the document
    # ------------------------------------------------------------------
    def cell(self, *args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], str):
            args = (args[0], args[1], _safe(args[2]), *args[3:])
        for k in ("text", "txt"):
            if k in kwargs and isinstance(kwargs[k], str):
                kwargs[k] = _safe(kwargs[k])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], str):
            args = (args[0], args[1], _safe(args[2]), *args[3:])
        for k in ("text", "txt"):
            if k in kwargs and isinstance(kwargs[k], str):
                kwargs[k] = _safe(kwargs[k])
        return super().multi_cell(*args, **kwargs)

    # ------------------------------------------------------------------
    # Page chrome
    # ------------------------------------------------------------------
    def header(self):
        if self._is_cover:
            return
        # Small-caps eyebrow left, issuer wordmark right, hairline below
        self.set_xy(self.l_margin, 10)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*_TEXT_SOFT)
        try:
            self.set_char_spacing(0.8)
        except Exception:
            pass
        self.cell(120, 5, _t("report_eyebrow", self.lang))
        try:
            self.set_char_spacing(0)
        except Exception:
            pass
        if self.issuer:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*_BRAND)
            self.set_xy(self.w - self.r_margin - 80, 8.5)
            self.cell(80, 7, self.issuer.upper(), align="R")
        self.set_draw_color(*_HAIRLINE)
        self.set_line_width(0.3)
        self.line(self.l_margin, 17, self.w - self.r_margin, 17)
        self.set_text_color(*_TEXT)
        self.set_y(22)

    def footer(self):
        self.set_draw_color(*_HAIRLINE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.h - 20, self.w - self.r_margin, self.h - 20)
        self.set_y(-18)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(*_TEXT_SOFT)
        self.multi_cell(0, 3, _t("footer_line", self.lang), align="L")
        self.set_y(-11)
        self.set_font("Helvetica", "", 7)
        left = self.doc_ref or _t("series_title", self.lang)
        self.cell(0, 5, left, align="L")
        self.set_y(-11)
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")
        self.set_text_color(*_TEXT)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def section_title(self, text: str):
        """Accent-colour section heading (Barclays-style 'Introduction')."""
        if self.get_y() > self.h - 60:
            self.add_page()
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_ACCENT)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_TEXT)
        self.ln(1.5)

    def subsection(self, text: str):
        # Keep the heading with its content — never orphan it at a page foot
        if self.get_y() > self.h - 60:
            self.add_page()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_TEXT)
        self.cell(0, 6, text.upper(), new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def body(self, text: str, h: float = 4.6):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_TEXT)
        self.multi_cell(0, h, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_TEXT)
        x0 = self.get_x()
        self.cell(5, 4.8, chr(149))      # latin-1 bullet
        self.multi_cell(self.w - self.r_margin - x0 - 5, 4.8, text)
        self.ln(1.2)

    def kv_table(self, rows: list[tuple[str, str]], col_w: tuple[float, float] = (78, 100)):
        """Thin-rule key/value table: bold label, hairline under each row."""
        self.set_text_color(*_TEXT)
        for k, v in rows:
            y0 = self.get_y()
            if y0 > self.h - 32:
                self.add_page()
                y0 = self.get_y()
            self.set_font("Helvetica", "B", 8.5)
            self.cell(col_w[0], 6.4, k)
            self.set_font("Helvetica", "", 8.5)
            self.cell(col_w[1], 6.4, v, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*_RULE_LIGHT)
            self.set_line_width(0.2)
            self.line(self.l_margin, self.get_y(), self.l_margin + col_w[0] + col_w[1], self.get_y())
        self.ln(3)

    def data_table(self, headers: list[str], rows: list[list[str]],
                   col_widths: list[float] | None = None,
                   aligns: list[str] | None = None):
        n = len(headers)
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / n] * n
        if aligns is None:
            aligns = ["L"] + ["R"] * (n - 1)

        def _header_row():
            self.set_fill_color(*_BRAND)
            self.set_text_color(*_WHITE)
            self.set_font("Helvetica", "B", 7.5)
            for h, w, a in zip(headers, col_widths, aligns):
                self.cell(w, 6.5, f" {h} ", border=0, fill=True, align=a)
            self.ln()
            self.set_text_color(*_TEXT)
            self.set_font("Helvetica", "", 7.5)

        # Don't start a table that can only fit a couple of orphan rows
        if self.get_y() > self.h - 55:
            self.add_page()
        _header_row()

        for i, row in enumerate(rows):
            if self.get_y() > self.h - 30:
                self.add_page()
                _header_row()          # repeat header after a page break
            self.set_fill_color(*(_ROW_ALT if i % 2 == 0 else _WHITE))
            for cell_val, w, a in zip(row, col_widths, aligns):
                self.cell(w, 5.8, f" {cell_val} ", border=0, fill=True, align=a)
            self.ln()
        # closing hairline
        self.set_draw_color(*_HAIRLINE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.l_margin + sum(col_widths), self.get_y())
        self.ln(4)

    def metric_band(self, metrics: list[tuple[str, str]]):
        """Row of key figures: small-caps grey label over a large value."""
        n = len(metrics)
        usable = self.w - self.l_margin - self.r_margin
        w = usable / n
        y0 = self.get_y()
        self.set_draw_color(*_BRAND)
        self.set_line_width(0.5)
        self.line(self.l_margin, y0, self.w - self.r_margin, y0)
        self.ln(2)
        x = self.l_margin
        for label, value in metrics:
            # Label on ONE line — shrink the font until it fits its column
            lbl = _safe(label.upper())
            size = 6.7
            self.set_font("Helvetica", "B", size)
            while self.get_string_width(lbl) > (w - 3) and size > 4.5:
                size -= 0.2
                self.set_font("Helvetica", "B", size)
            self.set_xy(x, y0 + 2.5)
            self.set_text_color(*_TEXT_SOFT)
            self.cell(w - 2, 3.4, lbl)
            self.set_xy(x, y0 + 8)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*_TEXT)
            self.cell(w - 2, 7, value)
            x += w
        self.set_y(y0 + 17)
        self.set_draw_color(*_RULE_LIGHT)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def figure(self, img_bytes: bytes | None, caption: str, source: str,
               w: float = 172, h: float = 80):
        """Centred figure with accent 'Figure N' caption and source line."""
        if img_bytes is None:
            return
        self._fig_no += 1
        needed = h + 16
        if self.get_y() + needed > self.h - 26:
            self.add_page()
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*_ACCENT)
        self.multi_cell(0, 4.4, f"Figure {self._fig_no}: {caption}", align="C")
        self.ln(0.5)
        x = (self.w - w) / 2
        self.image(io.BytesIO(img_bytes), x=x, w=w, h=h)
        self.ln(1)
        self.set_font("Helvetica", "", 6.8)
        self.set_text_color(*_TEXT_SOFT)
        self.cell(0, 3.6, source, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_TEXT)
        self.ln(3)

    def callout(self, title: str, text: str, w: float | None = None):
        """Light filled callout box with bold header (QIS sidebar style)."""
        if w is None:
            w = self.w - self.l_margin - self.r_margin
        x0, y0 = self.l_margin, self.get_y()
        # measure: render text invisibly first via split_only
        self.set_font("Helvetica", "", 8)
        lines = self.multi_cell(w - 8, 4.2, _safe(text), dry_run=True, output="LINES")
        box_h = 9 + len(lines) * 4.2 + 4
        if y0 + box_h > self.h - 26:
            self.add_page()
            y0 = self.get_y()
        self.set_fill_color(*_PANEL)
        try:
            self.rect(x0, y0, w, box_h, style="F", round_corners=True, corner_radius=2)
        except TypeError:
            self.rect(x0, y0, w, box_h, style="F")
        self.set_xy(x0 + 4, y0 + 3)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*_TEXT)
        self.cell(w - 8, 5, title)
        self.set_xy(x0 + 4, y0 + 9)
        self.set_font("Helvetica", "", 8)
        self.multi_cell(w - 8, 4.2, text)
        self.set_y(y0 + box_h + 4)


# ──────────────────────────────────────────────────────────────────────────────
# Figure export helper
# ──────────────────────────────────────────────────────────────────────────────

def _fig_to_png(fig, width: int = 900, height: int = 420) -> bytes | None:
    """
    Export a Plotly figure to a print-sharp PNG. The in-chart title is
    stripped — the PDF's 'Figure N' caption carries it (QIS convention) and a
    duplicated title wastes vertical space. Returns None on failure.
    """
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
        fig = go.Figure(fig)
        fig.update_layout(title=None, margin=dict(t=24, b=40))
        return pio.to_image(fig, format="png", width=width, height=height,
                            scale=2, engine="kaleido")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Page builders
# ──────────────────────────────────────────────────────────────────────────────

def _term_rows(terms, lang: str) -> list[tuple[str, str]]:
    rows = [
        (_t("maturity",         lang), f"{terms.maturity:g}Y ({terms.n_obs} observations, {terms.payment_freq})"),
        (_t("coupon_pa",        lang), f"{terms.coupon_pa * 100:.2f}%  ({terms.coupon_rate * 100:.4f}% per period)"),
        (_t("coupon_barrier",   lang), f"{terms.coupon_barrier:.1%}" if terms.coupon_barrier > 0 else
                                       ("Guaranteed (0%)" if lang == "en" else "Garantizado (0%)")),
        (_t("memory",           lang), _t("yes", lang) if terms.memory else _t("no", lang)),
        (_t("autocall_barrier", lang), f"{terms.autocall_barrier:.1%}"),
        (_t("autocall_start",   lang), f"P{terms.autocall_start_period}"),
        (_t("ki_barrier",       lang), f"{terms.knock_in_barrier:.1%}"),
        (_t("coupon_basket",    lang), terms.coupon_basket.replace("_", "-")),
        (_t("autocall_basket",  lang), terms.autocall_basket.replace("_", "-")),
        (_t("final_basket",     lang), f"{terms.final_basket.replace('_', '-')}"
                                       + (f"  ({_t('rescue_barrier', lang)} {terms.final_redemption_barrier:.0%})"
                                          if terms.final_basket == "best_of" else "")),
    ]
    if getattr(terms, "autocall_step_down", 0.0):
        rows.append((_t("ac_step_down", lang), f"{terms.autocall_step_down:.1%}"))
        if getattr(terms, "autocall_floor", None) is not None:
            rows.append((_t("ac_floor", lang), f"{terms.autocall_floor:.0%}"))
    if getattr(terms, "coupon_at_autocall_only", False):
        rows.append((_t("premium_at_call", lang),
                     f"{_t('yes', lang)} ({terms.coupon_pa * 100:.2f}% p.a.)"))
    if getattr(terms, "issue_date", None):
        rows.append((_t("issue_date", lang), terms.issue_date))
    return rows


def _exec_bullets(terms, results, bt_summary, live_data, lang: str) -> list[str]:
    """Compose executive-summary bullets from the available numbers."""
    b = []
    if lang == "es":
        b.append(
            f"La simulación Monte Carlo (modelo Heston multi-activo, "
            f"{len(results.get('annualized_returns', [])):,} caminos) estima una TIR anual simple "
            f"esperada de {results.get('expected_irr', 0):.1%} y un retorno total esperado de "
            f"{results.get('expected_total_return', 0):.1%} a vencimiento ({terms.maturity:g} años).")
        b.append(
            f"La probabilidad de autocall anticipado es {results.get('prob_autocall', 0):.0%}; "
            f"la probabilidad de pérdida de capital a vencimiento (knock-in sin rescate) es "
            f"{results.get('prob_knock_in_total', 0):.1%} con barrera al {terms.knock_in_barrier:.1%}.")
        if bt_summary:
            b.append(
                f"En el backtest histórico ({bt_summary.get('n_issues', 0)} fechas de emisión), la TIR media "
                f"realizada fue {bt_summary.get('mean_irr', 0):.1%} (mediana {bt_summary.get('median_irr', 0):.1%}), "
                f"con autocall en el {bt_summary.get('prob_called', 0):.0%} de los casos y knock-in en el "
                f"{bt_summary.get('prob_knock_in', 0):.1%}.")
        if live_data:
            b.append(
                f"Desde emisión, el worst-of cotiza al {live_data.get('wof_today', 0):.1%} del strike "
                f"({live_data.get('worst_asset', '')} es el peor activo); la TIR de cupones a fecha es "
                f"{live_data.get('irr_to_date', 0):.1%} anualizada.")
    else:
        b.append(
            f"Monte Carlo simulation (multi-asset Heston model, "
            f"{len(results.get('annualized_returns', [])):,} paths) estimates an expected simple "
            f"annualised IRR of {results.get('expected_irr', 0):.1%} and an expected total return of "
            f"{results.get('expected_total_return', 0):.1%} over the {terms.maturity:g}-year tenor.")
        b.append(
            f"The probability of early redemption (autocall) is {results.get('prob_autocall', 0):.0%}; "
            f"the probability of capital loss at maturity (knock-in without rescue) is "
            f"{results.get('prob_knock_in_total', 0):.1%} against a {terms.knock_in_barrier:.1%} barrier.")
        if bt_summary:
            b.append(
                f"Across {bt_summary.get('n_issues', 0)} historical issue dates, the realised mean IRR was "
                f"{bt_summary.get('mean_irr', 0):.1%} (median {bt_summary.get('median_irr', 0):.1%}); the note "
                f"autocalled in {bt_summary.get('prob_called', 0):.0%} of cases and knocked in on "
                f"{bt_summary.get('prob_knock_in', 0):.1%}.")
        if live_data:
            b.append(
                f"Since issue, the worst-of basket trades at {live_data.get('wof_today', 0):.1%} of strike "
                f"({live_data.get('worst_asset', '')} is the worst performer); coupon IRR to date is "
                f"{live_data.get('irr_to_date', 0):.1%} annualised.")
    return b


def _cover_page(pdf: _NotePDF, terms, results, asset_names, bt_summary, live_data, lang: str):
    pdf._is_cover = True
    pdf.add_page()

    # Top hairline + micro-disclaimer (Barclays style)
    pdf.set_draw_color(*_HAIRLINE)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, 12, pdf.w - pdf.r_margin, 12)
    pdf.set_y(13.5)
    pdf.set_font("Helvetica", "", 6.3)
    pdf.set_text_color(*_TEXT_SOFT)
    pdf.multi_cell(0, 2.9, _t("cover_topline", lang), align="C")
    pdf.line(pdf.l_margin, pdf.get_y() + 1.5, pdf.w - pdf.r_margin, pdf.get_y() + 1.5)

    # Sidebar panel (right)
    sb_x, sb_w = 138, pdf.w - pdf.r_margin - 138
    sb_y, sb_h = 34, 150
    pdf.set_fill_color(*_PANEL)
    pdf.rect(sb_x, sb_y, sb_w, sb_h, style="F")

    def _sb_text(y, txt, style="", size=8, color=_TEXT):
        pdf.set_xy(sb_x + 5, y)
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(sb_w - 10, 4.2, _safe(txt))
        return pdf.get_y()

    y = _sb_text(sb_y + 6, datetime.date.today().strftime("%-d %B %Y")
                 if hasattr(datetime.date.today(), "strftime") else str(datetime.date.today()),
                 "", 8.5, _TEXT)
    y = _sb_text(y + 5, _t("underlyings", lang), "B", 8, _ACCENT)
    for nm in asset_names:
        y = _sb_text(y + 0.5, nm, "B", 8.5)
    y = _sb_text(y + 5, _t("key_terms", lang), "B", 8, _ACCENT)
    mini = [
        (_t("maturity", lang), f"{terms.maturity:g}Y {terms.payment_freq}"),
        (_t("coupon_pa", lang), f"{terms.coupon_pa*100:.2f}%"),
        (_t("autocall_barrier", lang), f"{terms.autocall_barrier:.0%}"),
        (_t("ki_barrier", lang).split(" (")[0], f"{terms.knock_in_barrier:.1%}"),
    ]
    if getattr(terms, "issue_date", None):
        mini.append((_t("issue_date", lang), terms.issue_date))
    for k, v in mini:
        pdf.set_xy(sb_x + 5, y + 1.2)
        pdf.set_font("Helvetica", "", 7.3)
        pdf.set_text_color(*_TEXT_SOFT)
        pdf.cell(sb_w - 10, 3.6, _safe(k))
        pdf.set_xy(sb_x + 5, y + 4.8)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*_TEXT)
        pdf.cell(sb_w - 10, 4, _safe(v))
        y += 9.5

    # Main column (left of sidebar)
    main_w = sb_x - pdf.l_margin - 8

    pdf.set_xy(pdf.l_margin, 32)
    if pdf.issuer:
        pdf.set_font("Helvetica", "B", 21)
        pdf.set_text_color(*_BRAND)
        pdf.cell(main_w, 10, pdf.issuer.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*_TEXT)
    pdf.cell(main_w, 8, _t("series_title", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.set_text_color(*_ACCENT)
    pdf.multi_cell(main_w, 6.2, _safe(terms.name))
    pdf.ln(6)

    # Executive summary bullets
    for txt in _exec_bullets(terms, results, bt_summary, live_data, lang):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 9.2)
        pdf.set_text_color(*_TEXT)
        pdf.cell(5, 5.2, chr(149))
        pdf.multi_cell(main_w - 5, 5.2, _safe(txt), align="J")
        pdf.ln(2.5)

    # Contents block
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_ACCENT)
    pdf.cell(main_w, 5, _t("in_this_report", lang).upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_HAIRLINE)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.l_margin + main_w, pdf.get_y() + 0.5)
    pdf.ln(2)
    toc = [_t("note_terms", lang), _t("sim_summary", lang), _t("calibration", lang)]
    if bt_summary:
        toc.append(_t("backtest", lang))
    if live_data:
        toc.append(_t("live", lang))
    toc.append(_t("disclaimer_title", lang))
    for item in toc:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*_TEXT)
        pdf.cell(main_w, 5.4, item, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_RULE_LIGHT)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + main_w, pdf.get_y())

    pdf._is_cover = False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    terms,
    results: dict,
    asset_names: list[str],
    figures: dict,
    lang: str = "en",
    bt_summary: dict | None = None,
    bt_figures: dict | None = None,
    live_data: dict | None = None,
    live_figure=None,
) -> bytes:
    """
    Build the full institutional-style PDF report: cover with executive
    summary, note terms, Monte Carlo section, calibration, historical
    backtest, current performance (when available), and disclaimers.
    Returns raw PDF bytes.
    """
    issuer  = getattr(terms, "issuer", "") or ""
    doc_ref = f"{_t('series_title', lang)} | {terms.name}"
    pdf = _NotePDF(lang=lang, issuer=issuer, doc_ref=doc_ref)

    # ── 1. Cover ───────────────────────────────────────────────────────────
    _cover_page(pdf, terms, results, asset_names, bt_summary, live_data, lang)

    # ── 2. Note terms ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title(_t("note_terms", lang))
    pdf.kv_table(_term_rows(terms, lang))

    pdf.subsection(_t("obs_schedule", lang))
    obs_times = terms.obs_times()
    sched     = terms.autocall_barrier_schedule()
    ac_rows = []
    for i, t_obs in enumerate(obs_times):
        eligible = _t("yes", lang) if (i + 1) >= terms.autocall_start_period else _t("no", lang)
        ac_rows.append([f"P{i+1}", f"{t_obs:.3g}", f"{sched[i]:.1%}", eligible])
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.data_table(
        [_t("period", lang), _t("time_y", lang), _t("ac_level", lang), _t("eligible", lang)],
        ac_rows,
        col_widths=[usable * 0.2, usable * 0.25, usable * 0.3, usable * 0.25],
        aligns=["L", "R", "R", "R"],
    )

    pdf.callout(_t("model_box_title", lang), _t("model_box_body", lang))

    # ── 3. Monte Carlo ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title(_t("sim_summary", lang))
    n_paths_val = int(np.asarray(results.get("annualized_returns", np.array([]))).shape[0])
    pdf.metric_band([
        (_t("expected_irr",          lang), f"{results.get('expected_irr', 0):.2%}"),
        (_t("total_return_short", lang), f"{results.get('expected_total_return', 0):.2%}"),
        (_t("prob_autocall",         lang), f"{results.get('prob_autocall', 0):.1%}"),
        (_t("prob_knock_in",         lang), f"{results.get('prob_knock_in_total', 0):.2%}"),
        (_t("n_paths",               lang), f"{n_paths_val:,}"),
    ])

    src_mc = f"{_t('src_mc', lang)}, {n_paths_val:,} paths"
    pdf.figure(_fig_to_png(figures.get("irr_dist")), _t("fig_irr", lang), src_mc)
    pdf.figure(_fig_to_png(figures.get("wof_fan")),  _t("fig_wof", lang), src_mc)

    # Autocall probability by period
    prob_by_period = results.get("prob_autocall_by_period", [])
    if prob_by_period:
        pdf.subsection(_t("autocall_by_period", lang))
        rows = []
        for i, (t_obs, p_ac) in enumerate(zip(obs_times, prob_by_period)):
            eligible = _t("yes", lang) if (i + 1) >= terms.autocall_start_period else _t("no", lang)
            rows.append([f"P{i+1}", f"{t_obs:.3g}", f"{p_ac:.2%}", eligible])
        pdf.data_table(
            [_t("period", lang), _t("time_y", lang), _t("p_autocall", lang), _t("eligible", lang)],
            rows,
            col_widths=[usable * 0.2, usable * 0.25, usable * 0.3, usable * 0.25],
            aligns=["L", "R", "R", "R"],
        )

    # ── 4. Calibration ─────────────────────────────────────────────────────
    params = results.get("params", [])
    if params:
        pdf.add_page()
        pdf.section_title(_t("calibration", lang))
        hp_rows = []
        for p in params:
            try:
                ok, _ = p.feller_condition()
            except Exception:
                ok = False
            hp_rows.append([
                str(p.name), f"{p.S0:,.1f}", f"{p.mu * 100:.1f}%",
                f"{np.sqrt(p.V0) * 100:.1f}%", f"{np.sqrt(p.theta) * 100:.1f}%",
                f"{p.kappa:.2f}", f"{p.xi:.2f}", f"{p.rho:.2f}",
                "OK" if ok else "!",
            ])
        pdf.data_table(
            [_t("asset", lang), "S0", "mu p.a.", "Vol (V0)", "Vol (theta)",
             "kappa", "xi", "rho", _t("feller", lang)],
            hp_rows,
            col_widths=[usable * 0.18] + [usable * 0.1025] * 8,
        )
        pdf.figure(_fig_to_png(figures.get("corr"), width=560, height=460),
                   _t("fig_corr", lang), _t("src_hist", lang), w=105, h=86)

    # ── 5. Historical backtest ─────────────────────────────────────────────
    if bt_summary:
        pdf.add_page()
        pdf.section_title(_t("backtest", lang))
        pdf.metric_band([
            (_t("bt_n_issues",       lang), str(bt_summary.get("n_issues", 0))),
            (_t("bt_mean_irr",       lang), f"{bt_summary.get('mean_irr', 0):.2%}"),
            (_t("bt_median_irr",     lang), f"{bt_summary.get('median_irr', 0):.2%}"),
            (_t("bt_autocalled_pct", lang), f"{bt_summary.get('prob_called', 0):.1%}"),
            (_t("bt_knock_in_pct",   lang), f"{bt_summary.get('prob_knock_in', 0):.1%}"),
        ])
        if bt_figures:
            pdf.figure(_fig_to_png(bt_figures.get("outcome")),
                       _t("fig_bt_outcome", lang), _t("src_hist", lang))
            pdf.figure(_fig_to_png(bt_figures.get("irr_scatter")),
                       _t("fig_bt_irr", lang), _t("src_hist", lang))

    # ── 6. Current performance ─────────────────────────────────────────────
    if live_data:
        pdf.add_page()
        pdf.section_title(_t("live", lang))
        pdf.metric_band([
            (_t("live_wof_today",   lang), f"{live_data.get('wof_today', 0):.1%}"),
            (_t("live_worst_asset", lang), str(live_data.get("worst_asset", ""))),
            (_t("live_irr_to_date", lang), f"{live_data.get('irr_to_date', 0):.2%}"),
            (_t("live_elapsed",     lang), f"{live_data.get('elapsed_years', 0):.2f}"),
        ])

        perf_today = live_data.get("perf_today", {})
        if perf_today:
            pdf.subsection(_t("live_asset_perf", lang))
            pdf.data_table(
                [_t("asset", lang), _t("performance", lang)],
                [[name, f"{perf:.1%}"] for name, perf in perf_today.items()],
                col_widths=[usable * 0.5, usable * 0.5],
            )

        obs_rows = live_data.get("obs_rows", [])
        if obs_rows:
            pdf.subsection(_t("live_obs_history", lang))
            obs_headers = list(obs_rows[0].keys())
            obs_data    = [[str(r.get(h, "")) for h in obs_headers] for r in obs_rows]
            n_cols = len(obs_headers)
            if n_cols == 6:
                # Period | Date | Status | Worst-of | Coupon | Cumulative —
                # Status needs the width (e.g. "No periodic coupon (premium at call)")
                obs_w = [usable * f for f in (0.08, 0.13, 0.37, 0.14, 0.13, 0.15)]
            else:
                obs_w = [usable / n_cols] * n_cols
            pdf.data_table(obs_headers, obs_data, col_widths=obs_w,
                           aligns=["L"] * n_cols)

        pdf.figure(_fig_to_png(live_figure), _t("fig_live", lang), _t("src_hist", lang))

    # ── 7. Disclaimers ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title(_t("disclaimer_title", lang))
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*_TEXT_SOFT)
    for para in _t("disclaimer_body", lang).split("\n\n"):
        pdf.multi_cell(0, 3.8, _safe(para))
        pdf.ln(2)
    pdf.set_text_color(*_TEXT)

    return bytes(pdf.output())
