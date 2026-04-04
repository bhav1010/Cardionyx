"""
Cardionyx Desktop Application
==============================
PyQt5 GUI frontend for the CAD prediction pipeline.
Run with:  python cardionyx_app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── ensure the app finds pipeline modules next to this file ──────────────────
APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QScrollArea, QGroupBox,
    QStackedWidget, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy, QProgressBar, QMessageBox,
    QSplitter, QToolButton,
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QSize,
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QFontDatabase,
    QPixmap, QPalette, QLinearGradient, QIcon, QDragEnterEvent,
    QDropEvent,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE & STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":           "#0A0F1E",
    "surface":      "#111827",
    "surface2":     "#1A2236",
    "border":       "#1E2D45",
    "accent":       "#00D4FF",
    "accent_dark":  "#0099BB",
    "text":         "#E2E8F0",
    "text_dim":     "#64748B",
    "low":          "#22C55E",
    "moderate":     "#F59E0B",
    "high":         "#EF4444",
    "high_dark":    "#B91C1C",
}

QSS = f"""
QMainWindow, QWidget {{
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {PALETTE['surface']};
    width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {PALETTE['border']};
    border-radius: 4px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}

/* ── Group boxes ── */
QGroupBox {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    margin-top: 18px;
    padding: 12px 14px 14px 14px;
    font-weight: 600;
    font-size: 12px;
    color: {PALETTE['accent']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px; top: -1px;
    padding: 2px 6px;
    background: {PALETTE['surface']};
    color: {PALETTE['accent']};
    border-radius: 4px;
}}

/* ── Inputs ── */
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 5px 8px;
    color: {PALETTE['text']};
    selection-background-color: {PALETTE['accent']};
}}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {PALETTE['accent']};
}}
QComboBox::drop-down {{
    border: none; width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    selection-background-color: {PALETTE['accent_dark']};
    color: {PALETTE['text']};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background: {PALETTE['border']};
    border-radius: 3px;
    width: 16px;
}}

/* ── CheckBox ── */
QCheckBox {{
    spacing: 8px;
    color: {PALETTE['text']};
}}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border-radius: 4px;
    border: 1px solid {PALETTE['border']};
    background: {PALETTE['surface2']};
}}
QCheckBox::indicator:checked {{
    background: {PALETTE['accent']};
    border-color: {PALETTE['accent']};
    image: none;
}}

/* ── Buttons ── */
QPushButton {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 7px 16px;
    color: {PALETTE['text']};
    font-weight: 500;
}}
QPushButton:hover {{
    background: {PALETTE['border']};
    border-color: {PALETTE['accent']};
}}
QPushButton#primary {{
    background: {PALETTE['accent']};
    border: none;
    color: #0A0F1E;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 28px;
    border-radius: 10px;
}}
QPushButton#primary:hover {{
    background: {PALETTE['accent_dark']};
}}
QPushButton#danger {{
    background: {PALETTE['high']};
    border: none;
    color: white;
    font-weight: 600;
}}
QPushButton#danger:hover {{
    background: {PALETTE['high_dark']};
}}

/* ── Tables ── */
QTableWidget {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    gridline-color: {PALETTE['border']};
    color: {PALETTE['text']};
}}
QTableWidget::item:selected {{
    background: {PALETTE['accent_dark']};
}}
QHeaderView::section {{
    background: {PALETTE['surface2']};
    border: none;
    border-bottom: 1px solid {PALETTE['border']};
    padding: 6px 10px;
    font-weight: 600;
    color: {PALETTE['accent']};
}}

/* ── Progress bar ── */
QProgressBar {{
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    background: {PALETTE['surface2']};
    text-align: center;
    color: {PALETTE['text']};
}}
QProgressBar::chunk {{
    background: {PALETTE['accent']};
    border-radius: 5px;
}}

/* ── Label ── */
QLabel#dim {{ color: {PALETTE['text_dim']}; font-size: 11px; }}
QLabel#section {{ color: {PALETTE['text']}; font-weight: 600; font-size: 13px; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKERS
# ─────────────────────────────────────────────────────────────────────────────

class YOLOWorker(QThread):
    results_ready = pyqtSignal(list)   # list of detection dicts
    error         = pyqtSignal(str)

    def __init__(self, image_path: str, yolo_model_path: str):
        super().__init__()
        self.image_path      = image_path
        self.yolo_model_path = yolo_model_path

    def run(self):
        try:
            from ultralytics import YOLO
            model   = YOLO(self.yolo_model_path)
            results = model(self.image_path, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id     = int(box.cls[0])
                    conf       = float(box.conf[0])
                    class_name = r.names[cls_id]
                    bbox       = [round(c, 1) for c in box.xyxy[0].tolist()]
                    dets.append({"class": class_name, "confidence": round(conf, 4), "bbox": bbox})
            self.results_ready.emit(dets)
        except Exception as e:
            self.error.emit(str(e))


class PredictionWorker(QThread):
    prediction_ready = pyqtSignal(dict)
    error            = pyqtSignal(str)

    def __init__(self, yolo_detections, patient_data, model_path, conf_threshold):
        super().__init__()
        self.yolo_detections = yolo_detections
        self.patient_data    = patient_data
        self.model_path      = model_path
        self.conf_threshold  = conf_threshold

    def run(self):
        try:
            from cad_pipeline import CADPredictor, CADPipelineConfig
            config    = CADPipelineConfig(model_path=self.model_path,
                                         confidence_threshold=self.conf_threshold)
            predictor = CADPredictor(self.model_path, config=config)
            result    = predictor.predict(self.yolo_detections, self.patient_data,
                                          return_details=True)
            self.prediction_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATED GAUGE WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class GaugeWidget(QWidget):
    """Custom arc-based probability gauge drawn with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value   = 0        # 0–100
        self._target  = 0
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setMinimumSize(260, 160)

    def set_value(self, v: float):      # v in [0,1]
        self._target = int(v * 100)
        self._timer.start(12)

    def _tick(self):
        if self._value < self._target:
            self._value = min(self._value + 1, self._target)
        elif self._value > self._target:
            self._value = max(self._value - 1, self._target)
        else:
            self._timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        cx, cy = w // 2, h - 20
        r      = min(cx, cy) - 14
        rect   = QRect(cx - r, cy - r, 2 * r, 2 * r)

        # Background arc
        bg_pen = QPen(QColor(PALETTE['border']), 14, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0 * 16, 180 * 16)   # 180° semicircle

        # Value arc
        v = self._value / 100.0
        if v < 0.30:
            col = QColor(PALETTE['low'])
        elif v < 0.60:
            col = QColor(PALETTE['moderate'])
        else:
            col = QColor(PALETTE['high'])

        val_pen = QPen(col, 14, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(val_pen)
        span = int(180 * v * 16)
        painter.drawArc(rect, 0 * 16, span)

        # Centre text
        painter.setPen(QColor(PALETTE['text']))
        font = QFont("Segoe UI", 28, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRect(cx - 80, cy - 60, 160, 56),
                         Qt.AlignCenter, f"{self._value}%")

        painter.setPen(QColor(PALETTE['text_dim']))
        font2 = QFont("Segoe UI", 10)
        painter.setFont(font2)
        painter.drawText(QRect(cx - 80, cy - 10, 160, 22),
                         Qt.AlignCenter, "CAD Probability")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR BUTTON
# ─────────────────────────────────────────────────────────────────────────────

class SidebarButton(QPushButton):
    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.icon_text = icon_text
        self.label_txt = label
        self.setCheckable(True)
        self.setFixedHeight(60)
        self.setCursor(Qt.PointingHandCursor)
        self._active = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.isChecked():
            painter.fillRect(self.rect(), QColor(PALETTE['surface2']))
            # left accent bar
            painter.fillRect(0, 8, 3, self.height() - 16, QColor(PALETTE['accent']))

        # icon
        painter.setPen(QColor(PALETTE['accent'] if self.isChecked() else PALETTE['text_dim']))
        font = QFont("Segoe UI", 18)
        painter.setFont(font)
        painter.drawText(QRect(14, 0, 36, self.height()), Qt.AlignVCenter, self.icon_text)

        # label
        painter.setPen(QColor(PALETTE['text'] if self.isChecked() else PALETTE['text_dim']))
        font2 = QFont("Segoe UI", 11, QFont.Bold if self.isChecked() else QFont.Normal)
        painter.setFont(font2)
        painter.drawText(QRect(54, 0, self.width() - 60, self.height()),
                         Qt.AlignVCenter, self.label_txt)


# ─────────────────────────────────────────────────────────────────────────────
# ECG DROP ZONE
# ─────────────────────────────────────────────────────────────────────────────

class DropZone(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(160)
        self.setText("🖼  Drag & Drop ECG Image Here\nor click Browse")
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {PALETTE['border']};
                border-radius: 12px;
                background: {PALETTE['surface']};
                color: {PALETTE['text_dim']};
                font-size: 14px;
            }}
        """)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(f"""
                QLabel {{
                    border: 2px dashed {PALETTE['accent']};
                    border-radius: 12px;
                    background: {PALETTE['surface2']};
                    color: {PALETTE['accent']};
                    font-size: 14px;
                }}
            """)

    def dragLeaveEvent(self, e):
        self._reset_style()

    def dropEvent(self, e: QDropEvent):
        self._reset_style()
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)

    def _reset_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {PALETTE['border']};
                border-radius: 12px;
                background: {PALETTE['surface']};
                color: {PALETTE['text_dim']};
                font-size: 14px;
            }}
        """)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: labelled form row
# ─────────────────────────────────────────────────────────────────────────────

def form_row(label_text: str, widget: QWidget, unit: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(220)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {PALETTE['text']}; font-size: 12px;")
    row.addWidget(lbl)
    row.addWidget(widget)
    if unit:
        u = QLabel(unit)
        u.setFixedWidth(60)
        u.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px;")
        row.addWidget(u)
    row.addStretch()
    return row


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {PALETTE['border']};")
    return line


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PATIENT INFORMATION
# ─────────────────────────────────────────────────────────────────────────────

class PatientPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.setWidget(container)
        outer = QVBoxLayout(container)
        outer.setSpacing(16)
        outer.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🧑‍⚕️  Patient Clinical Data")
        title.setStyleSheet(f"font-size:20px; font-weight:700; color:{PALETTE['text']};")
        outer.addWidget(title)
        sub = QLabel("Fill in patient details below. Defaults are shown where available.")
        sub.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:12px;")
        outer.addWidget(sub)
        outer.addWidget(separator())

        # ── build all widgets ────────────────────────────────────────────────
        self.widgets: dict[str, QWidget] = {}

        # ── Demographics ─────────────────────────────────────────────────────
        grp = QGroupBox("Demographics")
        gl  = QVBoxLayout(grp)

        self.widgets["Age"]    = self._spin(1, 120, 0, 50, "years")
        self.widgets["Weight"] = self._dspin(20, 250, 1, 70, "kg")
        self.widgets["Length"] = self._dspin(50, 250, 1, 170, "cm")

        self.widgets["Sex"] = QComboBox()
        self.widgets["Sex"].addItems(["Male", "Female"])

        self.widgets["BMI"] = self._dspin(10, 70, 1, 0, "kg/m²")
        self._bmi_auto = QCheckBox("Auto-calculate from Weight & Height")
        self._bmi_auto.setChecked(True)
        self._bmi_auto.stateChanged.connect(self._toggle_bmi)
        self.widgets["BMI"].setEnabled(False)

        for lbl, key, unit in [
            ("Age", "Age", "years"), ("Weight", "Weight", "kg"),
            ("Height", "Length", "cm"),
        ]:
            gl.addLayout(form_row(lbl, self.widgets[key], unit))

        row_sex = form_row("Sex", self.widgets["Sex"])
        gl.addLayout(row_sex)
        gl.addLayout(form_row("BMI", self.widgets["BMI"], "kg/m²"))
        gl.addWidget(self._bmi_auto)

        # Connect weight/length to re-calc BMI
        self.widgets["Weight"].valueChanged.connect(self._calc_bmi)
        self.widgets["Length"].valueChanged.connect(self._calc_bmi)
        outer.addWidget(grp)

        # ── Medical History ───────────────────────────────────────────────────
        grp2 = QGroupBox("Medical History")
        gl2  = QVBoxLayout(grp2)
        history_fields = [
            ("DM",              "Diabetes Mellitus",             "binary"),
            ("HTN",             "Hypertension",                  "binary"),
            ("Current Smoker",  "Current Smoker",                "binary"),
            ("EX-Smoker",       "Ex-Smoker",                     "binary"),
            ("FH",              "Family History of CAD",         "binary"),
            ("Obesity",         "Obesity",                       "yn"),
            ("CRF",             "Chronic Renal Failure",         "yn"),
            ("CVA",             "Cerebrovascular Accident",      "yn"),
            ("Airway disease",  "Airway Disease",                "yn"),
            ("Thyroid Disease", "Thyroid Disease",               "yn"),
            ("CHF",             "Congestive Heart Failure",      "yn"),
            ("DLP",             "Dyslipidemia",                  "yn"),
        ]
        for key, lbl, itype in history_fields:
            cb = QCheckBox()
            self.widgets[key] = cb
            gl2.addLayout(form_row(lbl, cb))
        outer.addWidget(grp2)

        # ── Vitals & Exam ─────────────────────────────────────────────────────
        grp3 = QGroupBox("Vitals & Physical Examination")
        gl3  = QVBoxLayout(grp3)
        self.widgets["BP"] = self._dspin(40, 250, 0, 120, "mmHg")
        self.widgets["PR"] = self._dspin(20, 250, 0, 75, "/min")
        gl3.addLayout(form_row("Blood Pressure — MAP", self.widgets["BP"], "mmHg"))
        gl3.addLayout(form_row("Pulse Rate", self.widgets["PR"], "/min"))
        for key, lbl in [
            ("Edema", "Edema"), ("Weak Peripheral Pulse", "Weak Peripheral Pulse"),
            ("Lung rales", "Lung Rales"), ("Systolic Murmur", "Systolic Murmur"),
            ("Diastolic Murmur", "Diastolic Murmur"),
        ]:
            cb = QCheckBox()
            self.widgets[key] = cb
            gl3.addLayout(form_row(lbl, cb))
        outer.addWidget(grp3)

        # ── Symptoms ──────────────────────────────────────────────────────────
        grp4 = QGroupBox("Symptoms")
        gl4  = QVBoxLayout(grp4)
        for key, lbl in [
            ("Typical Chest Pain", "Typical Chest Pain"),
            ("Dyspnea", "Dyspnea"),
            ("Atypical", "Atypical Angina"),
            ("Nonanginal", "Nonanginal Chest Pain"),
            ("Exertional CP", "Exertional Chest Pain"),
            ("LowTH Ang", "Low Threshold Angina"),
        ]:
            cb = QCheckBox()
            self.widgets[key] = cb
            gl4.addLayout(form_row(lbl, cb))

        self.widgets["Function Class"] = QComboBox()
        self.widgets["Function Class"].addItems(["0 — None", "1 — Mild", "2 — Moderate", "3 — Severe"])
        gl4.addLayout(form_row("NYHA Function Class", self.widgets["Function Class"]))
        outer.addWidget(grp4)

        # ── Lab Results ───────────────────────────────────────────────────────
        grp5 = QGroupBox("Laboratory Results  (press ↑↓ or type — leave 0 if unknown)")
        gl5  = QVBoxLayout(grp5)
        lab_fields = [
            ("FBS",  "Fasting Blood Sugar",  0, 600, "mg/dL"),
            ("CR",   "Creatinine",           0, 20,  "mg/dL"),
            ("TG",   "Triglycerides",        0, 2000,"mg/dL"),
            ("LDL",  "LDL Cholesterol",      0, 400, "mg/dL"),
            ("HDL",  "HDL Cholesterol",      0, 200, "mg/dL"),
            ("BUN",  "Blood Urea Nitrogen",  0, 200, "mg/dL"),
            ("ESR",  "ESR",                  0, 200, "mm/hr"),
            ("HB",   "Hemoglobin",           0, 25,  "g/dL"),
            ("K",    "Potassium",            0, 10,  "mmol/L"),
            ("Na",   "Sodium",               0, 200, "mmol/L"),
            ("WBC",  "WBC Count",            0, 50000,"/dL"),
            ("Lymph","Lymphocyte %",         0, 100, "%"),
            ("Neut", "Neutrophil %",         0, 100, "%"),
            ("PLT",  "Platelet Count",       0, 1000,"×10³/μL"),
        ]
        for key, lbl, lo, hi, unit in lab_fields:
            w = self._dspin(lo, hi, 1, 0, unit)
            self.widgets[key] = w
            gl5.addLayout(form_row(lbl, w, unit))
        outer.addWidget(grp5)

        # ── Heart Tests ───────────────────────────────────────────────────────
        grp6 = QGroupBox("Cardiac Investigations")
        gl6  = QVBoxLayout(grp6)
        self.widgets["EF-TTE"]      = self._dspin(5, 80, 0, 55, "%")
        self.widgets["Region RWMA"] = self._spin(0, 4, 0, 0, "")
        self.widgets["VHD"]         = QComboBox()
        self.widgets["VHD"].addItems(["N", "mild", "moderate", "Severe"])
        gl6.addLayout(form_row("Ejection Fraction EF-TTE", self.widgets["EF-TTE"], "%"))
        gl6.addLayout(form_row("Region RWMA (0–4)", self.widgets["Region RWMA"]))
        gl6.addLayout(form_row("Valvular Heart Disease", self.widgets["VHD"]))
        outer.addWidget(grp6)
        outer.addStretch()

    # ── widget factories ─────────────────────────────────────────────────────
    @staticmethod
    def _dspin(lo, hi, decimals, default, _unit="") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(default)
        w.setFixedWidth(120)
        w.setButtonSymbols(QDoubleSpinBox.PlusMinus)
        return w

    @staticmethod
    def _spin(lo, hi, decimals, default, _unit="") -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        w.setFixedWidth(120)
        return w

    def _toggle_bmi(self, state):
        self.widgets["BMI"].setEnabled(state == Qt.Unchecked)
        if state == Qt.Checked:
            self._calc_bmi()

    def _calc_bmi(self):
        if self._bmi_auto.isChecked():
            w = self.widgets["Weight"].value()
            h = self.widgets["Length"].value() / 100.0
            if h > 0:
                self.widgets["BMI"].setValue(round(w / (h ** 2), 1))

    # ── collect values ───────────────────────────────────────────────────────
    def get_patient_data(self) -> dict:
        d = {}

        # Demographics
        d["Age"]    = self.widgets["Age"].value()
        d["Weight"] = self.widgets["Weight"].value()
        d["Length"] = self.widgets["Length"].value()
        d["Sex"]    = self.widgets["Sex"].currentText()
        d["BMI"]    = self.widgets["BMI"].value()

        # Medical history (bool checkboxes → Y/N or 0/1)
        binary_keys = ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
                       "Typical Chest Pain", "Exertional CP", "LowTH Ang",
                       "Edema"]
        yn_keys = ["Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
                   "CHF", "DLP", "Weak Peripheral Pulse", "Lung rales",
                   "Systolic Murmur", "Diastolic Murmur", "Dyspnea",
                   "Atypical", "Nonanginal"]

        for k in binary_keys:
            d[k] = 1 if self.widgets[k].isChecked() else 0
        for k in yn_keys:
            d[k] = "Y" if self.widgets[k].isChecked() else "N"

        d["BP"] = self.widgets["BP"].value()
        d["PR"] = self.widgets["PR"].value()

        d["Function Class"] = self.widgets["Function Class"].currentIndex()

        for k in ["FBS","CR","TG","LDL","HDL","BUN","ESR","HB","K","Na",
                  "WBC","Lymph","Neut","PLT","EF-TTE"]:
            v = self.widgets[k].value()
            d[k] = v if v > 0 else np.nan

        d["Region RWMA"] = self.widgets["Region RWMA"].value()
        d["VHD"]         = self.widgets["VHD"].currentText()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ECG / IMAGING
# ─────────────────────────────────────────────────────────────────────────────

class ECGPage(QWidget):
    yolo_done = pyqtSignal(list)   # forwarded to MainWindow

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path     = ""
        self._yolo_model     = str(APP_DIR / "training_runs/yolo_run5/weights/best.pt")
        self._yolo_detections: list[dict] = []
        self._worker: YOLOWorker | None   = None

        outer = QVBoxLayout(self)
        outer.setSpacing(16)
        outer.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📈  ECG Image Analysis")
        title.setStyleSheet(f"font-size:20px; font-weight:700; color:{PALETTE['text']};")
        outer.addWidget(title)
        sub = QLabel("Upload an ECG image for automatic YOLO abnormality detection, or enter ECG features manually.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:12px;")
        outer.addWidget(sub)
        outer.addWidget(separator())

        # ── Image upload ──────────────────────────────────────────────────────
        img_grp = QGroupBox("ECG Image Upload")
        il      = QVBoxLayout(img_grp)

        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._set_image)
        il.addWidget(self._drop_zone)

        browse_row = QHBoxLayout()
        btn_browse = QPushButton("📂  Browse Image…")
        btn_browse.clicked.connect(self._browse_image)
        self._img_path_label = QLabel("No image selected")
        self._img_path_label.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:11px;")
        browse_row.addWidget(btn_browse)
        browse_row.addWidget(self._img_path_label, 1)
        il.addLayout(browse_row)

        # Preview
        self._preview = QLabel()
        self._preview.setFixedHeight(150)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.hide()
        il.addWidget(self._preview)

        outer.addWidget(img_grp)

        # ── YOLO model path ───────────────────────────────────────────────────
        yolo_grp = QGroupBox("YOLO Model")
        yl       = QHBoxLayout(yolo_grp)
        self._yolo_path_edit = QLineEdit(self._yolo_model)
        btn_yolo = QPushButton("Browse…")
        btn_yolo.clicked.connect(self._browse_yolo)
        yl.addWidget(QLabel("Weights path:"))
        yl.addWidget(self._yolo_path_edit, 1)
        yl.addWidget(btn_yolo)
        outer.addWidget(yolo_grp)

        # Run YOLO button
        self._run_yolo_btn = QPushButton("⚡  Run YOLO Analysis")
        self._run_yolo_btn.setObjectName("primary")
        self._run_yolo_btn.setEnabled(False)
        self._run_yolo_btn.clicked.connect(self._run_yolo)
        outer.addWidget(self._run_yolo_btn)

        self._yolo_status = QLabel("")
        self._yolo_status.setStyleSheet(f"color:{PALETTE['accent']}; font-size:12px;")
        outer.addWidget(self._yolo_status)

        # Detections table
        self._det_table = QTableWidget(0, 3)
        self._det_table.setHorizontalHeaderLabels(["Class", "Confidence", "BBox"])
        self._det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._det_table.setMaximumHeight(180)
        self._det_table.hide()
        outer.addWidget(self._det_table)

        # ── Manual ECG Features ───────────────────────────────────────────────
        self._manual_grp = QGroupBox("Manual ECG Features  (used when no image is uploaded)")
        ml = QVBoxLayout(self._manual_grp)
        note = QLabel("Toggle abnormalities present on the ECG:")
        note.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:11px;")
        ml.addWidget(note)

        self._ecg_checks: dict[str, QCheckBox] = {}
        ecg_labels = [
            ("Q Wave",            "Pathological Q Wave"),
            ("St Elevation",      "ST Elevation"),
            ("St Depression",     "ST Depression"),
            ("Tinversion",        "T Inversion"),
            ("LVH",               "LVH (Left Ventricular Hypertrophy)"),
            ("Poor R Progression","Poor R Progression"),
        ]
        for key, lbl in ecg_labels:
            cb = QCheckBox(lbl)
            self._ecg_checks[key] = cb
            ml.addWidget(cb)

        outer.addWidget(self._manual_grp)
        outer.addStretch()

    # ── slots ────────────────────────────────────────────────────────────────
    def _set_image(self, path: str):
        self._image_path = path
        self._img_path_label.setText(Path(path).name)
        px = QPixmap(path)
        if not px.isNull():
            self._preview.setPixmap(
                px.scaled(300, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._preview.show()
        self._drop_zone.setText(f"✔  {Path(path).name}")
        self._run_yolo_btn.setEnabled(True)
        self._yolo_detections = []
        self._det_table.hide()
        self._manual_grp.setEnabled(False)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ECG Image", str(APP_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if path:
            self._set_image(path)

    def _browse_yolo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Weights", str(APP_DIR), "Weights (*.pt *.pth)")
        if path:
            self._yolo_path_edit.setText(path)

    def _run_yolo(self):
        self._run_yolo_btn.setEnabled(False)
        self._yolo_status.setText("⏳  Running YOLO inference…")
        self._worker = YOLOWorker(self._image_path, self._yolo_path_edit.text())
        self._worker.results_ready.connect(self._on_yolo_done)
        self._worker.error.connect(self._on_yolo_error)
        self._worker.start()

    def _on_yolo_done(self, dets: list):
        self._yolo_detections = dets
        self._yolo_status.setText(
            f"✔  Found {len(dets)} detection(s)" if dets else "⚠  No abnormalities detected")
        self._run_yolo_btn.setEnabled(True)
        self._populate_det_table(dets)
        self.yolo_done.emit(dets)

    def _on_yolo_error(self, msg: str):
        self._yolo_status.setText(f"✖  YOLO error: {msg}")
        self._run_yolo_btn.setEnabled(True)

    def _populate_det_table(self, dets: list):
        self._det_table.setRowCount(len(dets))
        for r, d in enumerate(dets):
            self._det_table.setItem(r, 0, QTableWidgetItem(d["class"]))
            self._det_table.setItem(r, 1, QTableWidgetItem(f"{d['confidence']:.4f}"))
            bbox = d.get("bbox", [])
            self._det_table.setItem(r, 2, QTableWidgetItem(str([round(v) for v in bbox])))
        self._det_table.show()

    def get_yolo_detections(self) -> list[dict]:
        """Returns YOLO detections OR synthetic ones from manual checkboxes."""
        if self._yolo_detections:
            return self._yolo_detections
        # Build synthetic from manual checkboxes
        dets = []
        for key, cb in self._ecg_checks.items():
            if cb.isChecked():
                dets.append({"class": key, "confidence": 1.0})
        return dets

    def reset(self):
        self._image_path     = ""
        self._yolo_detections = []
        self._img_path_label.setText("No image selected")
        self._preview.hide()
        self._drop_zone.setText("🖼  Drag & Drop ECG Image Here\nor click Browse")
        self._run_yolo_btn.setEnabled(False)
        self._det_table.hide()
        self._yolo_status.setText("")
        self._manual_grp.setEnabled(True)
        for cb in self._ecg_checks.values():
            cb.setChecked(False)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class ResultsPage(QScrollArea):
    new_patient_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.setWidget(container)
        outer = QVBoxLayout(container)
        outer.setSpacing(18)
        outer.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📊  Prediction Results")
        title.setStyleSheet(f"font-size:20px; font-weight:700; color:{PALETTE['text']};")
        outer.addWidget(title)
        outer.addWidget(separator())

        # ── Gauge ─────────────────────────────────────────────────────────────
        self._gauge = GaugeWidget()
        outer.addWidget(self._gauge, 0, Qt.AlignCenter)

        # ── Risk badge ────────────────────────────────────────────────────────
        self._risk_badge = QLabel("—")
        self._risk_badge.setAlignment(Qt.AlignCenter)
        self._risk_badge.setFixedHeight(54)
        self._risk_badge.setStyleSheet(f"""
            font-size: 22px; font-weight: 800;
            background: {PALETTE['surface2']};
            border-radius: 27px;
            color: {PALETTE['text_dim']};
        """)
        outer.addWidget(self._risk_badge)

        # Summary cards row
        card_row = QHBoxLayout()
        self._prob_card  = self._stat_card("Probability", "—")
        self._pred_card  = self._stat_card("Diagnosis", "—")
        card_row.addWidget(self._prob_card[0])
        card_row.addWidget(self._pred_card[0])
        outer.addLayout(card_row)

        # ── ECG Findings table ────────────────────────────────────────────────
        ecg_grp = QGroupBox("ECG Findings (from YOLO / Manual)")
        ecg_l   = QVBoxLayout(ecg_grp)
        self._ecg_table = QTableWidget(6, 3)
        self._ecg_table.setHorizontalHeaderLabels(["Feature", "Detected", "Confidence"])
        self._ecg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ecg_table.verticalHeader().setVisible(False)
        self._ecg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ecg_l.addWidget(self._ecg_table)
        outer.addWidget(ecg_grp)

        # ── New patient button ─────────────────────────────────────────────────
        btn_new = QPushButton("🔄  New Patient")
        btn_new.setObjectName("primary")
        btn_new.clicked.connect(self.new_patient_requested)
        outer.addWidget(btn_new, 0, Qt.AlignLeft)
        outer.addStretch()

    # ── stat card helper ──────────────────────────────────────────────────────
    def _stat_card(self, heading: str, value: str):
        card  = QGroupBox(heading)
        cl    = QVBoxLayout(card)
        label = QLabel(value)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"font-size:26px; font-weight:800; color:{PALETTE['text']};")
        cl.addWidget(label)
        return card, label

    # ── populate ─────────────────────────────────────────────────────────────
    def show_result(self, result: dict):
        prob  = result["probability"]
        risk  = result["risk_level"]
        pred  = result["prediction"]

        # Colours
        if risk == "LOW":
            col = PALETTE["low"]
        elif risk == "MODERATE":
            col = PALETTE["moderate"]
        else:
            col = PALETTE["high"]

        self._gauge.set_value(prob)

        self._risk_badge.setText(risk)
        self._risk_badge.setStyleSheet(f"""
            font-size: 22px; font-weight: 800;
            background: {col}22;
            border: 2px solid {col};
            border-radius: 27px;
            color: {col};
        """)

        self._prob_card[1].setText(f"{prob * 100:.1f}%")
        self._prob_card[1].setStyleSheet(f"font-size:26px; font-weight:800; color:{col};")
        self._pred_card[1].setText("CAD" if pred else "Normal")
        self._pred_card[1].setStyleSheet(
            f"font-size:26px; font-weight:800; color:{col if pred else PALETTE['low']};")

        # ECG table
        ecg    = result.get("ecg_detections", {})
        conf_m = result.get("confidence_metadata", {})
        for row, (feat, val) in enumerate(ecg.items()):
            conf = conf_m.get(feat, 0.0)
            self._ecg_table.setItem(row, 0, QTableWidgetItem(feat))
            status_item = QTableWidgetItem("✔ Detected" if val else "✘ Not detected")
            status_item.setForeground(
                QColor(PALETTE["high"]) if val else QColor(PALETTE["text_dim"]))
            self._ecg_table.setItem(row, 1, status_item)
            self._ecg_table.setItem(row, 2, QTableWidgetItem(f"{conf:.4f}" if conf else "—"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🫀  Cardionyx — CAD Risk Prediction")
        self.setMinimumSize(1050, 720)
        self._pred_worker: PredictionWorker | None = None

        # ── Central layout ────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(f"background: {PALETTE['surface']}; border-right: 1px solid {PALETTE['border']};")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        # Logo area
        logo_area = QWidget()
        logo_area.setFixedHeight(72)
        logo_area.setStyleSheet(f"background: {PALETTE['surface']};")
        ll = QVBoxLayout(logo_area)
        logo_lbl = QLabel("🫀 Cardionyx")
        logo_lbl.setStyleSheet(f"font-size:16px; font-weight:800; color:{PALETTE['accent']}; padding: 0 14px;")
        ll.addWidget(logo_lbl)
        version_lbl = QLabel("CAD Risk Predictor")
        version_lbl.setStyleSheet(f"font-size:10px; color:{PALETTE['text_dim']}; padding: 0 14px;")
        ll.addWidget(version_lbl)
        sb_layout.addWidget(logo_area)
        sb_layout.addWidget(separator())
        sb_layout.addSpacing(8)

        self._nav_btns: list[SidebarButton] = []
        nav_items = [
            ("🧑‍⚕️", "Patient Info"),
            ("📈", "ECG / Imaging"),
            ("📊", "Results"),
        ]
        for icon, label in nav_items:
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda _, b=btn: self._nav_to(b))
            self._nav_btns.append(btn)
            sb_layout.addWidget(btn)

        sb_layout.addStretch()

        # Model path row at bottom of sidebar
        sb_layout.addWidget(separator())
        model_widget = QWidget()
        model_widget.setStyleSheet(f"background: {PALETTE['surface']};")
        ml = QVBoxLayout(model_widget)
        ml.setContentsMargins(10, 8, 10, 10)
        ml.addWidget(QLabel("XGBoost Model:"))
        self._xgb_path = QLineEdit(str(APP_DIR / "xgb_cad_model.joblib"))
        self._xgb_path.setToolTip("Path to xgb_cad_model.joblib")
        ml.addWidget(self._xgb_path)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("YOLO conf:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setDecimals(2)
        conf_row.addWidget(self._conf_spin)
        ml.addLayout(conf_row)
        sb_layout.addWidget(model_widget)

        root.addWidget(sidebar)

        # ── Content area ──────────────────────────────────────────────────────
        content_area = QWidget()
        cl = QVBoxLayout(content_area)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self._stack = QStackedWidget()
        self._patient_page = PatientPage()
        self._ecg_page     = ECGPage()
        self._results_page = ResultsPage()

        self._stack.addWidget(self._patient_page)   # index 0
        self._stack.addWidget(self._ecg_page)        # index 1
        self._stack.addWidget(self._results_page)    # index 2

        self._results_page.new_patient_requested.connect(self._reset_all)

        cl.addWidget(self._stack, 1)

        # ── Bottom toolbar ────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet(f"background: {PALETTE['surface']}; border-top: 1px solid {PALETTE['border']};")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(16, 0, 16, 0)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:12px;")
        tl.addWidget(self._status_label, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedWidth(160)
        self._progress.hide()
        tl.addWidget(self._progress)

        self._run_btn = QPushButton("⚡  Run Prediction")
        self._run_btn.setObjectName("primary")
        self._run_btn.setFixedHeight(40)
        self._run_btn.clicked.connect(self._run_prediction)
        tl.addWidget(self._run_btn)

        cl.addWidget(toolbar)
        root.addWidget(content_area, 1)

        # default nav
        self._nav_to(self._nav_btns[0])

    # ── navigation ────────────────────────────────────────────────────────────
    def _nav_to(self, btn: SidebarButton):
        for b in self._nav_btns:
            b.setChecked(False)
        btn.setChecked(True)
        idx = self._nav_btns.index(btn)
        self._stack.setCurrentIndex(idx)

    # ── prediction ────────────────────────────────────────────────────────────
    def _run_prediction(self):
        model_path = self._xgb_path.text().strip()
        if not Path(model_path).exists():
            QMessageBox.warning(self, "Model Not Found",
                                f"XGBoost model not found:\n{model_path}")
            return

        patient_data    = self._patient_page.get_patient_data()
        yolo_detections = self._ecg_page.get_yolo_detections()
        conf_threshold  = self._conf_spin.value()

        self._run_btn.setEnabled(False)
        self._progress.show()
        self._status_label.setText("Running prediction…")

        self._pred_worker = PredictionWorker(
            yolo_detections, patient_data, model_path, conf_threshold)
        self._pred_worker.prediction_ready.connect(self._on_prediction_done)
        self._pred_worker.error.connect(self._on_prediction_error)
        self._pred_worker.start()

    def _on_prediction_done(self, result: dict):
        self._progress.hide()
        self._run_btn.setEnabled(True)
        self._status_label.setText(
            f"Prediction complete — {result['risk_level']} risk  ({result['probability']*100:.1f}%)")
        self._results_page.show_result(result)
        self._nav_to(self._nav_btns[2])   # switch to results page

    def _on_prediction_error(self, msg: str):
        self._progress.hide()
        self._run_btn.setEnabled(True)
        self._status_label.setText("Prediction failed.")
        QMessageBox.critical(self, "Prediction Error", msg)

    # ── reset ─────────────────────────────────────────────────────────────────
    def _reset_all(self):
        self._ecg_page.reset()
        self._status_label.setText("Ready")
        self._nav_to(self._nav_btns[0])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)

    # Dark palette for native widgets
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(PALETTE['bg']))
    palette.setColor(QPalette.WindowText,      QColor(PALETTE['text']))
    palette.setColor(QPalette.Base,            QColor(PALETTE['surface']))
    palette.setColor(QPalette.AlternateBase,   QColor(PALETTE['surface2']))
    palette.setColor(QPalette.Text,            QColor(PALETTE['text']))
    palette.setColor(QPalette.ButtonText,      QColor(PALETTE['text']))
    palette.setColor(QPalette.Highlight,       QColor(PALETTE['accent']))
    palette.setColor(QPalette.HighlightedText, QColor(PALETTE['bg']))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
