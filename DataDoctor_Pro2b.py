"""
╔══════════════════════════════════════════════════════════════════╗
║                    DataDoctor Pro  v4.0                          ║
║            Pipeline de nettoyage de données avancé              ║
╚══════════════════════════════════════════════════════════════════╝

Améliorations v4.0 vs v3.x :
  - DataDoctorConfig → dataclass validée, sérialisable YAML/JSON
  - Détection d'encodage via chardet (+ sniff séparateur CSV)
  - Pipeline par étapes indépendantes avec gestion d'erreur isolée
  - DataQualityScore : score global 0-100 multi-dimensions
  - Smart text analysis : emails, URLs, téléphones, codes postaux
  - Rapport HTML self-contained (sans dépendance ydata-profiling)
  - Context manager + reproducible transformation plan (JSON)
  - Colonnes quasi-constantes (seuil configurable)
  - Détection de colonnes ID / texte libre par cardinalité
  - Aucun bare except, deprecations pandas 2.x corrigées
"""

from __future__ import annotations

import json
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import HTML, Markdown, display
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

# ─── Tentative d'import des dépendances optionnelles ──────────────────────────
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from category_encoders import TargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False

try:
    from ydata_profiling import ProfileReport
    HAS_YDATA = True
except ImportError:
    HAS_YDATA = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ImputeMethod  = Literal["auto", "knn", "iterative", "simple", "none"]
OutlierMethod = Literal["iqr", "zscore", "isolation_forest", "dbscan", "none"]
TextCase      = Literal["lower", "upper", "title", "none"]
OutputFormat  = Literal["csv", "xlsx", "parquet", "pickle", "json"]


@dataclass
class DataDoctorConfig:
    """Configuration complète et validée du DataDoctor."""

    # ── Fichiers ──────────────────────────────────────────────────────────────
    source_file:    str            = "data.csv"
    target_file:    Optional[str]  = None
    log_file:       Optional[str]  = None
    report_file:    Optional[str]  = None
    output_format:  OutputFormat   = "csv"

    # ── Nettoyage général ─────────────────────────────────────────────────────
    remove_duplicates:      bool  = True
    max_missing_percent:    float = 30.0    # seuil de suppression de colonne
    drop_constant_cols:     bool  = True
    quasi_constant_thresh:  float = 0.995   # colonne supprimée si une valeur > seuil
    detect_encoding:        bool  = True

    # ── Types ─────────────────────────────────────────────────────────────────
    convert_dtypes:   bool = True
    infer_datetime:   bool = True
    infer_numeric:    bool = True
    fix_boolean:      bool = True

    # ── Imputation ────────────────────────────────────────────────────────────
    impute_method:          ImputeMethod = "auto"
    knn_neighbors:          int          = 5
    add_missing_indicators: bool         = True

    # ── Outliers ──────────────────────────────────────────────────────────────
    outlier_method:    OutlierMethod = "iqr"
    outlier_threshold: float         = 1.5
    zscore_threshold:  float         = 3.0
    contamination:     float         = 0.01

    # ── Texte & catégories ────────────────────────────────────────────────────
    normalize_text:  bool     = True
    text_case:       TextCase = "title"
    max_categories:  int      = 20

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_steps: List[str] = field(default_factory=lambda: [
        "load_data",
        "remove_empty_columns",
        "convert_datatypes",
        "remove_duplicates",
        "handle_missing_columns",
        "impute_missing_values",
        "detect_and_fix_outliers",
        "normalize_text_data",
        "encode_categorical_data",
        "visualize_missing",
        "generate_profile",
        "save_outputs",
    ])

    # ── Performance & UI ──────────────────────────────────────────────────────
    verbose:               bool = True
    interactive:           bool = True
    generate_profile:      bool = True
    sample_for_analysis:   bool = True
    sample_size:           int  = 10_000
    n_jobs:                int  = -1
    progress_bar:          bool = True

    # ── Validation ────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if not 0 < self.max_missing_percent <= 100:
            raise ValueError("max_missing_percent doit être dans ]0, 100]")
        if not 0 < self.quasi_constant_thresh <= 1:
            raise ValueError("quasi_constant_thresh doit être dans ]0, 1]")
        if self.contamination <= 0 or self.contamination >= 0.5:
            raise ValueError("contamination doit être dans ]0, 0.5[")
        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors doit être ≥ 1")

    def set_source(self, filename: str) -> "DataDoctorConfig":
        """Dérive automatiquement les noms de fichiers de sortie."""
        self.source_file = filename
        base = os.path.splitext(filename)[0]
        ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
        ext  = "xlsx" if self.output_format in ("xls", "xlsx", "excel") else self.output_format
        self.target_file = self.target_file or f"{base}-doctored.{ext}"
        self.log_file    = self.log_file    or f"{base}-cleanlog-{ts}.txt"
        self.report_file = self.report_file or f"{base}-report-{ts}.html"
        return self

    # ── Sérialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataDoctorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "DataDoctorConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str) -> "DataDoctorConfig":
        if not HAS_YAML:
            raise ImportError("pip install pyyaml")
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE DE QUALITÉ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityScore:
    """Score de qualité global et par dimension, calculé avant/après."""

    completeness:  float = 0.0   # % de cellules non-nulles
    uniqueness:    float = 0.0   # 1 - taux de doublons
    consistency:   float = 0.0   # types cohérents, booléens propres
    validity:      float = 0.0   # outliers absents
    overall:       float = 0.0

    def compute(self, df: pd.DataFrame) -> "DataQualityScore":
        n_cells = df.size or 1

        # Complétude
        self.completeness = round((1 - df.isna().sum().sum() / n_cells) * 100, 1)

        # Unicité
        dup_rate = df.duplicated().mean()
        self.uniqueness = round((1 - dup_rate) * 100, 1)

        # Cohérence : pénalité si colonnes object contenant des nombres en string
        mixed = 0
        for col in df.select_dtypes("object").columns:
            sample = df[col].dropna().head(200).astype(str)
            numeric_ratio = sample.str.match(r"^[-+]?\d*\.?\d+$").mean()
            if numeric_ratio > 0.5:
                mixed += 1
        n_obj = max(df.select_dtypes("object").shape[1], 1)
        self.consistency = round((1 - mixed / n_obj) * 100, 1)

        # Validité : taux de valeurs dans l'IQR ×2 pour toutes colonnes numériques
        outlier_ratio_sum = 0.0
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n_out = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
            outlier_ratio_sum += n_out / max(len(df), 1)
        n_num = max(len(num_cols), 1)
        self.validity = round((1 - outlier_ratio_sum / n_num) * 100, 1)

        self.overall = round(
            0.35 * self.completeness
            + 0.25 * self.uniqueness
            + 0.20 * self.consistency
            + 0.20 * self.validity,
            1,
        )
        return self

    def _bar(self, pct: float, width: int = 20) -> str:
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def __str__(self) -> str:
        lines = ["", "  ┌─ DataQualityScore ─────────────────────────────────────┐"]
        for label, val in [
            ("Complétude  ", self.completeness),
            ("Unicité     ", self.uniqueness),
            ("Cohérence   ", self.consistency),
            ("Validité    ", self.validity),
        ]:
            color = "🟢" if val >= 85 else "🟡" if val >= 60 else "🔴"
            lines.append(f"  │  {label} {self._bar(val)} {val:5.1f}%  {color}  │")
        lines.append(f"  │  ── Score global ──────────────────── {self.overall:5.1f}%       │")
        lines.append("  └──────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTEUR DE PATTERNS TEXTE
# ══════════════════════════════════════════════════════════════════════════════

_TEXT_PATTERNS: Dict[str, re.Pattern] = {
    "email":        re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$"),
    "url":          re.compile(r"^https?://\S+$"),
    "phone_fr":     re.compile(r"^(?:\+33|0)\s*[1-9](?:[\s.-]?\d{2}){4}$"),
    "postal_fr":    re.compile(r"^\d{5}$"),
    "iban":         re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}$"),
    "siren":        re.compile(r"^\d{9}$"),
    "isbn":         re.compile(r"^(?:97[89])?\d{9}[\dX]$"),
}


def detect_text_pattern(series: pd.Series, threshold: float = 0.8) -> Optional[str]:
    """Retourne le pattern dominant d'une colonne texte, ou None."""
    sample = series.dropna().astype(str).head(500)
    if len(sample) == 0:
        return None
    for name, pat in _TEXT_PATTERNS.items():
        if sample.str.match(pat).mean() >= threshold:
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  RÉSULTAT D'UNE ÉTAPE DE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    name:    str
    success: bool
    message: str
    elapsed: float = 0.0
    error:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOCTOR PRO
# ══════════════════════════════════════════════════════════════════════════════

class DataDoctor:
    """
    Pipeline de nettoyage de données professionnel.

    Utilisation basique :
        config = DataDoctorConfig(source_file="data.csv")
        doc = DataDoctor(config)
        df_clean = doc.run()

    Context manager :
        with DataDoctor(config) as doc:
            df_clean = doc.run()
    """

    VERSION = "4.0"

    def __init__(self, config: DataDoctorConfig) -> None:
        self.config = config
        self.log_entries:   List[str]       = []
        self.step_results:  List[StepResult] = []
        self.start_time     = datetime.now()
        self.df:            Optional[pd.DataFrame] = None
        self.df_clean:      Optional[pd.DataFrame] = None
        self.df_sample:     Optional[pd.DataFrame] = None
        self.original_shape: Optional[Tuple[int, int]] = None
        self.data_profile:  Dict[str, Any] = {}
        self.col_patterns:  Dict[str, str] = {}   # résultats smart_text_analysis
        self.score_before:  Optional[DataQualityScore] = None
        self.score_after:   Optional[DataQualityScore] = None
        self.transformations: Dict[str, Any] = {
            "dropped_columns":   [],
            "type_conversions":  {},
            "imputation":        {},
            "outliers":          {},
            "text_normalization": {},
            "duplicates":        0,
            "encoding_fixes":    {},
            "col_patterns":      {},
        }

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "DataDoctor":
        return self

    def __exit__(self, *_) -> None:
        pass

    def __repr__(self) -> str:
        shape = self.df_clean.shape if self.df_clean is not None else "non chargé"
        return f"DataDoctor(source={self.config.source_file!r}, shape={shape})"

    # ── Utilitaires internes ──────────────────────────────────────────────────

    def _sanitize_for_plotting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les dtypes pandas nullable en types numpy natifs pour éviter les erreurs de sérialisation Plotly/Seaborn."""
        df_plot = df.copy()
        # Mapping SANS espaces pour correspondre exactement à str(dtype)
        nullable_to_numpy = {
            "Float64": "float64", "Int64": "int64", "Int8": "int8", "Int16": "int16",
            "Int32": "int32", "UInt8": "uint8", "UInt16": "uint16", "UInt32": "uint32", "UInt64": "uint64",
            "boolean": "bool", "string": "object"
        }
        for col in df_plot.columns:
            dtype_str = str(df_plot[col].dtype)
            if dtype_str in nullable_to_numpy:
                # Conversion locale juste pour l'affichage, sans toucher au df original du pipeline
                df_plot[col] = df_plot[col].astype(nullable_to_numpy[dtype_str])
        return df_plot

    def _progress(self, iterable, desc: Optional[str] = None):
        if self.config.progress_bar:
            return tqdm(iterable, desc=desc, leave=False)
        return iterable

    def log_action(self, action: str, details: Any) -> None:
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {action}: {details}"
        self.log_entries.append(entry)
        if self.config.verbose:
            if self.config.interactive:
                display(HTML(f"<b>{action}:</b> {details}"))
            else:
                print(entry)

    def _run_step(self, method_name: str) -> StepResult:
        """Exécute une étape du pipeline de façon isolée."""
        t0 = datetime.now()
        try:
            getattr(self, method_name)()
            elapsed = (datetime.now() - t0).total_seconds()
            result  = StepResult(name=method_name, success=True,
                                 message="OK", elapsed=elapsed)
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            msg     = f"⚠️ Étape '{method_name}' échouée : {exc}"
            self.log_action("ERREUR ÉTAPE", msg)
            result  = StepResult(name=method_name, success=False,
                                 message="ERREUR", elapsed=elapsed, error=str(exc))
        self.step_results.append(result)
        return result

    # ── Chargement ────────────────────────────────────────────────────────────

    def load_data(self) -> "DataDoctor":
        """Charge les données avec détection automatique du format et de l'encodage."""
        self.log_action("Début du traitement", f"Fichier source: {self.config.source_file}")
        path = Path(self.config.source_file)
        ext  = path.suffix.lower()

        if ext == ".csv":
            self.df = self._load_csv(path)
        elif ext in (".xlsx", ".xls"):
            self.df = pd.read_excel(path)
        elif ext == ".parquet":
            self.df = pd.read_parquet(path)
        elif ext == ".json":
            self.df = pd.read_json(path)
        elif ext in (".pkl", ".pickle"):
            self.df = pd.read_pickle(path)
        else:
            self.df = self._load_csv(path)  # tentative CSV par défaut

        self.original_shape = self.df.shape
        self.log_action(
            "Chargement réussi",
            f"{self.original_shape[0]:,} lignes × {self.original_shape[1]:,} colonnes"
        )
        self.df_clean = self.df.copy()

        # Score initial
        self.score_before = DataQualityScore().compute(self.df)
        self.log_action("Score qualité initial", str(self.score_before))

        self._analyze_initial_data()

        if self.config.interactive:
            display(HTML("<h3>📋 Aperçu initial des données :</h3>"))
            display(self.df_clean.head(3))
        return self

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Charge un CSV avec détection de l'encodage et du séparateur."""
        encoding = self._detect_encoding(path)

        # Détection du séparateur via csv.Sniffer
        sep = ","
        try:
            import csv
            with open(path, encoding=encoding, errors="replace") as f:
                dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;\t|")
                sep = dialect.delimiter
        except Exception:
            pass

        df = pd.read_csv(path, encoding=encoding, sep=sep,
                         on_bad_lines="warn", low_memory=False)
        self.log_action("CSV chargé", f"encodage={encoding}, séparateur={sep!r}")
        return df

    def _detect_encoding(self, path: Path) -> str:
        """Détecte l'encodage du fichier via chardet ou heuristique."""
        if HAS_CHARDET:
            raw = path.read_bytes()[:65_536]
            result = chardet.detect(raw)
            enc = result.get("encoding") or "utf-8"
            confidence = result.get("confidence", 0)
            self.log_action("Encodage détecté", f"{enc} (confiance: {confidence:.0%})")
            return enc

        # Fallback : boucle heuristique
        for enc in ("utf-8", "latin1", "ISO-8859-1", "cp1252"):
            try:
                path.read_text(encoding=enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "utf-8"

    # ── Analyse initiale ──────────────────────────────────────────────────────

    def _analyze_initial_data(self) -> None:
        assert self.df is not None
        self.data_profile["initial"] = {
            "shape":         self.df.shape,
            "dtypes":        self.df.dtypes.astype(str).to_dict(),
            "missing":       self.df.isna().sum().to_dict(),
            "missing_pct":   (self.df.isna().mean() * 100).round(1).to_dict(),
            "unique":        {c: self.df[c].nunique() for c in self.df.columns},
            "memory_mb":     self.df.memory_usage(deep=True).sum() / 1024**2,
            "cardinality":   self._cardinality_analysis(),
        }

        if self.config.sample_for_analysis and len(self.df) > self.config.sample_size:
            self.df_sample = self.df.sample(self.config.sample_size, random_state=42)
            self.log_action("Échantillonnage", f"{self.config.sample_size:,} lignes pour l'analyse")
        else:
            self.df_sample = self.df

        # Smart text pattern analysis
        self._smart_text_analysis()

        if self.config.interactive:
            self._show_initial_visualizations()

    def _cardinality_analysis(self) -> Dict[str, str]:
        """Classifie chaque colonne : id | category | freetext | other."""
        assert self.df is not None
        result = {}
        n = max(len(self.df), 1)
        for col in self.df.columns:
            uniq_ratio = self.df[col].nunique() / n
            if uniq_ratio > 0.95:
                result[col] = "id_or_freetext"
            elif self.df[col].nunique() <= self.config.max_categories:
                result[col] = "category"
            else:
                result[col] = "high_cardinality"
        return result

    def _smart_text_analysis(self) -> None:
        """Détecte les patterns sémantiques dans les colonnes texte."""
        assert self.df_clean is not None
        for col in self.df_clean.select_dtypes("object").columns:
            pattern = detect_text_pattern(self.df_clean[col])
            if pattern:
                self.col_patterns[col] = pattern
                self.transformations["col_patterns"][col] = pattern

        if self.col_patterns:
            self.log_action(
                "Patterns texte détectés",
                ", ".join(f"{c}={p}" for c, p in self.col_patterns.items()),
            )

    def _show_initial_visualizations(self) -> None:
        assert self.df is not None
        # Conversion préalable pour éviter le crash Plotly sur Float64DType / Int64DType
        df_viz = self._sanitize_for_plotting(self.df)

        # 1. Types : .astype(str) est CRUCIAL car .dtypes contient des objets dtype non sérialisables
        dtype_counts = df_viz.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ["Type", "Count"]
        fig = px.bar(dtype_counts, x="Type", y="Count",
                     title="Distribution des Types de Données", color="Type", text_auto=True)
        fig.show()

        # 2. Valeurs manquantes
        missing = df_viz.isna().mean().sort_values(ascending=False)
        missing = missing[missing > 0].reset_index()
        if not missing.empty:
            missing.columns = ["Colonne", "% Manquant"]
            missing["% Manquant"] = (missing["% Manquant"] * 100).round(1)
            fig = px.bar(missing, x="Colonne", y="% Manquant",
                         title="Pourcentage de Valeurs Manquantes par Colonne", color="% Manquant")
            fig.update_yaxes(range=[0, 100])
            fig.show()

    # ── Étapes du pipeline ────────────────────────────────────────────────────

    def remove_empty_columns(self) -> "DataDoctor":
        """Supprime les colonnes vides, constantes et quasi-constantes."""
        assert self.df_clean is not None
        initial_cols = self.df_clean.columns.tolist()

        # Colonnes totalement vides
        self.df_clean.dropna(axis=1, how="all", inplace=True)
        empty_removed = set(initial_cols) - set(self.df_clean.columns)

        constant_removed: List[str] = []
        quasi_removed:    List[str] = []

        if self.config.drop_constant_cols:
            for col in self.df_clean.columns.tolist():
                top_freq = self.df_clean[col].value_counts(normalize=True, dropna=True)
                if top_freq.empty:
                    continue
                if top_freq.iloc[0] >= self.config.quasi_constant_thresh:
                    if self.df_clean[col].nunique() <= 1:
                        constant_removed.append(col)
                    else:
                        quasi_removed.append(col)
            all_drop = constant_removed + quasi_removed
            if all_drop:
                self.df_clean.drop(columns=all_drop, inplace=True)

        removed = list(empty_removed) + constant_removed + quasi_removed
        if removed:
            self.transformations["dropped_columns"].extend(removed)
            self.log_action(
                "Colonnes supprimées",
                f"{len(empty_removed)} vides, {len(constant_removed)} constantes, "
                f"{len(quasi_removed)} quasi-constantes",
            )
            if self.config.interactive:
                reasons = (
                    ["Vide"] * len(empty_removed)
                    + ["Constante"] * len(constant_removed)
                    + [f"Quasi-constante (>{self.config.quasi_constant_thresh:.0%})"]
                    * len(quasi_removed)
                )
                display(pd.DataFrame({"Colonne": removed, "Raison": reasons}))
        return self

    def convert_datatypes(self) -> "DataDoctor":
        """Convertit les types avec détection améliorée et sans déprecation pandas 2.x."""
        if not self.config.convert_dtypes:
            return self
        assert self.df_clean is not None

        type_changes: Dict[str, str] = {}

        for col in self._progress(self.df_clean.columns, "Conversion des types"):
            original_type = str(self.df_clean[col].dtype)

            # ── Numérique ────────────────────────────────────────────────────
            if self.config.infer_numeric and self.df_clean[col].dtype == "object":
                try:
                    sample = self.df_clean[col].dropna().head(200).astype(str)
                    pattern = r"^[-+]?[0-9]*[.,]?[0-9]+([eE][-+]?[0-9]+)?$"
                    if sample.str.replace(",", ".", regex=False).str.match(pattern).mean() > 0.9:
                        converted = pd.to_numeric(
                            self.df_clean[col].astype(str).str.replace(",", ".", regex=False),
                            errors="coerce",
                        )
                        if not converted.isna().all():
                            self.df_clean[col] = converted
                            type_changes[col] = f"{original_type} → {self.df_clean[col].dtype}"
                            continue
                except Exception:
                    pass

            # ── Datetime (sans infer_datetime_format, déprécié pandas 2.0) ──
            if self.config.infer_datetime and self.df_clean[col].dtype == "object":
                try:
                    sample = self.df_clean[col].dropna().head(200).astype(str)
                    date_patterns = [
                        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
                        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
                        r"\d{1,2}\s+[a-zA-Z]+\s+\d{4}",
                        r"\d{1,2}[./]\d{1,2}[./]\d{2,4}",
                    ]
                    if any(sample.str.match(p).mean() > 0.8 for p in date_patterns):
                        converted_dt = pd.to_datetime(
                            self.df_clean[col], errors="coerce", dayfirst=True
                        )
                        if not converted_dt.isna().all():
                            self.df_clean[col] = converted_dt
                            type_changes[col] = f"{original_type} → datetime"
                            continue
                except Exception:
                    pass

            # ── Booléen ──────────────────────────────────────────────────────
            if self.config.fix_boolean and self.df_clean[col].dtype == "object":
                try:
                    unique_vals = {
                        str(v).lower().strip()
                        for v in self.df_clean[col].dropna().unique()
                    }
                    bool_pairs = [
                        {"true", "false"}, {"t", "f"}, {"yes", "no"}, {"y", "n"},
                        {"oui", "non"}, {"o", "n"}, {"1", "0"}, {"vrai", "faux"},
                    ]
                    for pair in bool_pairs:
                        if unique_vals.issubset(pair):
                            positives = {"true", "t", "yes", "y", "1", "oui", "o", "vrai"}
                            self.df_clean[col] = (
                                self.df_clean[col].astype(str).str.lower().str.strip()
                                .map(lambda v: v in positives)
                            )
                            type_changes[col] = f"{original_type} → bool"
                            break
                except Exception:
                    pass

        # Optimisation mémoire finale
        try:
            self.df_clean = self.df_clean.convert_dtypes()
        except Exception:
            pass

        if type_changes:
            self.transformations["type_conversions"].update(type_changes)
            self.log_action(
                "Conversions de type",
                pd.DataFrame.from_dict(type_changes, orient="index", columns=["Conversion"]),
            )
        return self

    def remove_duplicates(self) -> "DataDoctor":
        assert self.df_clean is not None
        if not self.config.remove_duplicates:
            return self
        n_dup = int(self.df_clean.duplicated().sum())
        if n_dup > 0:
            before = len(self.df_clean)
            self.df_clean.drop_duplicates(inplace=True)
            self.transformations["duplicates"] = n_dup
            self.log_action(
                "Suppression des doublons",
                f"{n_dup:,} lignes dupliquées supprimées (de {before:,} à {len(self.df_clean):,})",
            )
        return self

    def handle_missing_columns(self) -> "DataDoctor":
        assert self.df_clean is not None
        missing_pct = self.df_clean.isna().mean() * 100
        cols_to_drop = missing_pct[missing_pct > self.config.max_missing_percent].index.tolist()
        if cols_to_drop:
            self.df_clean.drop(columns=cols_to_drop, inplace=True)
            self.transformations["dropped_columns"].extend(cols_to_drop)
            self.log_action(
                "Colonnes supprimées (trop de NA)",
                f"{len(cols_to_drop)} colonnes avec >{self.config.max_missing_percent}% NA",
            )
            if self.config.interactive:
                display(pd.DataFrame({
                    "Colonne":     cols_to_drop,
                    "% Manquant":  [round(float(missing_pct[c]), 1) for c in cols_to_drop],
                }))
        return self

    def impute_missing_values(self) -> "DataDoctor":
        """Imputation multi-stratégie avec indicateurs de valeurs manquantes."""
        if self.config.impute_method == "none":
            return self
        assert self.df_clean is not None

        numeric_cols    = self.df_clean.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df_clean.select_dtypes(exclude=np.number).columns.tolist()
        datetime_cols   = self.df_clean.select_dtypes(include=["datetime"]).columns.tolist()

        # Indicateurs avant imputation
        if self.config.add_missing_indicators:
            for col in self.df_clean.columns:
                if self.df_clean[col].isna().any():
                    self.df_clean[f"{col}_was_missing"] = self.df_clean[col].isna().astype("int8")

        # Catégorielle → mode
        for col in categorical_cols:
            if col in datetime_cols:
                continue
            if self.df_clean[col].isna().any():
                mode_series = self.df_clean[col].mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
                    self.df_clean[col] = self.df_clean[col].fillna(mode_val)
                    self.transformations["imputation"][col] = f"Mode: {mode_val}"

        # Datetime → médiane
        for col in datetime_cols:
            if self.df_clean[col].isna().any():
                median_date = self.df_clean[col].dropna().quantile(0.5)
                self.df_clean[col] = self.df_clean[col].fillna(median_date)
                self.transformations["imputation"][col] = f"Médiane date"

        # Numérique
        missing_numeric = [c for c in numeric_cols if self.df_clean[c].isna().any()]
        if not missing_numeric:
            return self

        if self.config.impute_method == "auto":
            for col in missing_numeric:
                skew = self.df_clean[col].skew()
                if abs(skew) > 1:
                    val = self.df_clean[col].median()
                    method_label = f"Auto-médiane (skew={skew:.2f})"
                else:
                    val = self.df_clean[col].mean()
                    method_label = f"Auto-moyenne (skew={skew:.2f})"
                self.df_clean[col] = self.df_clean[col].fillna(val)
                self.transformations["imputation"][col] = method_label

        elif self.config.impute_method == "knn":
            scaler = StandardScaler()
            num_data = self.df_clean[numeric_cols].copy()
            scaled   = scaler.fit_transform(num_data.fillna(num_data.mean()))
            imputed  = KNNImputer(n_neighbors=self.config.knn_neighbors).fit_transform(scaled)
            result   = scaler.inverse_transform(imputed)
            for i, col in enumerate(numeric_cols):
                self.df_clean[col] = result[:, i]
                if col in missing_numeric:
                    self.transformations["imputation"][col] = f"KNN (k={self.config.knn_neighbors})"

        elif self.config.impute_method == "iterative":
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=50, n_jobs=self.config.n_jobs),
                random_state=42,
                max_iter=10,
            )
            self.df_clean[numeric_cols] = imputer.fit_transform(self.df_clean[numeric_cols])
            for col in missing_numeric:
                self.transformations["imputation"][col] = "Iterative MICE"

        elif self.config.impute_method == "simple":
            imputer = SimpleImputer(strategy="mean")
            self.df_clean[numeric_cols] = imputer.fit_transform(self.df_clean[numeric_cols])
            for col in missing_numeric:
                self.transformations["imputation"][col] = "Simple (moyenne)"

        if self.transformations["imputation"]:
            self.log_action(
                "Imputation terminée",
                pd.DataFrame.from_dict(
                    self.transformations["imputation"], orient="index", columns=["Méthode"]
                ),
            )
        return self

    def detect_and_fix_outliers(self) -> "DataDoctor":
        """Corrige les outliers avec 4 méthodes + visualisation avant/après."""
        if self.config.outlier_method == "none":
            return self
        assert self.df_clean is not None

        numeric_cols = self.df_clean.select_dtypes(include=np.number).columns

        if self.config.outlier_method == "iqr":
            for col in self._progress(numeric_cols, "Outliers IQR"):
                q1, q3 = self.df_clean[col].quantile(0.25), self.df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - self.config.outlier_threshold * iqr, q3 + self.config.outlier_threshold * iqr
                n_out = int(((self.df_clean[col] < lo) | (self.df_clean[col] > hi)).sum())
                if n_out:
                    self.df_clean[col] = np.clip(self.df_clean[col], lo, hi)
                    self.transformations["outliers"][col] = {"method": "IQR", "count": n_out, "bounds": [lo, hi]}
                    if self.config.interactive:
                        self._plot_outlier_fix(col, "IQR", lo, hi)

        elif self.config.outlier_method == "zscore":
            for col in self._progress(numeric_cols, "Outliers Z-score"):
                z = np.abs(stats.zscore(self.df_clean[col].dropna()))
                # reconstituons le masque sur le df entier
                col_clean = self.df_clean[col].dropna()
                outlier_vals_mask = self.df_clean[col].notna() & (
                    np.abs(stats.zscore(self.df_clean[col].fillna(self.df_clean[col].mean()))) > self.config.zscore_threshold
                )
                n_out = int(outlier_vals_mask.sum())
                if n_out:
                    mean_v, std_v = self.df_clean[col].mean(), self.df_clean[col].std()
                    lo, hi = mean_v - self.config.zscore_threshold * std_v, mean_v + self.config.zscore_threshold * std_v
                    self.df_clean[col] = np.clip(self.df_clean[col], lo, hi)
                    self.transformations["outliers"][col] = {"method": "Z-score", "count": n_out, "bounds": [lo, hi]}
                    if self.config.interactive:
                        self._plot_outlier_fix(col, "Z-score", lo, hi)

        elif self.config.outlier_method == "isolation_forest":
            df_num = self.df_clean[numeric_cols].fillna(self.df_clean[numeric_cols].median())
            scaled = StandardScaler().fit_transform(df_num)
            preds  = IsolationForest(
                contamination=self.config.contamination, random_state=42, n_jobs=self.config.n_jobs
            ).fit_predict(scaled)
            idx = np.where(preds == -1)[0]
            if len(idx):
                for col in numeric_cols:
                    self.df_clean.iloc[idx, self.df_clean.columns.get_loc(col)] = self.df_clean[col].median()
                self.transformations["outliers"]["global"] = {
                    "method": "Isolation Forest", "count": len(idx)
                }

        elif self.config.outlier_method == "dbscan":
            df_num = self.df_clean[numeric_cols].fillna(self.df_clean[numeric_cols].median())
            scaled = StandardScaler().fit_transform(df_num)
            reduced = PCA(n_components=2).fit_transform(scaled) if len(numeric_cols) > 2 else scaled
            labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(reduced)
            idx = np.where(labels == -1)[0]
            if len(idx):
                for col in numeric_cols:
                    self.df_clean.iloc[idx, self.df_clean.columns.get_loc(col)] = self.df_clean[col].median()
                self.transformations["outliers"]["global"] = {"method": "DBSCAN", "count": len(idx)}

        n_fixed = sum(
            v.get("count", 0) for k, v in self.transformations["outliers"].items()
        )
        if n_fixed:
            self.log_action("Outliers corrigés", f"{n_fixed:,} valeurs aberrantes traitées")
        return self

    def _plot_outlier_fix(self, col: str, method: str, lower: float, upper: float) -> None:
        assert self.df is not None and self.df_clean is not None
        # Seaborn/Matplotlib peuvent aussi lever des erreurs avec les dtypes nullable pandas 2.x
        df_orig_viz = self._sanitize_for_plotting(self.df)
        df_clean_viz = self._sanitize_for_plotting(self.df_clean)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=df_orig_viz, y=col, ax=ax1)
        ax1.set_title(f"Avant ({col})")
        sns.boxplot(data=df_clean_viz, y=col, ax=ax2)
        ax2.axhline(lower, color="r", linestyle="--", alpha=0.6, label="bornes")
        ax2.axhline(upper, color="r", linestyle="--", alpha=0.6)
        ax2.set_title(f"Après {method} ({col})")
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def normalize_text_data(self) -> "DataDoctor":
        """Normalise le texte en ignorant les colonnes à patterns sémantiques détectés."""
        if not self.config.normalize_text or self.config.text_case == "none":
            return self
        assert self.df_clean is not None

        text_changes: Dict[str, str] = {}
        case_fn = {
            "lower": str.lower, "upper": str.upper,
            "title": str.title,
        }.get(self.config.text_case, str.title)

        for col in self._progress(
            self.df_clean.select_dtypes("object").columns, "Normalisation texte"
        ):
            # Ne pas toucher aux colonnes à pattern sémantique (emails, téléphones…)
            if col in self.col_patterns:
                continue
            if self.df_clean[col].nunique() > self.config.max_categories:
                continue
            self.df_clean[col] = (
                self.df_clean[col].astype(str).str.strip().apply(
                    lambda v: case_fn(v) if v not in ("nan", "None", "") else v
                )
            )
            text_changes[col] = self.config.text_case

        if text_changes:
            self.transformations["text_normalization"].update(text_changes)
            self.log_action(
                "Normalisation texte",
                pd.DataFrame.from_dict(text_changes, orient="index", columns=["Format"]),
            )
        return self

    def encode_categorical_data(self) -> "DataDoctor":
        """Encodage des colonnes à haute cardinalité."""
        assert self.df_clean is not None
        encoding_changes: Dict[str, str] = {}

        for col in self._progress(
            self.df_clean.select_dtypes("object").columns, "Encodage catégoriel"
        ):
            if self.df_clean[col].nunique() <= self.config.max_categories:
                continue
            if HAS_TARGET_ENCODER:
                encoder = TargetEncoder()
                self.df_clean[col] = encoder.fit_transform(
                    self.df_clean[col], self.df_clean[col]
                )
                encoding_changes[col] = "TargetEncoder"
            else:
                # Fallback : label encoding ordinal
                self.df_clean[col] = pd.Categorical(self.df_clean[col]).codes
                encoding_changes[col] = "OrdinalFallback"

        if encoding_changes:
            self.transformations["encoding_fixes"].update(encoding_changes)
            self.log_action(
                "Encodage catégoriel",
                pd.DataFrame.from_dict(encoding_changes, orient="index", columns=["Méthode"]),
            )
        return self

    def visualize_missing(self) -> "DataDoctor":
        assert self.df_clean is not None
        if self.config.interactive:
            # missingno utilise matplotlib en interne ; la conversion évite les warnings/crashes silencieux
            msno.matrix(self._sanitize_for_plotting(self.df_clean), figsize=(10, 4))
            plt.title("Matrice des Valeurs Manquantes (post-nettoyage)")
            plt.show()
        else:
            missing = self.df_clean.isna().mean() * 100
            top = missing[missing > 0].sort_values(ascending=False).head(10)
            if not top.empty:
                print(tabulate(top.reset_index(), headers=["Colonne", "% Manquant"]))
        return self

    def generate_profile(self) -> "DataDoctor":
        """Génère un profil HTML : ydata-profiling si dispo, sinon rapport maison."""
        if not self.config.generate_profile:
            return self
        assert self.df_clean is not None

        if HAS_YDATA:
            try:
                profile = ProfileReport(self.df_clean, title="DataDoctor Pro — Profiling Report")
                if self.config.interactive:
                    display(HTML("<h3>📊 Profil des Données :</h3>"))
                    display(profile)
                profile.to_file(self.config.report_file)
                self.log_action("Profil ydata-profiling", f"→ {self.config.report_file}")
                return self
            except Exception as exc:
                self.log_action("ydata-profiling indisponible", str(exc))

        # ── Rapport HTML maison ───────────────────────────────────────────────
        self._generate_builtin_html_report()
        return self

    def _generate_builtin_html_report(self) -> None:
        """Génère un rapport HTML self-contained sans dépendance externe."""
        assert self.df_clean is not None
        df = self.df_clean

        rows = []
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            pct_miss  = round(n_missing / max(len(df), 1) * 100, 1)
            dtype_str = str(df[col].dtype)
            nuniq     = df[col].nunique()
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_str = (
                    f"min={df[col].min():.3g} | "
                    f"mean={df[col].mean():.3g} | "
                    f"max={df[col].max():.3g} | "
                    f"std={df[col].std():.3g}"
                )
            else:
                top = df[col].value_counts().head(3).index.tolist()
                stats_str = f"top-3: {', '.join(str(v) for v in top)}"
            pattern = self.col_patterns.get(col, "—")
            rows.append(f"""
            <tr>
              <td>{col}</td><td>{dtype_str}</td><td>{nuniq:,}</td>
              <td>{n_missing:,} ({pct_miss}%)</td>
              <td>{stats_str}</td><td>{pattern}</td>
            </tr>""")

        score_html = ""
        if self.score_before and self.score_after:
            score_html = f"""
            <h2>📊 Score Qualité</h2>
            <table>
              <tr><th>Dimension</th><th>Avant</th><th>Après</th></tr>
              <tr><td>Complétude</td><td>{self.score_before.completeness}%</td><td>{self.score_after.completeness}%</td></tr>
              <tr><td>Unicité</td><td>{self.score_before.uniqueness}%</td><td>{self.score_after.uniqueness}%</td></tr>
              <tr><td>Cohérence</td><td>{self.score_before.consistency}%</td><td>{self.score_after.consistency}%</td></tr>
              <tr><td>Validité</td><td>{self.score_before.validity}%</td><td>{self.score_after.validity}%</td></tr>
              <tr><td><b>Global</b></td><td><b>{self.score_before.overall}%</b></td><td><b>{self.score_after.overall}%</b></td></tr>
            </table>"""

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>DataDoctor Pro — Rapport</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; color: #222; }}
    h1   {{ color: #1a56db; }}
    h2   {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
    th   {{ background: #1a56db; color: white; padding: 8px 12px; text-align: left; }}
    td   {{ padding: 6px 12px; border-bottom: 1px solid #e5e7eb; }}
    tr:hover td {{ background: #f0f4ff; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
    .ok  {{ background: #d1fae5; color: #065f46; }}
    .warn{{ background: #fef3c7; color: #92400e; }}
    .err {{ background: #fee2e2; color: #991b1b; }}
  </style>
</head>
<body>
  <h1>DataDoctor Pro v{self.VERSION} — Rapport de Nettoyage</h1>
  <p><b>Fichier source :</b> {self.config.source_file} &nbsp;|&nbsp;
     <b>Généré le :</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <h2>📋 Dimensions</h2>
  <p>Avant : <b>{self.original_shape}</b> &nbsp;→&nbsp;
     Après : <b>{df.shape}</b></p>
  {score_html}
  <h2>🔬 Profil des Colonnes</h2>
  <table>
    <tr>
      <th>Colonne</th><th>Type</th><th>Valeurs uniques</th>
      <th>Manquantes</th><th>Statistiques</th><th>Pattern détecté</th>
    </tr>
    {"".join(rows)}
  </table>
</body>
</html>"""

        Path(self.config.report_file).write_text(html, encoding="utf-8")
        self.log_action("Rapport HTML généré", f"→ {self.config.report_file}")
        if self.config.interactive:
            display(HTML(f"<p>📄 Rapport enregistré : <code>{self.config.report_file}</code></p>"))

    def save_outputs(self) -> "DataDoctor":
        """Sauvegarde le dataset nettoyé + le plan de transformation JSON."""
        assert self.df_clean is not None
        fmt = self.config.output_format.lower()
        targets = {
            "csv":     lambda: self.df_clean.to_csv(self.config.target_file, index=False),
            "xlsx":    lambda: self.df_clean.to_excel(self.config.target_file, index=False),
            "excel":   lambda: self.df_clean.to_excel(self.config.target_file, index=False),
            "parquet": lambda: self.df_clean.to_parquet(self.config.target_file),
            "pickle":  lambda: self.df_clean.to_pickle(self.config.target_file),
            "json":    lambda: self.df_clean.to_json(self.config.target_file, orient="records", force_ascii=False),
        }
        if fmt in targets:
            targets[fmt]()

        # Journal texte
        with open(self.config.log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_entries))

        # Plan de transformation reproductible
        plan_file = self.config.log_file.replace(".txt", "-plan.json")
        plan = {
            "datadoctor_version": self.VERSION,
            "generated_at": datetime.now().isoformat(),
            "source_file": self.config.source_file,
            "config": self.config.to_dict(),
            "transformations": {
                k: (list(v) if isinstance(v, (set, list)) else v)
                for k, v in self.transformations.items()
            },
            "score_before": asdict(self.score_before) if self.score_before else None,
            "score_after":  asdict(self.score_after)  if self.score_after  else None,
            "pipeline_results": [asdict(r) for r in self.step_results],
        }
        Path(plan_file).write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")

        self.log_action(
            "Sauvegarde effectuée",
            f"Dataset → {self.config.target_file}\n"
            f"Log     → {self.config.log_file}\n"
            f"Plan    → {plan_file}",
        )
        return self

    # ── Résumé ────────────────────────────────────────────────────────────────

    def show_summary(self) -> None:
        assert self.df_clean is not None and self.original_shape is not None

        summary_data = {
            "Opération": [
                "Colonnes initiales",
                "Colonnes supprimées",
                "Lignes initiales",
                "Doublons supprimés",
                "Conversions de type",
                "Colonnes imputées",
                "Colonnes outliers corrigées",
                "Colonnes texte normalisées",
                "Colonnes encodées",
                "Patterns texte détectés",
            ],
            "Valeur": [
                self.original_shape[1],
                len(self.transformations["dropped_columns"]),
                self.original_shape[0],
                self.transformations["duplicates"],
                len(self.transformations["type_conversions"]),
                len(self.transformations["imputation"]),
                len([k for k in self.transformations["outliers"] if k != "global"]),
                len(self.transformations["text_normalization"]),
                len(self.transformations["encoding_fixes"]),
                len(self.transformations["col_patterns"]),
            ],
        }

        display(HTML("<h2>✅ Résumé des Transformations DataDoctor Pro</h2>"))
        display(pd.DataFrame(summary_data))

        # Score avant / après
        if self.score_before and self.score_after:
            display(HTML("<h3>📈 Évolution du Score Qualité</h3>"))
            score_df = pd.DataFrame({
                "Dimension":   ["Complétude", "Unicité", "Cohérence", "Validité", "🏆 Global"],
                "Avant (%)":   [self.score_before.completeness, self.score_before.uniqueness,
                                self.score_before.consistency, self.score_before.validity,
                                self.score_before.overall],
                "Après (%)":   [self.score_after.completeness, self.score_after.uniqueness,
                                self.score_after.consistency, self.score_after.validity,
                                self.score_after.overall],
            })
            score_df["Δ"] = (score_df["Après (%)"] - score_df["Avant (%)"]).map(
                lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
            )
            display(score_df)

        # Pipeline steps
        if self.step_results:
            display(HTML("<h3>🔄 Étapes du Pipeline</h3>"))
            steps_df = pd.DataFrame([{
                "Étape":   r.name,
                "Statut":  "✅" if r.success else "❌",
                "Durée":   f"{r.elapsed:.2f}s",
                "Erreur":  r.error or "",
            } for r in self.step_results])
            display(steps_df)

        if self.config.interactive:
            display(HTML("<h3>👀 Aperçu des données nettoyées :</h3>"))
            display(self.df_clean.head(3))
            display(HTML("<h3>🔠 Types finaux :</h3>"))
            display(pd.DataFrame(self.df_clean.dtypes, columns=["Type"]))

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Exécute le pipeline complet avec gestion d'erreur par étape."""
        self.config.set_source(self.config.source_file)

        steps = [s for s in self.config.pipeline_steps if s != "load_data"]

        # Chargement toujours en premier (si erreur → exception propagée)
        self.load_data()

        for step in self._progress(steps, "Pipeline DataDoctor Pro"):
            self._run_step(step)

        # Score final
        assert self.df_clean is not None
        self.score_after = DataQualityScore().compute(self.df_clean)
        self.log_action("Score qualité final", str(self.score_after))
        self.log_action(
            "Pipeline terminé",
            f"Durée totale : {datetime.now() - self.start_time}",
        )
        self.show_summary()
        return self.df_clean


# ══════════════════════════════════════════════════════════════════════════════
#  EXEMPLE D'UTILISATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Usage basique ─────────────────────────────────────────────────────────
    config = DataDoctorConfig(
        source_file="your_data.csv",
        impute_method="auto",
        outlier_method="iqr",
        interactive=True,
        progress_bar=True,
    )

    with DataDoctor(config) as doc:
        df_clean = doc.run()

    display(df_clean.head())

    # ── Sauvegarder / recharger la config ─────────────────────────────────────
    # config.to_json("my_config.json")
    # config2 = DataDoctorConfig.from_json("my_config.json")

    # ── Pipeline personnalisé (sans encodage catégoriel) ──────────────────────
    # config.pipeline_steps.remove("encode_categorical_data")
    # doc = DataDoctor(config)
    # df_clean = doc.run()

    # ── Inspection du score de qualité seul ──────────────────────────────────
    # score = DataQualityScore().compute(df_clean)
    # print(score)
