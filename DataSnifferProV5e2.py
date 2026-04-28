#!/usr/bin/env python3
"""
DataSniffeR Pro V5 - Deep Data Sniffer
======================================
Analyse exploratoire profonde avec profilage sémantique, audit qualité,
recommandations KNIME et pont vers workflows de data mining.

Philosophie: "Sniffer avant de miner" - Comprendre, sentir, décider.
"""

import argparse
import os
import sys
import logging
import json
import warnings
import gc
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, normaltest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def cprint(text, color=Colors.END, bold=False):
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.END}")

def setup_logging(output_dir, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"sniffer_log_{datetime.now():%Y%m%d_%H%M%S}.log")
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=handlers)
    return log_file

def load_and_clean(file_path, sample_frac=None, detect_hidden_missing=True):
    cprint(f"\n📂 Chargement : {file_path}", Colors.HEADER, bold=True)
    try:
        if sample_frac and sample_frac < 1.0:
            total_rows = 0
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                while chunk:
                    total_rows += chunk.count(b'\n')
                    chunk = f.read(8192)
            total_rows = max(total_rows - 1, 0)
            nrows = int(total_rows * sample_frac)
            df = pd.read_csv(file_path, low_memory=False, nrows=nrows, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
            cprint(f"⚠️ Échantillonnage : {sample_frac*100:.1f}% ({nrows:,} lignes / {total_rows:,})", Colors.YELLOW)
        else:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')
    except Exception as e:
        cprint(f"❌ Échec de lecture CSV: {e}", Colors.RED, bold=True)
        sys.exit(1)
    df = df.dropna(axis=1, how='all').drop_duplicates()
    hidden_missing_report = {}
    if detect_hidden_missing:
        hidden_missing_report = _detect_hidden_missing_values(df)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.8 * len(df):
                df[col] = converted
            else:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df, hidden_missing_report

def _detect_hidden_missing_values(df):
    hidden_patterns = {
        'string_na': ['N/A', 'n/a', 'NA', 'na', 'NULL', 'null', 'None', 'none', ''],
        'numeric_sentinels': [-999, -9999, 999, 9999, -1, 0],
        'special_chars': ['?', '*', '.', '-', '—']
    }
    report = {}
    for col in df.columns:
        findings = []
        col_data = df[col]
        if col_data.dtype == 'object':
            for pattern in hidden_patterns['string_na']:
                count = (col_data == pattern).sum()
                if count > 0:
                    findings.append(f"'{pattern}' × {count}")
            for pattern in hidden_patterns['special_chars']:
                count = (col_data == pattern).sum()
                if count > 0:
                    findings.append(f"'{pattern}' × {count}")
        if pd.api.types.is_numeric_dtype(col_data):
            for sentinel in hidden_patterns['numeric_sentinels']:
                count = (col_data == sentinel).sum()
                if count > 0 and count < len(col_data) * 0.5:
                    findings.append(f"{sentinel} × {count}")
        if findings:
            report[col] = findings
    return report

def memory_analysis(df):
    cprint("\n💾 Analyse Mémoire", Colors.HEADER, bold=True)
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024**2
    rows = []
    for col in df.columns:
        mem = df[col].memory_usage(deep=True)
        rows.append({'Colonne': col, 'Type': str(df[col].dtype), 'Memory_MB': mem/1024**2, '%': mem/memory_usage.sum()*100})
    per_col = pd.DataFrame(rows).sort_values('Memory_MB', ascending=False)
    cprint(f"Mémoire totale : {total_mb:.2f} MB", Colors.CYAN)
    print(per_col.head(10).to_string(index=False))
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        cprint(f"\n💡 Optimisation possible : {len(obj_cols)} colonnes 'object' → 'category'", Colors.YELLOW)
    return per_col

def detect_problem_type(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    candidates = []
    for col in numeric_cols:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 10:
            candidates.append((col, 'classification'))
        elif n_unique > 20:
            candidates.append((col, 'regression'))
    keywords = ['target', 'class', 'label', 'y', 'output', 'result', 'price', 'value', 'survived', 'churn']
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            for candidate_col, candidate_type in candidates:
                if candidate_col == col:
                    return col, candidate_type
    if candidates:
        return candidates[0]
    return None, None

class SemanticProfiler:
    @staticmethod
    def profile_column(series, col_name):
        profile = {
            'name': col_name,
            'technical_type': str(series.dtype),
            'semantic_type': 'unknown',
            'confidence': 0.0,
            'characteristics': [],
            'warnings': []
        }
        n_total = len(series)
        n_unique = series.nunique()
        n_missing = series.isna().sum()

        if n_unique == n_total - n_missing and n_missing == 0:
            if any(kw in col_name.lower() for kw in ['id', 'key', 'code', 'num', 'no', 'ref']):
                profile['semantic_type'] = 'identifier'
                profile['confidence'] = 0.95
                profile['warnings'].append('A exclure des features (fuite de donnees)')
                return profile

        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().between(1e9, 2e9).all():
                profile['semantic_type'] = 'timestamp_unix'
                profile['confidence'] = 0.85
                profile['warnings'].append('Convertir en datetime avant analyse')
                return profile
            if series.dropna().between(19000101, 20991231).all():
                profile['semantic_type'] = 'date_compact'
                profile['confidence'] = 0.80
                return profile

        if series.dtype == 'object' or series.dtype.name == 'category':
            unique_vals = series.dropna().unique()
            ordinal_keywords = {
                'size': ['xs', 's', 'm', 'l', 'xl', 'xxl', 'small', 'medium', 'large'],
                'level': ['low', 'medium', 'high', 'faible', 'moyen', 'eleve'],
                'grade': ['a', 'b', 'c', 'd', 'e', 'f'],
                'satisfaction': ['bad', 'poor', 'fair', 'good', 'excellent'],
                'education': ['primary', 'secondary', 'bachelor', 'master', 'phd']
            }
            vals_lower = [str(v).lower() for v in unique_vals]
            for ord_type, ord_vals in ordinal_keywords.items():
                matches = sum(1 for v in vals_lower if v in ord_vals)
                if matches >= len(unique_vals) * 0.7:
                    profile['semantic_type'] = f'ordinal_{ord_type}'
                    profile['confidence'] = matches / len(unique_vals)
                    profile['characteristics'].append(f'Categorie ordinale ({ord_type})')
                    break

            str_vals = series.dropna().astype(str)
            if str_vals.str.match(r'^[0-9]{5}$').sum() / len(str_vals) > 0.9:
                profile['semantic_type'] = 'postal_code_fr'
                profile['confidence'] = 0.90
                profile['warnings'].append('Ne pas traiter comme numerique continu')
                return profile

            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if len(str_vals) > 0 and str_vals.str.match(email_pattern).sum() / len(str_vals) > 0.8:
                profile['semantic_type'] = 'email'
                profile['confidence'] = 0.95
                profile['warnings'].append('Extraire domaine pour analyse')
                return profile
            url_pattern = r'^https?://'
            if len(str_vals) > 0 and str_vals.str.match(url_pattern).sum() / len(str_vals) > 0.8:
                profile['semantic_type'] = 'url'
                profile['confidence'] = 0.95
                return profile

            #avg_length = series.dropna().str.len().mean()
            avg_length = str_vals.str.len().mean()
            if avg_length > 50 and n_unique > n_total * 0.3:
                profile['semantic_type'] = 'free_text'
                profile['confidence'] = 0.80
                profile['characteristics'].append(f'Texte libre (longueur moyenne: {avg_length:.0f} chars)')
                profile['warnings'].append('Necessite NLP ou feature extraction')
            elif n_unique <= 50:
                profile['semantic_type'] = 'categorical'
                profile['confidence'] = 0.85
                profile['characteristics'].append(f'Categorielle ({n_unique} niveaux)')

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= 20 and (series.dropna() == series.dropna().astype(int)).all():
                if series.min() >= 0 and series.max() <= 10:
                    profile['semantic_type'] = 'score_rating'
                    profile['confidence'] = 0.80
                else:
                    profile['semantic_type'] = 'discrete_numeric'
                    profile['confidence'] = 0.75
            else:
                if series.min() >= 0 and skew(series.dropna()) > 1.5:
                    profile['semantic_type'] = 'count_positive'
                    profile['confidence'] = 0.70
                else:
                    profile['semantic_type'] = 'continuous_numeric'
                    profile['confidence'] = 0.75

        if n_unique == 1:
            profile['semantic_type'] = 'constant'
            profile['confidence'] = 1.0
            profile['warnings'].append('A SUPPRIMER (variance nulle)')
        elif n_unique == 2:
            profile['semantic_type'] = 'binary'
            profile['confidence'] = 0.90
        elif n_unique / n_total < 0.01:
            profile['semantic_type'] = 'quasi_constant'
            profile['confidence'] = 0.80
            profile['warnings'].append(f'Quasi-constante ({n_unique} valeurs uniques)')

        return profile

    @classmethod
    def profile_dataset(cls, df):
        cprint("\n🔬 Profilage Semantique Profond", Colors.HEADER, bold=True)
        profiles = {}
        for col in df.columns:
            profiles[col] = cls.profile_column(df[col], col)
        semantic_counts = Counter(p['semantic_type'] for p in profiles.values())
        cprint("\n📊 Repartition semantique:", Colors.CYAN, bold=True)
        for sem_type, count in semantic_counts.most_common():
            cprint(f"  • {sem_type}: {count} colonne(s)", Colors.CYAN)
        warnings_found = [(col, p['warnings']) for col, p in profiles.items() if p['warnings']]
        if warnings_found:
            cprint("\n⚠️ Alertes semantiques:", Colors.YELLOW, bold=True)
            for col, warns in warnings_found:
                for w in warns:
                    cprint(f"  [{col}] {w}", Colors.YELLOW)
        return profiles

class QualityAuditor:
    @staticmethod
    def audit(df, hidden_missing_report):
        cprint("\n🔍 Audit Qualite des Donnees", Colors.HEADER, bold=True)
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        missing_total = df.isna().sum().sum()
        missing_pct = (missing_total / total_cells) * 100
        missing_pattern = QualityAuditor._analyze_missing_pattern(df)
        duplicated_rows = df.duplicated().sum()
        duplicated_pct = (duplicated_rows / n_rows) * 100

        type_issues = 0
        for col in df.select_dtypes(include=['object']).columns:
            col_str = df[col].dropna().astype(str)
            if len(col_str) > 0 and col_str.str.contains(r'\d', regex=True, na=False).sum() / len(col_str) > 0.8:
                type_issues += 1

        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_cols = 0
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
            if outliers > 0:
                outlier_cols += 1

        completeness_score = max(0, 100 - missing_pct * 2)
        uniqueness_score = max(0, 100 - duplicated_pct * 5)
        validity_score = max(0, 100 - (type_issues / max(n_cols, 1)) * 100)
        consistency_score = max(0, 100 - (outlier_cols / max(len(numeric_cols), 1)) * 30)
        overall_score = (completeness_score * 0.3 + uniqueness_score * 0.25 +
                        validity_score * 0.25 + consistency_score * 0.2)

        audit_report = {
            'overall_score': round(overall_score, 1),
            'scores': {
                'completeness': round(completeness_score, 1),
                'uniqueness': round(uniqueness_score, 1),
                'validity': round(validity_score, 1),
                'consistency': round(consistency_score, 1)
            },
            'details': {
                'missing_cells': int(missing_total),
                'missing_percentage': round(missing_pct, 2),
                'missing_pattern': missing_pattern,
                'duplicate_rows': int(duplicated_rows),
                'duplicate_percentage': round(duplicated_pct, 2),
                'type_mismatch_columns': type_issues,
                'columns_with_extreme_outliers': outlier_cols,
                'hidden_missing': hidden_missing_report
            },
            'recommendations': QualityAuditor._generate_recommendations(
                missing_pct, duplicated_pct, missing_pattern, hidden_missing_report
            )
        }

        cprint(f"\n📈 Score Qualite Global: {overall_score:.1f}/100",
               Colors.GREEN if overall_score > 70 else Colors.YELLOW if overall_score > 50 else Colors.RED, bold=True)
        for metric, score in audit_report['scores'].items():
            color = Colors.GREEN if score > 80 else Colors.YELLOW if score > 60 else Colors.RED
            cprint(f"  • {metric.capitalize()}: {score:.1f}", color)

        if audit_report['recommendations']:
            cprint("\n💡 Recommandations qualite:", Colors.CYAN, bold=True)
            for rec in audit_report['recommendations'][:5]:
                cprint(f"  → {rec}", Colors.CYAN)
        return audit_report

    @staticmethod
    def _analyze_missing_pattern(df):
        missing_matrix = df.isna()
        if missing_matrix.sum().sum() == 0:
            return "Aucune valeur manquante"
        missing_cols = missing_matrix.columns[missing_matrix.any()].tolist()
        if len(missing_cols) == 0:
            return "Aucune valeur manquante"
        missing_corr = missing_matrix[missing_cols].corr()
        high_corr = (missing_corr.abs() > 0.5).sum().sum() - len(missing_cols)
        if high_corr > 0:
            return "MAR (Missing At Random) - Pattern detecte entre colonnes"
        else:
            return "MCAR (Missing Completely At Random) - Distribution aleatoire"

    @staticmethod
    def _generate_recommendations(missing_pct, dup_pct, missing_pattern, hidden_missing):
        recs = []
        if missing_pct > 5:
            recs.append(f"{missing_pct:.1f}% de manquantes → Utiliser Missing Value node (KNIME)")
        if dup_pct > 1:
            recs.append(f"{dup_pct:.1f}% de doublons → Row Filter node avec 'Remove Duplicates'")
        if hidden_missing:
            recs.append(f"Valeurs manquantes masquees dans {len(hidden_missing)} colonnes → String Replacer node")
        if "MAR" in missing_pattern:
            recs.append("Pattern MAR detecte → Imputation conditionnelle recommandee")
        return recs

class DistributionAnalyst:
    @staticmethod
    def analyze(df):
        cprint("\n📐 Analyse des Distributions", Colors.HEADER, bold=True)
        numeric_cols = df.select_dtypes(include=np.number).columns
        distributions = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 8:
                continue
            dist_info = {
                'skewness': round(skew(data), 3),
                'kurtosis': round(kurtosis(data), 3),
                'mean': round(data.mean(), 4),
                'median': round(data.median(), 4),
                'std': round(data.std(), 4),
                'range': (round(data.min(), 4), round(data.max(), 4)),
                'iqr': round(data.quantile(0.75) - data.quantile(0.25), 4),
                'normality_tests': {},
                'recommended_transform': 'none',
                'transformation_reason': ''
            }
            if len(data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(data)
                    dist_info['normality_tests']['shapiro_wilk'] = {
                        'statistic': round(shapiro_stat, 4),
                        'p_value': round(shapiro_p, 6),
                        'is_normal': shapiro_p > 0.05
                    }
                except:
                    pass
            try:
                dagostino_stat, dagostino_p = normaltest(data)
                dist_info['normality_tests']['dagostino_pearson'] = {
                    'statistic': round(dagostino_stat, 4),
                    'p_value': round(dagostino_p, 6),
                    'is_normal': dagostino_p > 0.05
                }
            except:
                pass

            sk = dist_info['skewness']
            if abs(sk) > 2:
                if data.min() >= 0:
                    dist_info['recommended_transform'] = 'log1p'
                    dist_info['transformation_reason'] = f'Asymetrie forte ({sk:.2f}), valeurs positives'
                else:
                    dist_info['recommended_transform'] = 'yeo_johnson'
                    dist_info['transformation_reason'] = f'Asymetrie forte ({sk:.2f}), valeurs mixtes'
            elif abs(sk) > 1:
                if data.min() >= 0:
                    dist_info['recommended_transform'] = 'sqrt'
                    dist_info['transformation_reason'] = f'Asymetrie moderee ({sk:.2f})'
                else:
                    dist_info['recommended_transform'] = 'standardize'
                    dist_info['transformation_reason'] = 'Asymetrie moderee, standardisation suffisante'
            else:
                dist_info['recommended_transform'] = 'standardize'
                dist_info['transformation_reason'] = 'Distribution approximativement symetrique'
            distributions[col] = dist_info

        n_normal = sum(1 for d in distributions.values() if d['normality_tests'].get('shapiro_wilk', {}).get('is_normal', False))
        n_skewed = sum(1 for d in distributions.values() if abs(d['skewness']) > 1)
        cprint(f"\n📊 Resume distributions:", Colors.CYAN)
        cprint(f"  • {n_normal}/{len(distributions)} colonnes approx. normales",
               Colors.GREEN if n_normal > len(distributions)/2 else Colors.YELLOW)
        cprint(f"  • {n_skewed} colonnes asymetriques → transformation recommandee",
               Colors.YELLOW if n_skewed > 0 else Colors.GREEN)
        return distributions

class RelationshipMapper:
    @staticmethod
    def analyze(df, target_col=None, problem_type=None):
        cprint("\n🔗 Cartographie des Relations", Colors.HEADER, bold=True)
        numeric_df = df.select_dtypes(include=np.number)
        report = {
            'correlation_matrix': {},
            'strong_correlations': [],
            'redundant_pairs': [],
            'mutual_information': {},
            'cramers_v': {}
        }
        if len(numeric_df.columns) >= 2:
            corr_pearson = numeric_df.corr(method='pearson')
            corr_spearman = numeric_df.corr(method='spearman')
            strong_corr = []
            redundant = []
            cols = numeric_df.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    pearson = corr_pearson.iloc[i, j]
                    spearman = corr_spearman.iloc[i, j]
                    if abs(pearson) > 0.9:
                        redundant.append({'col1': cols[i], 'col2': cols[j], 'pearson': round(pearson, 3), 'type': 'redundant_high'})
                    elif abs(pearson) > 0.7 or abs(spearman) > 0.7:
                        strong_corr.append({'col1': cols[i], 'col2': cols[j], 'pearson': round(pearson, 3), 'spearman': round(spearman, 3)})
            report['strong_correlations'] = strong_corr
            report['redundant_pairs'] = redundant
            derived = RelationshipMapper._detect_derived_columns(numeric_df)
            report['derived_columns'] = derived

        if target_col and target_col in df.columns and problem_type:
            report['mutual_information'] = RelationshipMapper._compute_mutual_info(df, target_col, problem_type)

        cat_df = df.select_dtypes(include=['object', 'category'])
        if len(cat_df.columns) >= 2:
            cramers_results = []
            cat_cols = cat_df.columns
            for i in range(len(cat_cols)):
                for j in range(i+1, len(cat_cols)):
                    v = RelationshipMapper._cramers_v(cat_df[cat_cols[i]], cat_df[cat_cols[j]])
                    if v > 0.3:
                        cramers_results.append({'col1': cat_cols[i], 'col2': cat_cols[j], 'cramers_v': round(v, 3)})
            report['cramers_v'] = cramers_results

        if report['redundant_pairs']:
            cprint(f"\n⚠️ {len(report['redundant_pairs'])} paires redondantes (|ρ| > 0.9):", Colors.YELLOW, bold=True)
            for pair in report['redundant_pairs'][:5]:
                cprint(f"  • {pair['col1']} ↔ {pair['col2']} (ρ={pair['pearson']})", Colors.YELLOW)
        if report.get('derived_columns'):
            cprint(f"\n🔍 {len(report['derived_columns'])} colonnes potentiellement derivees:", Colors.CYAN, bold=True)
            for d in report['derived_columns'][:5]:
                cprint(f"  • {d['derived']} ≈ {d['formula']}", Colors.CYAN)
        if report['mutual_information']:
            mi_sorted = sorted(report['mutual_information'].items(), key=lambda x: x[1], reverse=True)
            cprint(f"\n🎯 Top features par Information Mutuelle:", Colors.GREEN, bold=True)
            for feat, mi in mi_sorted[:5]:
                cprint(f"  • {feat}: MI={mi:.4f}", Colors.GREEN)
        return report

    @staticmethod
    def _detect_derived_columns(df):
        derived = []
        cols = df.columns
        for i, col in enumerate(cols):
            y = df[col].dropna()
            if len(y) < 10:
                continue
            for j, other_col in enumerate(cols):
                if i == j:
                    continue
                x = df[other_col].dropna()
                common_idx = y.index.intersection(x.index)
                if len(common_idx) < 10:
                    continue
                y_common = y.loc[common_idx]
                x_common = x.loc[common_idx]
                if x_common.std() > 0:
                    slope, intercept, r_value, _, _ = stats.linregress(x_common, y_common)
                    if abs(r_value) > 0.99:
                        derived.append({'derived': col, 'base': other_col,
                                       'formula': f"{slope:.3f} * {other_col} + {intercept:.3f}",
                                       'r_squared': round(r_value**2, 6)})
                        break
        return derived

    @staticmethod
    def _compute_mutual_info(df, target_col, problem_type):
        numeric_df = df.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
        if numeric_df.empty:
            return {}
        X = numeric_df.fillna(numeric_df.median())
        y = df[target_col].fillna(df[target_col].median() if df[target_col].dtype.kind == 'f' else df[target_col].mode()[0])
        try:
            if problem_type == 'classification':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                mi = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)
            return dict(zip(X.columns, mi))
        except Exception as e:
            logging.warning(f"Erreur MI: {e}")
            return {}

    @staticmethod
    def _cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def auto_feature_engineering_v3(df, target_col=None, use_polynomials=True, max_interactions=3):
    cprint("\n🔧 Feature Engineering Automatique V3", Colors.HEADER, bold=True)
    num = df.select_dtypes(include=np.number).copy()
    if target_col and target_col in num.columns:
        num = num.drop(columns=[target_col])
    cols = list(num.columns)[:min(len(num.columns), 6)]
    new_count = 0
    created_features = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if new_count // 2 >= max_interactions * 2:
                break
            feat_name = f"{cols[i]}_x_{cols[j]}"
            df[feat_name] = num[cols[i]] * num[cols[j]]
            created_features.append(feat_name)
            new_count += 1
            feat_name = f"{cols[i]}_div_{cols[j]}"
            denom = num[cols[j]].replace(0, np.nan)
            df[feat_name] = num[cols[i]] / denom
            created_features.append(feat_name)
            new_count += 1
        if new_count // 2 >= max_interactions * 2:
            break
    if use_polynomials:
        for col in cols[:4]:
            feat_sq = f"{col}_squared"
            df[feat_sq] = num[col] ** 2
            created_features.append(feat_sq)
            feat_cu = f"{col}_cubed"
            df[feat_cu] = num[col] ** 3
            created_features.append(feat_cu)
            new_count += 2
    cprint(f"✅ {new_count} nouvelles features creees", Colors.GREEN)
    if created_features:
        cprint(f"   Features: {', '.join(created_features[:5])}{'...' if len(created_features) > 5 else ''}", Colors.CYAN)
    return df

class OutlierHunter:
    @staticmethod
    def detect(df, methods=['mad', 'zscore', 'iqr'], threshold=3.5, store_indices=True):
        cprint("\n🎯 Outlier Hunter V3 — Detection multi-methodes", Colors.HEADER, bold=True)
        report = {}
        for col in df.select_dtypes(include=np.number).columns:
            data = df[col].dropna()
            if len(data) < 10:
                report[col] = {'count': 0, 'percentage': 0, 'indices': [], 'consensus': 0}
                continue
            votes = pd.DataFrame(index=data.index)
            try:
                if 'mad' in methods:
                    med = np.median(data)
                    mad = np.median(np.abs(data - med))
                    if mad > 0:
                        modified_z = 0.6745 * (data - med) / mad
                        votes['mad'] = np.abs(modified_z) > threshold
                    else:
                        votes['mad'] = False
                if 'zscore' in methods:
                    z = np.abs(stats.zscore(data))
                    votes['zscore'] = z > threshold
                if 'iqr' in methods:
                    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    votes['iqr'] = (data < lower) | (data > upper)
                consensus = votes.sum(axis=1)
                outlier_mask = consensus >= 2
                outlier_indices = data.index[outlier_mask].tolist()
                report[col] = {
                    'count': int(outlier_mask.sum()),
                    'percentage': round(outlier_mask.sum() / len(data) * 100, 2),
                    'indices': outlier_indices[:10] if store_indices else [],
                    'consensus': int(consensus.max()) if len(consensus) > 0 else 0,
                    'method_breakdown': {m: int(votes[m].sum()) for m in votes.columns}
                }
            except Exception as e:
                logging.warning(f"Erreur outliers {col}: {e}")
                report[col] = {'count': 0, 'percentage': 0, 'indices': [], 'consensus': 0}
        total_outliers = sum(r['count'] for r in report.values())
        affected_cols = sum(1 for r in report.values() if r['count'] > 0)
        cprint(f"✅ {total_outliers} outliers dans {affected_cols} colonnes (consensus >=2)", Colors.GREEN)
        return report

def export_statistics(df, output_dir):
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) == 0:
        cprint("⚠️ Aucune colonne numerique pour les exports stats", Colors.YELLOW)
        return
    desc_stats = numeric_df.describe().round(4)
    desc_stats.to_csv(os.path.join(output_dir, 'descriptive_stats.csv'))
    cprint("✅ descriptive_stats.csv exporte", Colors.GREEN)
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr().round(3)
        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        cprint("✅ correlation_matrix.csv exporte", Colors.GREEN)

class KNIMEBridge:
    @staticmethod
    def generate_config(df, semantic_profiles, quality_report, distributions, relationships, outliers, target_col=None, problem_type=None):
        cprint("\n🌉 Generation du pont KNIME", Colors.HEADER, bold=True)
        config = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'datasniffer_version': '5.0',
                'dataset_shape': list(df.shape),
                'target_column': target_col,
                'problem_type': problem_type
            },
            'knime_workflow_recommendations': {
                'input_nodes': ['CSV Reader', 'File Reader'],
                'preprocessing_pipeline': [],
                'feature_engineering_nodes': [],
                'modeling_nodes': [],
                'evaluation_nodes': []
            },
            'column_actions': {},
            'transformations': {},
            'warnings': []
        }
        for col in df.columns:
            profile = semantic_profiles.get(col, {})
            sem_type = profile.get('semantic_type', 'unknown')
            action = {
                'column': col,
                'semantic_type': sem_type,
                'technical_type': str(df[col].dtype),
                'action': 'keep',
                'knime_nodes': []
            }
            if sem_type == 'identifier':
                action['action'] = 'drop'
                action['reason'] = 'Identifiant unique - risque de fuite de donnees'
                action['knime_nodes'].append('Column Filter')
            elif sem_type == 'constant':
                action['action'] = 'drop'
                action['reason'] = 'Variance nulle'
                action['knime_nodes'].append('Column Filter')
            elif sem_type == 'quasi_constant':
                action['action'] = 'review'
                action['reason'] = 'Quasi-constante'
                action['knime_nodes'].append('Value Counter')
            elif sem_type == 'timestamp_unix':
                action['action'] = 'transform'
                action['transformation'] = 'unix_to_datetime'
                action['knime_nodes'].append('String to Date&Time')
            elif sem_type == 'date_compact':
                action['action'] = 'transform'
                action['transformation'] = 'yyyymmdd_to_datetime'
                action['knime_nodes'].append('String to Date&Time')
            elif sem_type == 'postal_code_fr':
                action['action'] = 'transform'
                action['transformation'] = 'string_categorical'
                action['reason'] = 'Code postal → categoriel'
                action['knime_nodes'].append('Number To String')
            elif sem_type == 'email':
                action['action'] = 'feature_extract'
                action['extraction'] = 'domain'
                action['knime_nodes'].append('Regex Split')
            elif sem_type == 'free_text':
                action['action'] = 'nlp'
                action['knime_nodes'].append('Strings To Document')
                action['knime_nodes'].append('Bag Of Words Creator')
            elif sem_type.startswith('ordinal_'):
                action['action'] = 'encode'
                action['encoding'] = 'ordinal'
                action['knime_nodes'].append('Category To Number')
            if col in distributions:
                dist = distributions[col]
                if dist['recommended_transform'] != 'none':
                    config['transformations'][col] = {
                        'recommended': dist['recommended_transform'],
                        'reason': dist['transformation_reason'],
                        'skewness': dist['skewness'],
                        'knime_nodes': KNIMEBridge._transform_to_knime(dist['recommended_transform'])
                    }
            config['column_actions'][col] = action

        pipeline = config['knime_workflow_recommendations']['preprocessing_pipeline']
        if quality_report['details']['missing_percentage'] > 0:
            pipeline.append('Missing Value')
        if quality_report['details']['duplicate_percentage'] > 1:
            pipeline.append('Row Filter (Remove Duplicates)')
        if len(config['transformations']) > 0:
            pipeline.append('Normalizer')
            config['knime_workflow_recommendations']['feature_engineering_nodes'].append('Normalizer')
        n_cat = len(df.select_dtypes(include=['object', 'category']).columns)
        if n_cat > 0:
            pipeline.append('One to Many')
            config['knime_workflow_recommendations']['feature_engineering_nodes'].append('One to Many')
        if relationships.get('redundant_pairs'):
            pipeline.append('Correlation Filter')
            config['knime_workflow_recommendations']['feature_engineering_nodes'].append('Correlation Filter')

        if problem_type == 'classification':
            config['knime_workflow_recommendations']['modeling_nodes'] = [
                'Decision Tree Learner', 'Random Forest Learner', 'XGBoost Tree Learner', 'Logistic Regression Learner'
            ]
            config['knime_workflow_recommendations']['evaluation_nodes'] = ['Scorer', 'ROC Curve', 'Confusion Matrix']
        elif problem_type == 'regression':
            config['knime_workflow_recommendations']['modeling_nodes'] = [
                'Linear Regression Learner', 'Random Forest Learner', 'Gradient Boosted Trees Learner'
            ]
            config['knime_workflow_recommendations']['evaluation_nodes'] = ['Numeric Scorer', 'Regression Predictor']

        if quality_report['details']['hidden_missing']:
            config['warnings'].append("Valeurs manquantes masquees → String Replacer node requis")
        if relationships.get('redundant_pairs'):
            config['warnings'].append(f"{len(relationships['redundant_pairs'])} paires redondantes → Column Filter")
        return config

    @staticmethod
    def _transform_to_knime(transform_name):
        mapping = {
            'log1p': ['Math Formula (log("column"+1))'],
            'sqrt': ['Math Formula (sqrt("column"))'],
            'yeo_johnson': ['Python Script (scipy.stats.yeojohnson)'],
            'standardize': ['Normalizer'],
            'minmax': ['Normalizer (Min-Max)']
        }
        return mapping.get(transform_name, ['Math Formula'])

    @classmethod
    def export(cls, config, output_dir):
        path = os.path.join(output_dir, 'knime_bridge_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        cprint(f"✅ Pont KNIME exporte: {path}", Colors.GREEN)
        cprint("\n🎯 Pipeline KNIME recommandee:", Colors.CYAN, bold=True)
        pipeline = config['knime_workflow_recommendations']['preprocessing_pipeline']
        for i, node in enumerate(pipeline, 1):
            cprint(f"  {i}. {node}", Colors.CYAN)
        if config['transformations']:
            cprint("\n🔧 Transformations suggerees:", Colors.CYAN)
            for col, trans in list(config['transformations'].items())[:5]:
                cprint(f"  • {col}: {trans['recommended']} ({trans['reason']})", Colors.CYAN)

def missing_3d_heatmap(df, output_dir, max_rows=500, max_cols=30):
    try:
        missing_matrix = df.isna().astype(int)
        if missing_matrix.shape[1] > max_cols:
            missing_matrix = missing_matrix.iloc[:, :max_cols]
        if missing_matrix.shape[0] > max_rows:
            has_missing = missing_matrix.sum(axis=1) > 0
            n_missing = has_missing.sum()
            n_complete = (~has_missing).sum()
            sample_missing = missing_matrix[has_missing].sample(min(n_missing, max_rows//2), random_state=42) if n_missing > 0 else pd.DataFrame()
            sample_complete = missing_matrix[~has_missing].sample(min(n_complete, max_rows//2), random_state=42) if n_complete > 0 else pd.DataFrame()
            missing_matrix = pd.concat([sample_missing, sample_complete]).sort_index()
        if missing_matrix.shape[0] == 0 or missing_matrix.shape[1] == 0:
            cprint("⚠️ Pas assez de donnees pour la heatmap 3D", Colors.YELLOW)
            return
        fig = go.Figure(data=[go.Surface(
            z=missing_matrix.values,
            x=list(range(missing_matrix.shape[1])),
            y=list(range(missing_matrix.shape[0])),
            colorscale='Viridis',
            hovertemplate='Col: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>'
        )])
        fig.update_layout(
            title='Vue 3D des valeurs manquantes (echantillonnee)',
            scene=dict(xaxis_title='Colonnes', yaxis_title='Lignes (echantillon)', zaxis_title='Manquant (0=Non, 1=Oui)'),
            width=900, height=700
        )
        fig.write_html(os.path.join(output_dir, 'missing_3d.html'))
        cprint("✅ Heatmap 3D des manquantes generee", Colors.GREEN)
    except Exception as e:
        cprint(f"⚠️ Erreur generation heatmap 3D: {e}", Colors.YELLOW)

def create_ultimate_dashboard_v5(df, output_dir, outliers_report, semantic_profiles,
                                  quality_report, distributions, relationships, knime_config):
    cprint("\n🎨 Generation du dashboard V5 prescriptif...", Colors.YELLOW, bold=True)
    os.makedirs(output_dir, exist_ok=True)
    num = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(include=['object', 'category'])
    dates = df.select_dtypes(include=['datetime64'])

    corr_data = '{}'
    if len(num.columns) >= 2:
        corr = num.corr().round(3)
        corr_data = json.dumps({'values': corr.values.tolist(), 'columns': list(corr.columns)})

    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    missing_filtered = missing_pct[missing_pct > 0]
    missing_data = json.dumps({'columns': missing_filtered.index.tolist(), 'percent': missing_filtered.values.tolist()})

    dist_data = {}
    for col, info in distributions.items():
        dist_data[col] = {
            'skewness': float(info['skewness']),
            'kurtosis': float(info['kurtosis']),
            'transform': info['recommended_transform'],
            'reason': info['transformation_reason']
        }

    quality_json = json.dumps(quality_report, default=str)
    semantic_json = json.dumps({
        col: {'type': p['semantic_type'], 'confidence': p['confidence'], 'warnings': p['warnings']}
        for col, p in semantic_profiles.items()
    })
    pipeline_json = json.dumps(knime_config.get('knime_workflow_recommendations', {}), default=str)
    outliers_json = json.dumps({
        col: {'count': data['count'], 'percentage': data['percentage'],
              'indices': data.get('indices', [])[:5], 'methods': data.get('method_breakdown', {})}
        for col, data in sorted(outliers_report.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
    })

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>DataSniffeR Pro V5 — Deep Data Sniffer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  :root {{ --primary: #667eea; --secondary: #764ba2; --success: #28a745; --warning: #ffc107; --danger: #dc3545; }}
  body {{ background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); padding: 20px; min-height: 100vh; font-family: 'Segoe UI', system-ui, sans-serif; }}
  .container {{ max-width: 1500px; }}
  .card {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); border: none; }}
  .nav-tabs {{ border: none; margin-bottom: 0; }}
  .nav-tabs .nav-link {{ background: rgba(255,255,255,0.2); color: white; margin-right: 5px; border-radius: 10px 10px 0 0; border: none; }}
  .nav-tabs .nav-link.active {{ background: white; color: var(--primary); font-weight: bold; }}
  .stat-card {{ background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: white; padding: 15px; border-radius: 12px; text-align: center; transition: transform 0.2s; }}
  .stat-card:hover {{ transform: translateY(-3px); }}
  .stat-number {{ font-size: 32px; font-weight: bold; }}
  .score-badge {{ font-size: 24px; padding: 10px 20px; border-radius: 50px; }}
  .score-high {{ background: var(--success); color: white; }}
  .score-medium {{ background: var(--warning); color: #333; }}
  .score-low {{ background: var(--danger); color: white; }}
  .recommendation-box {{ background: #f8f9fa; border-left: 4px solid var(--primary); padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
  .warning-box {{ background: #fff3cd; border-left: 4px solid var(--warning); padding: 10px; margin: 5px 0; border-radius: 0 8px 8px 0; }}
  .knime-node {{ background: #e3f2fd; border: 1px solid #90caf9; padding: 5px 10px; border-radius: 15px; display: inline-block; margin: 2px; font-size: 0.85em; }}
  .transform-badge {{ background: #f3e5f5; border: 1px solid #ce93d8; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; }}
  .semantic-tag {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; }}
  .confidence-high {{ border-left: 4px solid var(--success); }}
  .confidence-medium {{ border-left: 4px solid var(--warning); }}
  .confidence-low {{ border-left: 4px solid var(--danger); }}
</style>
</head>
<body>
<div class="container">
  <div class="card text-center" style="color: white; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
    <h2>🚀 DataSniffeR Pro V5 — Deep Data Sniffer</h2>
    <p class="mb-0">{df.shape[0]:,} lignes × {df.shape[1]} colonnes | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p class="mb-0"><small>Philosophie: "Sniffer avant de miner"</small></p>
  </div>

  <div class="card">
    <div class="row align-items-center">
      <div class="col-md-3 text-center">
        <h4>Score Qualité</h4>
        <span id="qualityBadge" class="score-badge">--</span>
      </div>
      <div class="col-md-9">
        <div class="row" id="qualityScores"></div>
      </div>
    </div>
  </div>

  <div class="row mb-3">
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{len(num.columns)}</div>Numériques</div></div>
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{len(cat.columns)}</div>Catégorielles</div></div>
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{len(dates.columns)}</div>Temporelles</div></div>
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{int(df.isna().sum().sum())}</div>Manquantes</div></div>
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{sum(1 for p in semantic_profiles.values() if p['warnings'])}</div>Alertes</div></div>
    <div class="col-md-2"><div class="stat-card"><div class="stat-number">{len(knime_config.get('transformations', {}))}</div>Transforms</div></div>
  </div>

  <ul class="nav nav-tabs" id="tabs">
    <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#t-overview">📋 Vue d'ensemble</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-semantic">🔬 Sémantique</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-quality">✅ Qualité</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-corr">🔗 Corrélations</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-dist">📐 Distributions</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-out">🎯 Outliers</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#t-knime">🌉 Pont KNIME</a></li>
  </ul>

  <div class="tab-content p-3 bg-white rounded shadow-sm">
    <div class="tab-pane fade show active" id="t-overview">
      <h4>📋 Résumé Exécutif</h4>
      <div id="executiveSummary"></div>
      <div class="row mt-3">
        <div class="col-md-6"><div id="missingPlot"></div></div>
        <div class="col-md-6"><div id="qualityRadar"></div></div>
      </div>
    </div>
    <div class="tab-pane fade" id="t-semantic">
      <h4>🔬 Profilage Sémantique</h4>
      <div class="table-responsive">
        <table class="table table-sm table-hover" id="semanticTable">
          <thead><tr><th>Colonne</th><th>Type Technique</th><th>Type Sémantique</th><th>Confiance</th><th>Alertes</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="tab-pane fade" id="t-quality">
      <h4>✅ Audit Qualité</h4>
      <div id="qualityDetails"></div>
      <h5 class="mt-3">Valeurs manquantes masquées:</h5>
      <div id="hiddenMissing"></div>
    </div>
    <div class="tab-pane fade" id="t-corr">
      <div id="corrPlot"></div>
      <h5 class="mt-3">Paires redondantes (|ρ| > 0.9):</h5>
      <div id="redundantPairs"></div>
    </div>
    <div class="tab-pane fade" id="t-dist">
      <h4>📐 Analyse des Distributions</h4>
      <div class="table-responsive">
        <table class="table table-sm" id="distTable">
          <thead><tr><th>Colonne</th><th>Skewness</th><th>Kurtosis</th><th>Normalité</th><th>Transform Recommandée</th><th>Raison</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="tab-pane fade" id="t-out">
      <h4>🎯 Outliers (Consensus multi-méthodes)</h4>
      <table class="table table-striped table-sm">
        <thead><tr><th>Colonne</th><th>Nb Outliers</th><th>%</th><th>Méthodes</th><th>Top indices</th></tr></thead>
        <tbody>
"""
    for col, data in sorted(outliers_report.items(), key=lambda x: x[1]['count'], reverse=True)[:15]:
        indices_str = ', '.join(str(i) for i in data.get('indices', [])[:5]) if data.get('indices') else '-'
        methods_str = ', '.join(f"{k}({v})" for k, v in data.get('method_breakdown', {}).items())
        html += f"<tr><td>{col}</td><td>{data['count']}</td><td>{data['percentage']:.1f}%</td><td><small>{methods_str}</small></td><td>{indices_str}</td></tr>"

    html += f"""
        </tbody>
      </table>
    </div>
    <div class="tab-pane fade" id="t-knime">
      <h4>🌉 Recommandations KNIME</h4>
      <div id="knimePipeline"></div>
      <h5 class="mt-3">Actions par colonne:</h5>
      <div class="table-responsive">
        <table class="table table-sm" id="knimeActionsTable">
          <thead><tr><th>Colonne</th><th>Type Sémantique</th><th>Action</th><th>Raison</th><th>Nodes KNIME</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <h5 class="mt-3">Transformations numériques:</h5>
      <div id="knimeTransforms"></div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
const corrData = {corr_data};
const missingData = {missing_data};
const distData = {json.dumps(dist_data)};
const qualityData = {quality_json};
const semanticData = {semantic_json};
const pipelineData = {pipeline_json};
const outliersData = {outliers_json};

const score = qualityData.overall_score || 0;
const badge = document.getElementById('qualityBadge');
badge.textContent = score + '/100';
badge.className = 'score-badge ' + (score >= 80 ? 'score-high' : score >= 60 ? 'score-medium' : 'score-low');

const scoresDiv = document.getElementById('qualityScores');
if (qualityData.scores) {{
  for (const [metric, value] of Object.entries(qualityData.scores)) {{
    const color = value >= 80 ? 'success' : value >= 60 ? 'warning' : 'danger';
    scoresDiv.innerHTML += `<div class="col-md-3 mb-2"><div class="card p-2 text-center border-${{color}}"><small class="text-muted">${{metric}}</small><strong class="text-${{color}}">${{value}}</strong></div></div>`;
  }}
}}

const summary = document.getElementById('executiveSummary');
let summaryHTML = '<div class="recommendation-box">';
summaryHTML += `<strong>🎯 Dataset:</strong> ${{qualityData.details ? qualityData.details.missing_percentage : 0}}% manquantes, ${{qualityData.details ? qualityData.details.duplicate_rows : 0}} doublons.<br>`;
if (qualityData.recommendations && qualityData.recommendations.length > 0) {{
  summaryHTML += '<strong>💡 Actions prioritaires:</strong><ul>';
  qualityData.recommendations.slice(0,3).forEach(r => summaryHTML += `<li>${{r}}</li>`);
  summaryHTML += '</ul>';
}}
summaryHTML += '</div>';
summary.innerHTML = summaryHTML;

if (corrData.values) {{
  Plotly.newPlot('corrPlot', [{{
    z: corrData.values, x: corrData.columns, y: corrData.columns, type: 'heatmap',
    colorscale: 'RdBu_r', zmid: 0,
    hovertemplate: '%{{x}} vs %{{y}}<br>ρ = %{{z:.3f}}<extra></extra>'
  }}], {{title: 'Matrice de Corrélation (Pearson)', width: 900, height: 700, margin: {{t:40,b:80,l:80,r:40}}}});
}}

if(missingData.columns.length > 0) {{
  Plotly.newPlot('missingPlot', [{{
    x: missingData.columns, y: missingData.percent, type: 'bar',
    marker: {{color: 'orangered'}}, text: missingData.percent.map(p => p.toFixed(1)+'%')
  }}], {{title: 'Valeurs manquantes (%)', xaxis: {{tickangle: -45}}, margin: {{t:40,b:100}}}});
}} else {{
  document.getElementById('missingPlot').innerHTML = '<p class="text-success mt-3">✅ Aucune valeur manquante.</p>';
}}

if (qualityData.scores) {{
  const metrics = Object.keys(qualityData.scores);
  const values = Object.values(qualityData.scores);
  Plotly.newPlot('qualityRadar', [{{
    type: 'scatterpolar', r: [...values, values[0]],
    theta: [...metrics, metrics[0]], fill: 'toself',
    marker: {{color: '#667eea'}}
  }}], {{polar: {{radialaxis: {{visible: true, range: [0, 100]}}}}, showlegend: false, margin: {{t:30,b:30}}}});
}}

const semTbody = document.querySelector('#semanticTable tbody');
for (const [col, data] of Object.entries(semanticData)) {{
  const confClass = data.confidence > 0.8 ? 'confidence-high' : data.confidence > 0.5 ? 'confidence-medium' : 'confidence-low';
  const warnings = data.warnings ? data.warnings.map(w => `<div class="warning-box"><small>${{w}}</small></div>`).join('') : '';
  semTbody.innerHTML += `<tr class="${{confClass}}"><td><strong>${{col}}</strong></td><td>${{data.technical_type || ''}}</td><td><span class="semantic-tag">${{data.type}}</span></td><td>${{(data.confidence*100).toFixed(0)}}%</td><td>${{warnings}}</td></tr>`;
}}

const qd = document.getElementById('qualityDetails');
if (qualityData.details) {{
  qd.innerHTML = `
    <div class="row">
      <div class="col-md-6"><div class="card p-3"><h6>Manquantes</h6><p>${{qualityData.details.missing_cells}} cellules (${{qualityData.details.missing_percentage}}%)</p><small>Pattern: ${{qualityData.details.missing_pattern}}</small></div></div>
      <div class="col-md-6"><div class="card p-3"><h6>Doublons</h6><p>${{qualityData.details.duplicate_rows}} lignes (${{qualityData.details.duplicate_percentage}}%)</p></div></div>
    </div>`;
}}

const hm = document.getElementById('hiddenMissing');
if (qualityData.details && qualityData.details.hidden_missing && Object.keys(qualityData.details.hidden_missing).length > 0) {{
  hm.innerHTML = '<div class="table-responsive"><table class="table table-sm"><thead><tr><th>Colonne</th><th>Patterns détectés</th></tr></thead><tbody>';
  for (const [col, patterns] of Object.entries(qualityData.details.hidden_missing)) {{
    hm.innerHTML += `<tr><td>${{col}}</td><td><code>${{patterns.join(', ')}}</code></td></tr>`;
  }}
  hm.innerHTML += '</tbody></table></div>';
}} else {{
  hm.innerHTML = '<p class="text-success">✅ Aucune valeur manquante masquée détectée.</p>';
}}

const distTbody = document.querySelector('#distTable tbody');
for (const [col, data] of Object.entries(distData)) {{
  const isNormal = data.normality_tests && data.normality_tests.shapiro_wilk ? data.normality_tests.shapiro_wilk.is_normal : false;
  const normBadge = isNormal ? '<span class="badge bg-success">Normal</span>' : '<span class="badge bg-warning text-dark">Non-normal</span>';
  const transformBadge = data.transform !== 'none' ? `<span class="transform-badge">${{data.transform}}</span>` : '<span class="badge bg-secondary">Aucune</span>';
  distTbody.innerHTML += `<tr><td>${{col}}</td><td>${{data.skewness}}</td><td>${{data.kurtosis}}</td><td>${{normBadge}}</td><td>${{transformBadge}}</td><td><small>${{data.reason}}</small></td></tr>`;
}}

const kp = document.getElementById('knimePipeline');
if (pipelineData.preprocessing_pipeline) {{
  let pipeHTML = '<div class="recommendation-box"><h6>🔄 Pipeline de Prétraitement:</h6><div class="d-flex flex-wrap">';
  pipelineData.preprocessing_pipeline.forEach((node, i) => {{
    pipeHTML += `<span class="knime-node">${{i+1}}. ${{node}}</span>`;
  }});
  pipeHTML += '</div></div>';
  if (pipelineData.modeling_nodes) {{
    pipeHTML += '<div class="recommendation-box mt-2"><h6>🤖 Modélisation:</h6><div class="d-flex flex-wrap">';
    pipelineData.modeling_nodes.forEach(node => pipeHTML += `<span class="knime-node">${{node}}</span>`);
    pipeHTML += '</div></div>';
  }}
  kp.innerHTML = pipeHTML;
}}

const kaTbody = document.querySelector('#knimeActionsTable tbody');
const colActions = {json.dumps(knime_config.get('column_actions', {}))};
for (const [col, action] of Object.entries(colActions)) {{
  const nodes = action.knime_nodes ? action.knime_nodes.map(n => `<span class="knime-node">${{n}}</span>`).join(' ') : '';
  const actionBadge = action.action === 'drop' ? '<span class="badge bg-danger">DROP</span>' : action.action === 'transform' ? '<span class="badge bg-warning text-dark">TRANSFORM</span>' : '<span class="badge bg-success">KEEP</span>';
  kaTbody.innerHTML += `<tr><td>${{col}}</td><td><span class="semantic-tag">${{action.semantic_type}}</span></td><td>${{actionBadge}}</td><td><small>${{action.reason || ''}}</small></td><td>${{nodes}}</td></tr>`;
}}

const kt = document.getElementById('knimeTransforms');
const transforms = {json.dumps(knime_config.get('transformations', {}), default=str)};
if (Object.keys(transforms).length > 0) {{
  kt.innerHTML = '<div class="table-responsive"><table class="table table-sm"><thead><tr><th>Colonne</th><th>Transformation</th><th>Raison</th><th>Nodes KNIME</th></tr></thead><tbody>';
  for (const [col, t] of Object.entries(transforms)) {{
    const nodes = t.knime_nodes ? t.knime_nodes.map(n => `<span class="knime-node">${{n}}</span>`).join(' ') : '';
    kt.innerHTML += `<tr><td>${{col}}</td><td><span class="transform-badge">${{t.recommended}}</span></td><td><small>${{t.reason}}</small></td><td>${{nodes}}</td></tr>`;
  }}
  kt.innerHTML += '</tbody></table></div>';
}}
</script>
</body>
</html>"""

    with open(os.path.join(output_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    cprint("✅ Dashboard V5 prescriptif genere !", Colors.GREEN)

def generate_pdf_report(df, output_dir, outliers_report, problem_type, quality_report, semantic_profiles):
    try:
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        pdf_path = os.path.join(output_dir, 'rapport_analyse_v5.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#2c3e50'))
        story.append(Paragraph("DataSniffeR Pro V5 — Rapport d'Analyse Profonde", title_style))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Dataset: {df.shape[0]:,} lignes x {df.shape[1]} colonnes", styles['Normal']))
        story.append(Paragraph(f"Type detecte: {problem_type if problem_type else 'Non determine'}", styles['Normal']))
        story.append(Paragraph(f"Score Qualite: {quality_report.get('overall_score', 'N/A')}/100", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Profilage Semantique:", styles['Heading2']))
        sem_data = [['Colonne', 'Type Semantique', 'Confiance', 'Alertes']]
        for col, profile in list(semantic_profiles.items())[:15]:
            warnings_str = '; '.join(profile['warnings'])[:50] if profile['warnings'] else '-'
            sem_data.append([col, profile['semantic_type'], f"{profile['confidence']:.0%}", warnings_str])
        if len(sem_data) > 1:
            t = Table(sem_data, colWidths=[2*inch, 1.5*inch, 1*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 8)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Top Outliers (Consensus multi-methodes):", styles['Heading2']))
        outlier_data = [['Colonne', 'Nb outliers', '%', 'Indices']]
        for col, data in sorted(outliers_report.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            indices_str = ', '.join(str(i) for i in data.get('indices', [])[:3]) if data.get('indices') else '-'
            outlier_data.append([col, str(data['count']), f"{data['percentage']:.1f}%", indices_str])
        if len(outlier_data) > 1:
            t = Table(outlier_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            story.append(t)

        doc.build(story)
        cprint(f"✅ Rapport PDF V5 genere: {pdf_path}", Colors.GREEN)
    except ImportError:
        cprint("⚠️ ReportLab non installe. Ignore la generation PDF (pip install reportlab)", Colors.YELLOW)
    except Exception as e:
        cprint(f"⚠️ Erreur generation PDF: {e}", Colors.YELLOW)

def main():
    parser = argparse.ArgumentParser(description="DataSniffeR Pro V5 — Deep Data Sniffer")
    parser.add_argument('--file', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--sample', type=float, default=None)
    parser.add_argument('--feature-engineering', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--heatmap3d', action='store_true')
    parser.add_argument('--export-csv', action='store_true')
    parser.add_argument('--poly-features', action='store_true')
    parser.add_argument('--no-hidden-missing', action='store_true')
    parser.add_argument('--knime-bridge', action='store_true')
    args = parser.parse_args()

    out_dir = args.output or f"sniffer_{Path(args.file).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, verbose=not args.quiet)

    cprint("""
╔══════════════════════════════════════════════════════════════╗
║           🚀 DataSniffeR Pro V5 — Deep Data Sniffer          ║
║                 "Sniffer avant de miner"                      ║
╚══════════════════════════════════════════════════════════════╝
    """, Colors.CYAN, bold=True)

    df, hidden_missing = load_and_clean(args.file, sample_frac=args.sample,
                                         detect_hidden_missing=not args.no_hidden_missing)
    cprint(f"✅ Dataset charge : {df.shape}", Colors.BLUE, bold=True)

    memory_analysis(df)

    target_col, problem_type = detect_problem_type(df)
    if target_col:
        cprint(f"🎯 Cible probable : {target_col} ({problem_type})", Colors.GREEN)
    else:
        cprint("⚠️ Aucune cible detectee automatiquement.", Colors.YELLOW)

    semantic_profiles = SemanticProfiler.profile_dataset(df)
    quality_report = QualityAuditor.audit(df, hidden_missing)
    distributions = DistributionAnalyst.analyze(df)

    if args.feature_engineering:
        df = auto_feature_engineering_v3(df, target_col, use_polynomials=args.poly_features)
        cprint(f"📈 Shape apres FE: {df.shape}", Colors.GREEN)
        gc.collect()

    relationships = RelationshipMapper.analyze(df, target_col, problem_type)

    num_cols = df.select_dtypes(include=np.number).columns
    outliers = OutlierHunter.detect(df[num_cols], store_indices=True) if len(num_cols) > 0 else {}

    knime_config = {}
    if args.knime_bridge or True:
        knime_config = KNIMEBridge.generate_config(
            df, semantic_profiles, quality_report, distributions, relationships, outliers,
            target_col, problem_type
        )
        KNIMEBridge.export(knime_config, out_dir)

    create_ultimate_dashboard_v5(df, out_dir, outliers, semantic_profiles,
                                  quality_report, distributions, relationships, knime_config)

    if args.export_csv:
        export_statistics(df, out_dir)

    if args.heatmap3d:
        missing_3d_heatmap(df, out_dir)

    if args.pdf:
        generate_pdf_report(df, out_dir, outliers, problem_type, quality_report, semantic_profiles)

    if len(num_cols) > 0:
        df[num_cols].describe().to_csv(os.path.join(out_dir, 'statistiques.csv'))

    gc.collect()

    cprint(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎉 ANALYSE TERMINEE                                         ║
║  Dossier: {out_dir:<45}  ║
║                                                              ║
║  Fichiers generes:                                           ║
║    • dashboard.html       → Exploration interactive          ║
║    • knime_bridge_config.json → Pont vers workflow KNIME     ║
║    • statistiques.csv     → Stats descriptives               ║
║    • *.log                → Log d'execution                  ║
╚══════════════════════════════════════════════════════════════╝
    """, Colors.GREEN, bold=True)

if __name__ == "__main__":
    main()
