#!/usr/bin/env python3
"""
Advanced Synthetic Dataset Generator
Version ultra-étendue - conçoit pour tester Knime / RapidMiner à l'extrême
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Faker optionnel (très recommandé pour données réalistes)
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Generator Ultra-Puissant")
    parser.add_argument("--config", default="GenRules.json", help="Fichier de configuration JSON")
    parser.add_argument("--output", default="dataset.csv", help="Fichier de sortie")
    parser.add_argument("--format", choices=["csv", "parquet", "json", "excel"], default=None,
                        help="Format de sortie (détecté automatiquement par extension sinon)")
    parser.add_argument("--num-records", type=int, default=None, help="Surcharge du nombre d'enregistrements")
    parser.add_argument("--seed", type=int, default=None, help="Seed pour reproductibilité")
    parser.add_argument("--no-shuffle", action="store_true", help="Désactiver le mélange des lignes")
    parser.add_argument("--head", type=int, default=5, help="Nombre de lignes à afficher en console")
    return parser.parse_args()


def load_rules(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def generate_single_column(var: dict, n: int, df_so_far: pd.DataFrame, faker=None):
    name = var["name"]
    vtype = var.get("type", "").lower()

    # === Rétrocompatibilité avec ton ancien GenRules.json ===
    if not vtype:
        if "mean" in var and "stddev" in var:
            vtype = "continuous"
        elif "n" in var and "p" in var:
            vtype = "binomial"
        elif "coefficients" in var:
            vtype = "polynomial"

    # ==================== GÉNÉRATION ====================
    if vtype in ("continuous", "numerical"):
        dist = var.get("distribution", "normal").lower()
        p = var.get("params", {})
        if not p and "mean" in var:
            p = {"loc": var["mean"], "scale": var.get("stddev", 1)}

        if dist == "normal":
            data = np.random.normal(loc=p.get("loc", 0), scale=p.get("scale", 1), size=n)
        elif dist == "uniform":
            data = np.random.uniform(p.get("low", 0), p.get("high", 1), n)
        elif dist == "exponential":
            data = np.random.exponential(p.get("scale", 1.0), n)
        elif dist == "lognormal":
            data = np.random.lognormal(p.get("mean", 0), p.get("sigma", 1), n)
        elif dist == "gamma":
            data = np.random.gamma(p.get("shape", 2), p.get("scale", 2), n)
        elif dist == "poisson":
            data = np.random.poisson(p.get("lam", p.get("mu", 5)), n)
        else:
            data = np.random.normal(0, 1, n)

    elif vtype == "binomial":
        data = np.random.binomial(var.get("n", 1), var.get("p", 0.5), n)

    elif vtype == "polynomial":
        coeffs = var.get("coefficients", [1, 0, 0])
        x = np.random.random(n)
        data = np.polyval(coeffs, x)

    elif vtype == "categorical":
        cats = var.get("categories", ["A", "B", "C"])
        probs = var.get("probs")
        if probs is None:
            probs = [1 / len(cats)] * len(cats)
        data = np.random.choice(cats, size=n, p=probs)

    elif vtype == "boolean":
        data = np.random.binomial(1, var.get("p", 0.5), n).astype(bool)

    elif vtype == "integer":
        data = np.random.randint(var.get("low", 0), var.get("high", 100) + 1, n)

    elif vtype == "datetime":
        start = pd.to_datetime(var.get("start", "2020-01-01"))
        end = pd.to_datetime(var.get("end", "2025-12-31"))
        delta = (end - start).days
        data = start + pd.to_timedelta(np.random.randint(0, delta + 1, n), unit="d")

    elif vtype == "expression":
        formula = var["formula"]
        context = {
            "np": np, "pd": pd, "n": n, "random": random,
            **{col: df_so_far[col].values for col in df_so_far.columns}
        }
        try:
            data = eval(formula, {"__builtins__": None}, context)
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
        except Exception as e:
            print(f"Erreur évaluation expression '{name}': {e}")
            data = np.full(n, np.nan)

    elif vtype == "faker" and FAKER_AVAILABLE and faker:
        provider = var.get("provider", "name")
        method = getattr(faker, provider, faker.name)
        data = [method() for _ in range(n)]

    elif vtype == "string":
        length = var.get("length", 12)
        data = ["".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=length))
                for _ in range(n)]

    else:
        data = np.random.normal(0, 1, n)

    # === Post-traitements ===
    # Valeurs manquantes
    na_rate = var.get("na_rate") or var.get("missing_percent", 0.0)
    if na_rate > 0:
        mask = np.random.random(n) < na_rate
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            data = np.where(mask, np.nan, data)
        else:
            s = pd.Series(data)
            s[mask] = np.nan
            data = s.values

    # Clipping
    if var.get("clip", False) and "min" in var and "max" in var and isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            data = np.clip(data, var["min"], var["max"])

    return data


def generate_dataset(rules: dict, faker=None) -> pd.DataFrame:
    n = rules.get("num_records", 1000)
    seed = rules.get("seed")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        if FAKER_AVAILABLE and faker:
            faker.seed_instance(seed)

    df = pd.DataFrame()

    for var in rules.get("variables", []):
        col_data = generate_single_column(var, n, df, faker)
        df[var["name"]] = col_data

    return df


def save_dataset(df: pd.DataFrame, output_path: str, fmt: str = None):
    if fmt is None:
        ext = os.path.splitext(output_path)[1].lower()
        fmt = {"csv": "csv", ".parquet": "parquet", ".parq": "parquet",
               ".json": "json", ".xlsx": "excel", ".xls": "excel"}.get(ext, "csv")

    if fmt == "csv":
        df.to_csv(output_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(output_path, index=False)
    elif fmt == "json":
        df.to_json(output_path, orient="records", lines=True)
    elif fmt == "excel":
        try:
            df.to_excel(output_path, index=False)
        except ImportError:
            print("openpyxl/xlsxwriter non installé → sauvegarde en CSV")
            df.to_csv(output_path + ".csv", index=False)
            return

    print(f"✅ Dataset généré : {len(df):,} lignes × {len(df.columns)} colonnes → {output_path}")


def main():
    args = parse_args()
    rules = load_rules(args.config)

    if args.num_records:
        rules["num_records"] = args.num_records
    if args.seed:
        rules["seed"] = args.seed

    faker = Faker() if FAKER_AVAILABLE else None

    df = generate_dataset(rules, faker)

    # Mélange (très utile pour éviter les biais de génération)
    if not args.no_shuffle:
        df = df.sample(frac=1, random_state=rules.get("seed")).reset_index(drop=True)

    save_dataset(df, args.output, args.format)

    print("\n=== Aperçu ===")
    print(df.head(args.head))
    print("\n=== Statistiques ===")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
