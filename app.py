# -*- coding: utf-8 -*-
"""Civil360 : étude pédagogique intégrée d'un bâtiment rectangulaire de plain-pied.

Trois fichiers livrés ; données SQLite hors du dossier. Aucune validation d'ouvrage.
Les fonctions de calcul sont indépendantes de l'interface et testables directement.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import html
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
from datetime import date, timedelta
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

VERSION = "1.0"
PURPLE, LIME, BLUE, ORANGE = "#6845E8", "#D7F86B", "#468BD8", "#E6A251"
INK, MUTED = "#282238", "#706980"
ITEMS = {
    "excavation": ("Terrassement", "m³", "Longueur × largeur × profondeur"),
    "concrete": ("Béton", "m³", "Dalle + N semelles rectangulaires"),
    "walls": ("Maçonnerie", "m²", "Périmètre × hauteur × (1 − ouvertures / 100)"),
    "plaster": ("Enduits", "m²", "2 × surface nette des murs extérieurs"),
    "roof": ("Couverture", "m²", "Surface au sol × coefficient de toiture"),
}
GEOMETRY = {
    "length": ("Longueur (m)", 12.0, 0.5, 100.0),
    "width": ("Largeur (m)", 8.0, 0.5, 100.0),
    "height": ("Hauteur des murs (m)", 3.0, 0.5, 8.0),
    "slab": ("Épaisseur de dalle (m)", 0.12, 0.01, 0.8),
    "depth": ("Profondeur de décapage (m)", 0.30, 0.01, 3.0),
    "openings": ("Ouvertures dans les murs (%)", 20.0, 0.0, 95.0),
    "pads": ("Nombre de semelles", 8, 0, 200),
    "pad_l": ("Longueur d'une semelle (m)", 1.0, 0.1, 5.0),
    "pad_w": ("Largeur d'une semelle (m)", 1.0, 0.1, 5.0),
    "pad_h": ("Épaisseur d'une semelle (m)", 0.4, 0.05, 2.0),
    "roof_factor": ("Coefficient de surface de toiture", 1.15, 1.0, 2.0),
}
TASK_COLUMNS = ["id", "name", "item", "workers", "rate", "wage", "pred", "progress"]


def uid():
    return uuid.uuid4().hex[:12]


def number(value, name, low=0.0, high=1e12, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} : une valeur numérique est attendue.")
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} : champ vide ou valeur non numérique.") from None
    if not math.isfinite(val) or not low <= val <= high:
        raise ValueError(f"{name} : valeur attendue entre {low:g} et {high:g}.")
    if integer and val != int(val):
        raise ValueError(f"{name} : saisir un nombre entier.")
    return int(val) if integer else val


def text(value, name, limit=120, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > limit:
        raise ValueError(f"{name} : texte {'facultatif, ' if empty else ''}limité à {limit} caractères.")
    return value.strip()


def new_project(name="Nouveau projet", demo=False):
    return {
        "id": uid(), "name": name, "location": "Site fictif" if demo else "",
        "start": "2026-11-02" if demo else date.today().isoformat(), "budget_limit": 9000000.0, "deadline": 45,
        "fixed": 250000.0, "reserve": 10.0, "spent": 0.0, "waste": 5.0,
        "geometry": {k: v[1] for k, v in GEOMETRY.items()},
        "prices": dict(excavation=3000., concrete=90000., walls=8000., plaster=2500., roof=10000.),
        "tasks": [
            dict(id="T1", name="Préparer le terrain", item="excavation", workers=3, rate=2., wage=6000., pred="", progress=100. if demo else 0.),
            dict(id="T2", name="Réaliser le béton", item="concrete", workers=4, rate=1., wage=6000., pred="T1", progress=50. if demo else 0.),
            dict(id="T3", name="Monter les murs", item="walls", workers=4, rate=4., wage=6000., pred="T2", progress=0.),
            dict(id="T4", name="Appliquer les enduits", item="plaster", workers=3, rate=6., wage=6000., pred="T3", progress=0.),
            dict(id="T5", name="Poser la couverture", item="roof", workers=3, rate=5., wage=6000., pred="T3", progress=0.),
        ],
    }


def quantities(project):
    g = project["geometry"]
    area = g["length"] * g["width"]
    wall = 2 * (g["length"] + g["width"]) * g["height"] * (1 - g["openings"] / 100)
    return dict(excavation=area * g["depth"],
                concrete=area * g["slab"] + g["pads"] * g["pad_l"] * g["pad_w"] * g["pad_h"],
                walls=wall, plaster=2 * wall, roof=area * g["roof_factor"])


def normalize_tasks(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 30:
        raise ValueError("Prévoir entre 1 et 30 tâches.")
    tasks = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Une ligne de tâche est invalide.")
        t = {k: row.get(k) for k in TASK_COLUMNS}
        for key, label in [("id", "Identifiant"), ("name", "Nom de tâche"), ("item", "Poste")]:
            t[key] = text(t[key], label, 80)
        if not t["id"].isascii() or not all(c.isalnum() or c in "_-" for c in t["id"]):
            raise ValueError("Identifiant de tâche : lettres ASCII, chiffres, _ ou - uniquement.")
        if t["item"] not in ITEMS:
            raise ValueError("Poste de quantité inconnu.")
        t["pred"] = text("" if t["pred"] is None else t["pred"], "Prédécesseurs", 300, empty=True)
        t["pred"] = ",".join(dict.fromkeys(x.strip() for x in t["pred"].split(",") if x.strip()))
        t["workers"] = number(t["workers"], "Effectif", 1, 100, True)
        t["rate"] = number(t["rate"], "Rendement", .01, 10000)
        t["wage"] = number(t["wage"], "Salaire journalier", 0, 1e7)
        t["progress"] = number(t["progress"], "Avancement", 0, 100)
        tasks.append(t)
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Chaque identifiant de tâche doit être unique.")
    # Un poste = une tâche : aucune quantité ne peut être comptée deux fois.
    if len(set(t["item"] for t in tasks)) != len(tasks):
        raise ValueError("Chaque poste ne peut être affecté qu'à une seule tâche.")
    for t in tasks:
        for pred in filter(None, t["pred"].split(",")):
            if pred not in ids:
                raise ValueError(f"{t['id']} : prédécesseur {pred} inconnu.")
    topological(tasks)
    return tasks


def topological(tasks):
    pending, done, ordered = list(tasks), set(), []
    while pending:
        ready = [t for t in pending if set(filter(None, t["pred"].split(","))) <= done]
        if not ready:
            raise ValueError("Les dépendances forment un cycle : revoir les prédécesseurs.")
        for t in ready:
            ordered.append(t); done.add(t["id"]); pending.remove(t)
    return ordered


def validate_project(source):
    if not isinstance(source, dict):
        raise ValueError("Le projet doit être un objet JSON.")
    p = copy.deepcopy(source)
    p["id"] = text(p.get("id"), "Identifiant du projet", 60)
    p["name"] = text(p.get("name"), "Nom du projet", 100)
    p["location"] = text(p.get("location", ""), "Localisation", 160, True)
    try:
        p["start"] = date.fromisoformat(p["start"]).isoformat()
    except (TypeError, ValueError, KeyError):
        raise ValueError("Date de début invalide (AAAA-MM-JJ).") from None
    for k, lo, hi, integer in [("budget_limit", 0, 1e12, False), ("deadline", 1, 3650, True),
                               ("fixed", 0, 1e11, False), ("reserve", 0, 100, False),
                               ("spent", 0, 1e12, False), ("waste", 0, 100, False)]:
        p[k] = number(p.get(k), k, lo, hi, integer)
    if not isinstance(p.get("geometry"), dict) or not isinstance(p.get("prices"), dict):
        raise ValueError("Géométrie ou prix manquants.")
    p["geometry"] = {k: number(p["geometry"].get(k), label, lo, hi, k == "pads")
                     for k, (label, _, lo, hi) in GEOMETRY.items()}
    p["prices"] = {k: number(p["prices"].get(k), "Prix " + ITEMS[k][0], 0, 1e9) for k in ITEMS}
    p["tasks"] = normalize_tasks(p.get("tasks"))
    if set(t["item"] for t in p["tasks"]) != set(ITEMS):
        raise ValueError("Il faut une tâche pour chacun des cinq postes, sans doublon.")
    # Borner le calendrier pour éviter des allocations de mémoire démesurées.
    q = quantities(p)
    durations = [math.ceil(q[t["item"]] / (t["workers"] * t["rate"])) for t in p["tasks"]]
    if sum(durations) > 3650:
        raise ValueError("Planning supérieur à 3 650 jours : augmenter les rendements ou les effectifs.")
    return p


def schedule(project, productivity=1.0, wage_factor=1.0):
    q = quantities(project)
    result, ends = [], {}
    for t in topological(project["tasks"]):
        d = math.ceil(q[t["item"]] / (t["workers"] * t["rate"] * productivity))
        start = max((ends[x] for x in filter(None, t["pred"].split(","))), default=0)
        end = start + d
        ends[t["id"]] = end
        result.append(dict(**t, quantity=q[t["item"]], duration=d, start=start, end=end,
                           labor=d * t["workers"] * t["wage"] * wage_factor))
    finish = max(ends.values(), default=0)
    latest = {}
    for t in reversed(result):
        successors = [x for x in result if t["id"] in x["pred"].split(",")]
        latest_end = min((latest[x["id"]] for x in successors), default=finish)
        latest[t["id"]] = latest_end - t["duration"]
        t["slack"] = latest[t["id"]] - t["start"]
    return result


def estimate(project, price_factor=1.0, productivity=1.0, wage_factor=1.0):
    q, rows = quantities(project), schedule(project, productivity, wage_factor)
    materials = {k: q[k] * (1 + project["waste"] / 100) * project["prices"][k] * price_factor for k in ITEMS}
    labor = sum(t["labor"] for t in rows)
    direct = sum(materials.values()) + labor + project["fixed"]
    provision = direct * project["reserve"] / 100
    costs = {t["id"]: materials[t["item"]] + t["labor"] for t in rows}
    cost_sum = sum(costs.values())
    progress = sum(costs[t["id"]] * t["progress"] / 100 for t in rows) / cost_sum * 100 if cost_sum else 0.
    return dict(quantities=q, tasks=rows, materials=materials, labor=labor, direct=direct,
                provision=provision, total=direct + provision, duration=max(t["end"] for t in rows),
                progress=progress, costs=costs)


def daily_plan(project):
    e = estimate(project)
    days = e["duration"]
    workers, cash = np.zeros(days), np.zeros(days)
    for t in e["tasks"]:
        if t["duration"]:
            workers[t["start"]:t["end"]] += t["workers"]
            cash[t["start"]:t["end"]] += e["costs"][t["id"]] / t["duration"]
    if days:
        cash += (project["fixed"] + e["provision"]) / days
    return pd.DataFrame({"jour": np.arange(1, days + 1), "effectif": workers.astype(int),
                         "depense_planifiee": cash, "cumul_planifie": np.cumsum(cash)})


def simulate(project, count, seed, price_bounds, yield_bounds):
    """Deux facteurs triangulaires indépendants, chacun partagé par les cinq postes."""
    count = number(count, "Nombre de tirages", 1000, 50000, True)
    seed = number(seed, "Graine", 0, 2**32-1, True)
    for bounds in (price_bounds, yield_bounds):
        if len(bounds) != 3 or not all(math.isfinite(float(x)) for x in bounds):
            raise ValueError("Bornes triangulaires invalides.")
        lo, mode, hi = bounds
        if not 0.1 <= lo <= mode <= hi <= 3:
            raise ValueError("Facteurs attendus entre 0,1 et 3, avec minimum ≤ mode ≤ maximum.")
    rng = np.random.default_rng(seed)
    def draw(bounds):
        return np.full(count, bounds[0]) if bounds[0] == bounds[2] else rng.triangular(*bounds, size=count)
    prices, yields = draw(price_bounds), draw(yield_bounds)
    q, ends, labor = quantities(project), {}, np.zeros(count)
    for t in topological(project["tasks"]):
        d = np.ceil(q[t["item"]] / (t["workers"] * t["rate"] * yields)).astype(int)
        predecessors = list(filter(None, t["pred"].split(",")))
        begin = np.maximum.reduce([ends[x] for x in predecessors]) if predecessors else np.zeros(count, dtype=int)
        ends[t["id"]] = begin + d
        labor += d * t["workers"] * t["wage"]
    material = sum(estimate(project)["materials"].values())
    cost = material * prices + labor + project["fixed"]
    duration = np.maximum.reduce(list(ends.values()))
    return pd.DataFrame(dict(cout_simule=cost, duree_simulee=duration,
                             facteur_prix=prices, facteur_rendement=yields))


def db_path():
    custom = os.environ.get("CIVIL360_DATA_DIR")
    folder = Path(custom) if custom else Path.home() / "Documents" / "Codex" / "data" / "civil360"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "civil360.sqlite3"


@contextmanager
def connection():
    db = sqlite3.connect(db_path(), timeout=10)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO workspace VALUES (1,0,'[]')")
        db.commit()
        with db:
            yield db
    finally:
        db.close()


def load_store():
    with connection() as db:
        rev, payload = db.execute("SELECT revision,payload FROM workspace WHERE id=1").fetchone()
    return [validate_project(p) for p in json.loads(payload)], rev


def save_store(projects, revision):
    payload = json.dumps([validate_project(p) for p in projects], ensure_ascii=False, allow_nan=False)
    with connection() as db:
        cur = db.execute("UPDATE workspace SET payload=?,revision=revision+1 WHERE id=1 AND revision=?", (payload, revision))
        if cur.rowcount != 1:
            raise ValueError("Le registre a changé dans un autre onglet. Rechargez les projets avant de réessayer.")
    return revision + 1


def backup_bytes(projects):
    return json.dumps(dict(app="Civil360", schema=1, projects=projects), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def read_backup(raw):
    if len(raw) > 5_000_000:
        raise ValueError("Sauvegarde limitée à 5 Mo.")
    try:
        obj = json.loads(raw)
        if obj.get("app") != "Civil360" or obj.get("schema") != 1:
            raise ValueError("Format de sauvegarde Civil360 non reconnu.")
        projects = obj["projects"]
        if not isinstance(projects, list) or not 1 <= len(projects) <= 50:
            raise ValueError("La sauvegarde doit contenir de 1 à 50 projets.")
        result = []
        for source in projects:
            p = validate_project(source)
            p["id"] = uid(); p["name"] = p["name"][:88] + " (copie)"
            result.append(p)
        return result
    except (json.JSONDecodeError, UnicodeError, KeyError, AttributeError, TypeError):
        raise ValueError("Sauvegarde JSON invalide ou incomplète.") from None


def csv_bytes(frame):
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def task_frame(project, calculated=False):
    if not calculated:
        return pd.DataFrame(project["tasks"], columns=TASK_COLUMNS)
    e = estimate(project)
    return pd.DataFrame([dict(Tâche=t["id"], Libellé=t["name"], Poste=ITEMS[t["item"]][0],
                              Quantité=t["quantity"], Unité=ITEMS[t["item"]][1], Effectif=t["workers"],
                              Durée=t["duration"], Début_J=t["start"]+1, Fin_J=t["end"],
                              Marge_j=t["slack"], Avancement_pct=t["progress"], Main_oeuvre=t["labor"])
                         for t in e["tasks"]])


def quantity_frame(project):
    q = quantities(project)
    return pd.DataFrame([dict(Poste=label, Quantité=q[k], Unité=unit, Formule=formula) for k, (label, unit, formula) in ITEMS.items()])


def budget_frame(project):
    e = estimate(project)
    return pd.DataFrame([dict(Poste=ITEMS[t["item"]][0], Quantité=t["quantity"],
                              Unité=ITEMS[t["item"]][1], Prix_unitaire=project["prices"][t["item"]],
                              Fournitures_avec_pertes=e["materials"][t["item"]],
                              Main_oeuvre=t["labor"], Total=e["costs"][t["id"]]) for t in e["tasks"]])


def read_tasks(raw, extension):
    if len(raw) > 2_000_000:
        raise ValueError("Import limité à 2 Mo.")
    try:
        if extension == "xlsx":
            df = pd.read_excel(io.BytesIO(raw), sheet_name=0, keep_default_na=False)
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig", decimal=",", keep_default_na=False)
        if set(df.columns) != set(TASK_COLUMNS):
            raise ValueError("Utilisez le modèle de tâches : " + ", ".join(TASK_COLUMNS))
        return normalize_tasks(df.to_dict("records"))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Fichier CSV/Excel illisible. Vérifiez le format et le modèle.") from exc


def excel_bytes(project):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet, frame in [("Quantites", quantity_frame(project)), ("Budget", budget_frame(project)),
                             ("Planning", task_frame(project, True)), ("Effectifs_finances", daily_plan(project)),
                             ("Modele_taches", task_frame(project))]:
            # Force user strings to literal cells, including a leading '='.
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for row in ws:
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(42, max(14, max(len(str(c.value or "")) for c in col)+2))
    return out.getvalue()


def money(value):
    return f"{value:,.0f}".replace(",", " ")


def fmt(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def pdf_bytes(project, demo=False):
    e = estimate(project)
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=(21*cm, 29.7*cm), rightMargin=1.6*cm,
                            leftMargin=1.6*cm, topMargin=1.8*cm, bottomMargin=1.6*cm,
                            title="Civil360 - " + project["name"], author="Civil360")
    styles = getSampleStyleSheet()
    styles["Heading2"].keepWithNext = True
    styles.add(ParagraphStyle(name="CivilTitle", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.HexColor(PURPLE), spaceAfter=14))
    styles.add(ParagraphStyle(name="CivilBody", fontName="Helvetica", fontSize=9, leading=13, spaceAfter=8))
    styles.add(ParagraphStyle(name="CivilCell", fontSize=8, leading=11, alignment=TA_LEFT, wordWrap="CJK"))
    def para(s, style="CivilBody"):
        return Paragraph(html.escape(str(s)), styles[style])
    story = [para("CIVIL360 / NOTE DE PROJET", "CivilTitle"), para(project["name"], "Heading1"),
             para(("EXEMPLE FICTIF - " if demo else "DONNEES SAISIES - ") + date.today().isoformat()),
             para("Localisation : " + (project["location"] or "Non renseignée")),
             para("Bâtiment rectangulaire de plain-pied. Estimation pédagogique ; aucune validation technique ou réglementaire.")]
    def table(headers, rows, widths):
        data = [[para(x, "CivilCell") for x in headers]] + [[para(x, "CivilCell") for x in row] for row in rows]
        t = Table(data, colWidths=[x*cm for x in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9E2FC")),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F4FA")]),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8),
                               ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
        story.extend([t, Spacer(1, .4*cm)])
    table(["Indicateur", "Valeur"], [
        ["Coût direct", money(e["direct"])+" FCFA"], ["Provision", money(e["provision"])+" FCFA"],
        ["Enveloppe estimée", money(e["total"])+" FCFA"], ["Budget disponible", money(project["budget_limit"])+" FCFA"],
        ["Durée planifiée", str(e["duration"])+" jours"], ["Avancement saisi pondéré", fmt(e["progress"],1)+" %"],
        ["Dépenses constatées saisies", money(project["spent"])+" FCFA"]], [9.5,8.3])
    story.append(para("Hypothèses géométriques", "Heading2"))
    table(["Paramètre", "Valeur"], [[v[0], fmt(project["geometry"][k])] for k,v in GEOMETRY.items()], [12.5,5.3])
    story.append(para("Quantités et budget", "Heading2"))
    table(["Poste", "Quantité", "Fournitures", "Main-d'oeuvre"], [
        [r["Poste"], fmt(r["Quantité"])+" "+r["Unité"], money(r["Fournitures_avec_pertes"]), money(r["Main_oeuvre"])]
        for r in budget_frame(project).to_dict("records")], [5.3,3.5,4.5,4.5])
    story.append(para("Fournitures en FCFA ; prix hors main-d'oeuvre. Pertes : " + fmt(project["waste"],1)+" %. Frais fixes : "+money(project["fixed"])+" FCFA. Provision : "+fmt(project["reserve"],1)+" %."))
    story.append(para("Planning et équipes", "Heading2"))
    table(["Tâche", "Équipe", "Rendement / ouvrier / jour", "Durée", "Début / fin"], [
        [t["id"]+" - "+t["name"], str(t["workers"]), fmt(t["rate"])+" "+ITEMS[t["item"]][1], str(t["duration"])+" j", f"J{t['start']+1} / J{t['end']}"]
        for t in e["tasks"]], [5.8,1.7,4.1,2.2,4.0])
    story.append(para("Méthodes et limites", "Heading2"))
    for line in [
        "Volume de béton = dalle + semelles uniquement. Poteaux, poutres, ferraillage, cloisons et réseaux exclus. Dimensions saisies, jamais dimensionnées par l'application.",
        "Maçonnerie : murs extérieurs nets d'ouvertures. Enduits sur deux faces. Toiture : coefficient fourni par l'utilisateur.",
        "Durée = arrondi supérieur de quantité / (effectif × rendement). Dépendances fin-début, sans week-end ni capacité d'équipe partagée. Les équipes de tâches parallèles s'additionnent.",
        "Avancement global pondéré par les coûts de postes hors frais fixes et provision. Dépenses constatées : montant cumulé déclaré, sans justificatifs.",
        "La courbe financière est un étalement théorique des coûts ; elle n'est pas un relevé bancaire. La provision est une enveloppe, pas une dépense certaine.",
        "Les simulations utilisent deux facteurs triangulaires indépendants (prix et rendement), partagés par les postes. Aucun aléa climatique ni panne n'est inclus. Le dépassement est strict (>). Les résultats dépendent des hypothèses.",
        "Cette note résume le scénario de base enregistré. Les variantes temporaires et les tirages Monte-Carlo s'exportent séparément dans l'application.",
    ]:
        story.append(para(line))
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(1.6*cm, .9*cm, "Civil360 / étude pédagogique")
        canvas.drawRightString(19.4*cm, .9*cm, f"Page {document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()


CSS = """
<style>
:root {--c-ink:#282238;--c-muted:#706980;--c-purple:#6845E8;--c-line:#DFD9EB;}
.stApp {background:var(--c-bg,#F4F1FB);color:var(--c-ink);}
[data-testid="stApp"],input,textarea,button,select,[data-testid="stMarkdownContainer"]{font-family:'Segoe UI',sans-serif;}
[data-testid="stMainBlockContainer"]{max-width:1530px;padding-top:2.2rem;padding-bottom:3rem;}
[data-testid="stHeader"]{background:transparent;}
h1,h2,h3 {color:var(--c-ink);letter-spacing:-.035em!important;font-family:'Segoe UI',sans-serif!important;}
h1 {font-size:2.25rem!important;font-weight:750!important;}h3{font-size:1.15rem!important;}
[data-testid="stSidebar"]{background:#231B3A;color:#F1ECFF;border-right:1px solid #3B2E57;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stNavLink"]{color:#E7DFF8!important;}
[data-testid="stSidebar"] [data-testid="stNavLink"][aria-current="page"]{background:#4C357A;color:#FFF!important;}
[data-testid="stSidebar"] button[kind="secondary"] {background:#38284F;color:#F1ECFF;border-color:#76638E;}
[data-testid="stSidebar"] button p,[data-testid="stSidebar"] summary p,[data-testid="stSidebar"] summary span {color:#F1ECFF!important;}
button[kind="primary"],button[kind="primaryFormSubmit"] {background:#6845E8!important;border-color:#6845E8!important;color:#FFF!important;}
button[kind="primary"] p,button[kind="primaryFormSubmit"] p{color:#FFF!important;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stFormSubmitButton"] button {border-radius:11px;min-height:44px;white-space:normal;}
button p{white-space:normal!important;overflow-wrap:anywhere;}button:focus-visible,a:focus-visible{outline:3px solid #A37BFF!important;outline-offset:3px;}
[data-testid="stForm"]{background:#FFFEFF;border:1px solid #DED6EB;border-radius:18px;padding:22px;}
.mast {display:flex;align-items:center;justify-content:space-between;gap:18px;padding:8px 0 23px;border-bottom:1px solid var(--c-line);margin-bottom:25px;}
.wordmark {font:750 24px 'Segoe UI',sans-serif;letter-spacing:-1px;color:#282238;}.wordmark b{color:#6845E8;}
.mast small,.eyebrow {font:650 11px 'Segoe UI',sans-serif;letter-spacing:1.8px;text-transform:uppercase;color:#706980;}
.mast .tag {font-size:11px;background:#E8DEF9;color:#493271;padding:9px 13px;border-radius:30px;letter-spacing:.5px;}
.sidebrand{font:760 31px 'Segoe UI',sans-serif;letter-spacing:-1px;color:white;margin:14px 0 0;}.sidebrand b{color:#D7F86B;}
.sidetag {color:#BEAED6;font-size:12px;margin-bottom:22px;}
.hero {position:relative;overflow:hidden;background:linear-gradient(120deg,#39225E 0%,#6845B6 80%,#8164C7 100%);border-radius:24px;padding:32px 35px;color:#FFF;margin:6px 0 22px;display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px;}
.hero>div{min-width:0;}.hero .eyebrow{color:#DED2F5;font-size:10px;}.hero h2{font-size:35px!important;line-height:1.2;color:#FFF!important;margin:15px 0;padding-bottom:3px;max-width:100%;overflow-wrap:anywhere;}
.hero p{color:#E7DCF8;max-width:600px;font-size:14px;line-height:1.7;}.hero .pill{display:inline-block;background:#D7F86B;color:#303B11;border-radius:30px;padding:6px 12px;font-size:11px;font-weight:700;}
.building-scene{position:relative;min-height:225px;display:flex;align-items:center;justify-content:center;}
.orbit{position:absolute;width:235px;height:235px;border:1px solid #A48BC8;border-radius:50%;}.orbit:after{content:'';position:absolute;inset:22px;border:1px dashed #9C83C0;border-radius:50%;}
.building{position:relative;width:166px;height:126px;background:#E7DDF8;transform:skewY(-12deg);box-shadow:20px 18px 0 #322046;border-top:12px solid #D7F86B;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:15px;}
.building i{display:block;background:#8866BA;border-bottom:4px solid #B69CD9;border-radius:2px;}.building i:nth-child(3n){background:#544264;}
.scene-caption{position:absolute;bottom:0;right:0;color:#E2D4F2;font-size:10px;letter-spacing:1px;}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:5px 0 27px;}
.metric{background:#FFF;border:1px solid #E2DCEC;border-radius:17px;padding:20px;min-width:0;position:relative;overflow:hidden;}
.metric:after{content:'';position:absolute;right:-12px;top:-12px;width:55px;height:55px;background:#EEE7FB;border-radius:50%;}
.metric>*{position:relative;z-index:1;}
.metric:nth-child(2):after{background:#E9F4CE;}.metric:nth-child(3):after{background:#E0ECFC;}.metric:nth-child(4):after{background:#FAE9D4;}
.metric label{display:block;font-size:12px;color:#706980;margin-bottom:11px;}.metric strong{font-size:27px;font-weight:760;line-height:1.2;display:block;color:#282238;overflow-wrap:anywhere;font-variant-numeric:tabular-nums;}.metric small{display:block;font-size:11px;color:#706980;margin-top:9px;}
.hint{border-left:3px solid #8D68E6;background:#EBE4F7;border-radius:0 12px 12px 0;padding:16px 19px;color:#4A3B65;font-size:13px;line-height:1.6;margin:12px 0 20px;}
[class*="st-key-card_"]{background:#FFF;border:1px solid #E2DCEC;border-radius:19px;padding:20px;transition:transform .18s ease,box-shadow .18s ease;}
[class*="st-key-card_"]:hover{box-shadow:0 12px 28px #4E337214;transform:translateY(-2px);}
.project-icon{display:inline-flex;width:43px;height:43px;align-items:center;justify-content:center;background:#EDE5FC;color:#6845E8;border-radius:12px;font-size:23px;margin-bottom:10px;}
.project-row{display:flex;gap:15px;justify-content:space-between;align-items:center;margin-top:15px;color:#706980;font-size:12px;}
.ring{width:66px;height:66px;flex:none;border-radius:50%;display:grid;place-items:center;position:relative;background:conic-gradient(#6845E8 var(--p),#EBE5F4 0);}
.ring:before{content:'';position:absolute;inset:7px;background:white;border-radius:50%;}.ring span{position:relative;font-size:14px;font-weight:750;color:#493272;}
.foot{font-size:11px;color:#706980;border-top:1px solid #DFD9EB;margin-top:35px;padding-top:20px;}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#706980;margin-bottom:10px;}
.stepbadge{display:inline-block;border:1px solid #CFC0EB;color:#6845E8;border-radius:24px;padding:5px 11px;font-size:11px;margin-bottom:12px;}
@media(max-width:1000px){.hero{padding:25px;}.hero h2{font-size:28px!important;}.metric{padding:16px;}.metric strong{font-size:23px;}}
@media(max-width:640px){[data-testid="stMainBlockContainer"]{padding:3rem 1rem 2rem;}.mast{align-items:flex-start;}.mast small{display:none;}.mast .tag{font-size:10px;max-width:125px;text-align:center;}.hero{grid-template-columns:1fr;padding:24px;border-radius:18px;}.hero h2{font-size:29px!important;}.building-scene{min-height:180px;}.orbit{width:180px;height:180px;}.building{width:135px;height:100px;gap:8px;padding:10px;}.scene-caption{bottom:0;}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;}.metric{padding:14px;}.metric strong{font-size:22px;}.metric label{font-size:11px;}.metric small{font-size:10px;}h1{font-size:1.8rem!important;}[data-testid="stForm"]{padding:16px;}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important;}}
</style>
"""


def esc(value):
    return html.escape(str(value))


def header(title, subtitle):
    st.html(f'<div class="mast"><div class="wordmark">Civil<b>360</b> <small> / bureau de projet</small></div><div class="tag">{"EXEMPLE FICTIF" if st.session_state.demo else "ESPACE PERSONNEL"}</div></div>')
    st.title(title)
    st.caption(subtitle)


def metrics(items):
    st.html('<div class="metrics">'+''.join(f'<div class="metric"><label>{esc(a)}</label><strong>{esc(b)}</strong><small>{esc(c)}</small></div>' for a,b,c in items)+'</div>')


def hint(message):
    st.html('<div class="hint">'+esc(message)+'</div>')


def figure_layout(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=25,b=35),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, family="Segoe UI", size=12),
                      colorway=[PURPLE, BLUE, ORANGE, "#79A33D", "#B06CA9"],
                      legend=dict(orientation="h",y=-.25,x=0),
                      hoverlabel=dict(bgcolor="white",font_color=INK))
    fig.update_xaxes(gridcolor="#E1DAEB", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#E1DAEB", zeroline=False, automargin=True)
    return fig


def show_chart(fig, alt, height=330, key=None):
    st.plotly_chart(figure_layout(fig, height), width="stretch", alt=alt, key=key,
                    config=dict(displaylogo=False, responsive=True, scrollZoom=False))


def projects():
    return st.session_state.demo_projects if st.session_state.demo else st.session_state.projects


def current():
    data = projects()
    return next((p for p in data if p["id"] == st.session_state.selected), data[0] if data else None)


def commit(data, selected=None):
    checked = [validate_project(p) for p in data]
    if len(checked) > 50:
        raise ValueError("Le registre est limité à 50 projets.")
    if st.session_state.demo:
        st.session_state.demo_projects = checked
    else:
        st.session_state.revision = save_store(checked, st.session_state.revision)
        st.session_state.projects = checked
    if selected is not None:
        st.session_state.selected = selected
    st.session_state.flash = "Modifications enregistrées." + (" Exemple fictif uniquement." if st.session_state.demo else "")


def update_project(p):
    commit([p if x["id"] == p["id"] else x for x in projects()], p["id"])
    st.rerun()


def select_project(identity):
    st.session_state.selected = identity
    st.session_state.risk_result = None


def enter_demo():
    if not st.session_state.demo_projects:
        p = new_project("École des Horizons", True)
        p["spent"] = 1250000.
        st.session_state.demo_projects = [p]
    st.session_state.demo = True
    select_project(st.session_state.demo_projects[0]["id"])


def leave_demo():
    st.session_state.demo = False
    select_project(st.session_state.projects[0]["id"] if st.session_state.projects else None)


def require_project():
    p = current()
    if p is None:
        hint("Commencez par créer votre projet dans « Projets », ou chargez l’exemple fictif depuis le menu.")
        st.page_link(PAGES[1], label="Créer mon premier projet", icon=":material/add_circle:")
    return p


def draw_plan(project):
    g = project["geometry"]; L, W = g["length"], g["width"]
    fig = go.Figure()
    fig.add_shape(type="rect",x0=0,y0=0,x1=L,y1=W,fillcolor="#E4D9F9",line=dict(color=PURPLE,width=4))
    fig.add_trace(go.Scatter(x=[0,L,L,0,0],y=[0,0,W,W,0],mode="lines",line=dict(color=PURPLE,width=3),
                             hovertemplate="X : %{x:.2f} m<br>Y : %{y:.2f} m<extra>Contour extérieur</extra>",name="Emprise"))
    fig.add_annotation(x=L/2,y=W/2,text=f"{fmt(L*W)} m²<br><span style='font-size:11px'>Emprise rectangulaire</span>",showarrow=False,font=dict(size=20,color="#533394"))
    fig.add_annotation(x=L/2,y=-W*.16,text=f"L = {fmt(L)} m",showarrow=False)
    fig.add_annotation(x=-L*.17,y=W/2,text=f"l = {fmt(W)} m",showarrow=False,textangle=-90)
    fig.update_xaxes(range=[-L*.28,L*1.12],visible=False)
    fig.update_yaxes(range=[-W*.28,W*1.15],visible=False,scaleanchor="x",scaleratio=1)
    fig.update_layout(showlegend=False)
    show_chart(fig,"Schéma coté de l’emprise rectangulaire, sans implantation de fondations",300)


def cash_chart(p):
    frame = daily_plan(p)
    fig = go.Figure(go.Scatter(x=[0]+frame.jour.tolist(),y=[0]+frame.cumul_planifie.tolist(),mode="lines",fill="tozeroy",line=dict(color=PURPLE,width=3),fillcolor="rgba(104,69,232,.10)",name="Enveloppe répartie"))
    fig.add_hline(y=p["budget_limit"],line_dash="dot",line_color="#9B6227",annotation_text="Budget disponible",annotation_position="top left")
    fig.update_xaxes(title="Jour du projet"); fig.update_yaxes(title="FCFA cumulés")
    show_chart(fig,"Courbe financière théorique cumulée et budget disponible")


def render_dashboard():
    header("Votre projet, sous tous les angles.", "Quantités, argent et temps : une seule base, des décisions plus lisibles.")
    p = current()
    name = p["name"] if p else "Du premier croquis au planning."
    st.html(f'''<div class="hero"><div><span class="eyebrow">LE BUREAU DE PROJET / CIVIL360</span><h2>{esc(name)}</h2><p>Reliez les dimensions de votre bâtiment à ses matériaux, à ses équipes et à son budget. Modifiez une hypothèse, observez ses conséquences.</p><span class="pill">{'Simulation pédagogique · données fictives' if st.session_state.demo else 'Bâtiment de plain-pied · étude simplifiée'}</span></div><div class="building-scene" aria-hidden="true"><div class="orbit"></div><div class="building">{''.join('<i></i>' for _ in range(8))}</div><span class="scene-caption">SCHÉMA D’IDENTITÉ / NON CONTRACTUEL</span></div></div>''')
    if p is None:
        metrics([("Projets",str(len(projects())),"Créez votre premier dossier"),("Quantités","5 postes","Volumes et surfaces"),("Planning","Lié","Équipes et dépendances"),("Risques","Monte-Carlo","Coût et durée")])
        with st.container(horizontal=True):
            st.button("Explorer l’exemple fictif", type="primary", icon=":material/play_arrow:",on_click=enter_demo,key="home_demo")
            st.page_link(PAGES[1],label="Créer mon projet",icon=":material/add_circle:")
        hint("L’exemple est séparé de vos projets. Les prix proposés sont des hypothèses de travail à remplacer, pas des tarifs locaux vérifiés.")
        return
    e = estimate(p)
    metrics([("Enveloppe estimée",money(e["total"]),"FCFA · provision comprise"),
             ("Durée planifiée",str(e["duration"])+" j", "Équipes distinctes en parallèle"),
             ("Avancement saisi",fmt(e["progress"],1)+" %","Pondéré par coût des postes"),
             ("Marge de budget",money(p["budget_limit"]-e["total"]),"FCFA disponibles après estimation")])
    left,right = st.columns([1.1,1])
    with left, st.container(border=True):
        st.subheader("Le projet en plan")
        draw_plan(p)
        st.caption("Vue de l’emprise ; aucun plan d’exécution ni dimensionnement.")
    with right, st.container(border=True):
        st.subheader("Où va l’enveloppe ?")
        fig=go.Figure(go.Pie(labels=["Fournitures","Main-d’œuvre","Frais fixes","Provision"], values=[sum(e["materials"].values()),e["labor"],p["fixed"],e["provision"]],hole=.73,marker_colors=[PURPLE,BLUE,ORANGE,"#A2C85C"],textinfo="percent",sort=False))
        show_chart(fig,"Répartition de l’enveloppe entre fournitures, main-d’œuvre, frais fixes et provision",300)
    st.subheader("Vos projets")
    render_cards()
    st.subheader("Trajectoire financière prévue")
    cash_chart(p)
    if e["total"] > p["budget_limit"] or e["duration"] > p["deadline"]:
        hint("Le scénario dépasse au moins un objectif saisi. Essayez une variante de prix ou d’effectif ; cela ne signifie pas que le chantier réel aura ce résultat.")


def render_cards():
    data = projects()
    for start in range(0,len(data),3):
        cols=st.columns(3)
        for col,p in zip(cols,data[start:start+3]):
            e=estimate(p)
            with col,st.container(key="card_"+p["id"]):
                st.html('<div class="project-icon" aria-hidden="true">▥</div>')
                st.subheader(p["name"])
                st.caption(p["location"] or "Localisation à préciser")
                st.html(f'<div class="project-row"><div>{money(e["total"])} FCFA<br>{e["duration"]} jours planifiés</div><div class="ring" style="--p:{e["progress"]:.2f}%"><span>{e["progress"]:.0f}%</span></div></div>')
                st.button("Projet actif" if current()["id"]==p["id"] else "Ouvrir ce projet",key="open_"+p["id"],icon=":material/arrow_outward:",on_click=select_project,args=(p["id"],),width="stretch",disabled=current()["id"]==p["id"])


def render_projects():
    header("Projets", "Créez plusieurs dossiers et choisissez celui que vous souhaitez étudier.")
    if projects():
        render_cards()
    with st.expander("Créer un projet",expanded=not bool(projects()),icon=":material/add_circle:"):
        with st.form("create_project"):
            name=st.text_input("Nom du projet",max_chars=100,placeholder="Ex. Pavillon universitaire")
            location=st.text_input("Localisation",max_chars=160)
            st.caption("Un modèle rectangulaire paramétrable sera créé avec des hypothèses initiales à vérifier.")
            if st.form_submit_button("Créer le projet",type="primary"):
                try:
                    p=new_project(text(name,"Nom du projet"));p["location"]=location
                    commit(projects()+[p],p["id"]);st.rerun()
                except ValueError as exc: st.error(str(exc))
    p=current()
    if not p: return
    st.subheader("Identité et objectifs du projet actif")
    with st.form("project_meta_"+p["id"]):
        name=st.text_input("Nom",value=p["name"],max_chars=100)
        location=st.text_input("Localisation du projet",value=p["location"],max_chars=160)
        a,b=st.columns(2)
        start=a.date_input("Début du projet",value=date.fromisoformat(p["start"]))
        limit=b.number_input("Budget disponible (FCFA)",0.,1e12,p["budget_limit"],step=100000.)
        deadline=st.number_input("Délai objectif (jours)",1,3650,p["deadline"])
        if st.form_submit_button("Enregistrer les objectifs",type="primary"):
            candidate=copy.deepcopy(p);candidate.update(name=name,location=location,start=start.isoformat(),budget_limit=limit,deadline=deadline)
            try: update_project(candidate)
            except ValueError as exc: st.error(str(exc))
    with st.expander("Dupliquer le scénario"):
        st.caption("La copie conserve les dimensions, les prix et les tâches ; elle pourra être modifiée séparément.")
        if st.button("Créer une copie du projet",icon=":material/content_copy:"):
            candidate=copy.deepcopy(p);candidate["id"]=uid();candidate["name"]=candidate["name"][:88]+" (variante)"
            try: commit(projects()+[candidate],candidate["id"]);st.rerun()
            except ValueError as exc: st.error(str(exc))
    with st.expander("Supprimer ce projet"):
        confirmed=st.checkbox("Je souhaite supprimer ce projet de cet espace",key="delete_"+p["id"])
        if st.button("Supprimer le projet",disabled=not confirmed,icon=":material/delete:"):
            try:
                remaining=[x for x in projects() if x["id"]!=p["id"]]
                commit(remaining,remaining[0]["id"] if remaining else None);st.session_state.selected=remaining[0]["id"] if remaining else None;st.rerun()
            except ValueError as exc: st.error(str(exc))


def render_quantities():
    header("Quantités et matériaux", "Une géométrie simple, des formules visibles et des résultats vérifiables.")
    p=require_project()
    if not p:return
    with st.form("geometry_"+p["id"]):
        st.subheader("Dimensions du bâtiment")
        cols=st.columns(2);g={}
        for i,(k,(label,_,lo,hi)) in enumerate(GEOMETRY.items()):
            with cols[i%2]:
                g[k]=st.number_input(label,lo,hi,p["geometry"][k],step=1 if k=="pads" else .01,format="%d" if k=="pads" else "%.2f")
        if st.form_submit_button("Recalculer et enregistrer",type="primary",icon=":material/calculate:"):
            candidate=copy.deepcopy(p);candidate["geometry"]=g
            try:update_project(candidate)
            except ValueError as exc:st.error(str(exc))
    hint("Les semelles sont seulement comptabilisées avec les dimensions fournies. Béton des poteaux/poutres, acier, cloisons et réseaux sont exclus. Le décapage n’inclut pas les fouilles de semelles.")
    draw_plan(p)
    st.dataframe(quantity_frame(p),hide_index=True,alt="Quantités calculées avec unités et formules")
    st.download_button("Quantités CSV",csv_bytes(quantity_frame(p)),"Civil360_quantites.csv","text/csv",icon=":material/download:")
    with st.expander("Exemple à vérifier à la calculatrice"):
        st.markdown("Pour **12 × 8 m**, dalle **0,12 m**, **8 semelles de 1 × 1 × 0,40 m** : béton = **14,72 m³**. Murs de **3 m**, ouvertures **20 %** : maçonnerie = **96 m²**, enduits = **192 m²**. Décapage **0,30 m** : **28,80 m³**. Toiture × **1,15** : **110,40 m²**.")


def render_budget():
    header("Budget et variantes", "Distinguez les fournitures, le travail des équipes et l’enveloppe de précaution.")
    p=require_project()
    if not p:return
    with st.form("prices_"+p["id"]):
        st.subheader("Prix unitaires et hypothèses")
        st.caption("Prix de fournitures uniquement ; la main-d’œuvre est calculée séparément. Les valeurs initiales sont fictives.")
        cols=st.columns(2);prices={}
        for i,(k,(label,unit,_)) in enumerate(ITEMS.items()):
            prices[k]=cols[i%2].number_input(f"{label} (FCFA/{unit})",0.,1e9,p["prices"][k],step=100.)
        fixed=cols[1].number_input("Frais fixes (FCFA)",0.,1e11,p["fixed"],step=10000.)
        waste=cols[0].number_input("Majoration des quantités pour pertes (%)",0.,100.,p["waste"],step=1.)
        reserve=cols[1].number_input("Provision pour imprévus (%)",0.,100.,p["reserve"],step=1.)
        spent=st.number_input("Dépenses constatées cumulées (FCFA)",0.,1e12,p["spent"],step=10000.,help="Montant total déclaré ; ne modifie pas l’estimation ni la courbe théorique.")
        if st.form_submit_button("Enregistrer le budget",type="primary"):
            candidate=copy.deepcopy(p);candidate.update(prices=prices,fixed=fixed,waste=waste,reserve=reserve,spent=spent)
            try:update_project(candidate)
            except ValueError as exc:st.error(str(exc))
    e=estimate(p)
    metrics([("Fournitures",money(sum(e["materials"].values())),"FCFA · pertes comprises"),("Main-d’œuvre",money(e["labor"]),"FCFA · effectifs × jours × salaire"),("Provision",money(e["provision"]),"FCFA · marge prévue"),("Total estimé",money(e["total"]),"FCFA · frais fixes compris")])
    st.dataframe(budget_frame(p),hide_index=True,alt="Décomposition des coûts de chaque poste en FCFA")
    cash_chart(p)
    st.caption(f"Dépenses constatées déclarées : {money(p['spent'])} FCFA. Budget disponible restant après ces dépenses : {money(p['budget_limit']-p['spent'])} FCFA. Aucun historique réel n’est déduit de ce montant unique.")
    st.subheader("Comparer une variante temporaire")
    a,b=st.columns(2)
    pf=a.slider("Variation des prix de fournitures (%)",-50,100,10,key="variant_price_"+p["id"])
    yf=b.slider("Variation du rendement (%)",-50,100,0,key="variant_yield_"+p["id"])
    v=estimate(p,1+pf/100,1+yf/100)
    comparison=pd.DataFrame({"Scénario":["Base enregistrée","Variante temporaire"],"Enveloppe_FCFA":[e["total"],v["total"]],"Durée_jours":[e["duration"],v["duration"]]})
    st.dataframe(comparison,hide_index=True,alt="Comparaison du coût et de la durée entre base et variante")
    st.download_button("Comparaison CSV",csv_bytes(comparison),"Civil360_variante.csv","text/csv")
    hint("Le rendement est la quantité réalisée par un ouvrier en une journée. Une hausse du rendement réduit parfois la durée par paliers, car les jours sont arrondis au supérieur. Les équipes sont payées par journée entière.")


def render_planning():
    header("Planning et effectifs", "Modifiez vos équipes et voyez leurs effets sur le temps et les dépenses.")
    p=require_project()
    if not p:return
    hint("Une tâche par poste. Le rendement est exprimé en unité du poste / ouvrier / jour. Prédécesseurs : identifiants séparés par des virgules. Les équipes parallèles sont distinctes ; tous les jours du calendrier sont travaillés.")
    task_ids=[t["id"] for t in p["tasks"]]
    labels={t["id"]:t["id"]+" · "+t["name"] for t in p["tasks"]}
    chosen=st.selectbox("Tâche à modifier",task_ids,format_func=lambda identity:labels[identity],key="quick_task_"+p["id"])
    t=next(x for x in p["tasks"] if x["id"]==chosen)
    with st.form("quick_team_"+p["id"]+chosen):
        st.subheader("Modifier une équipe simplement")
        a,b=st.columns(2)
        workers=a.number_input("Nombre d’ouvriers",1,100,t["workers"])
        rate=b.number_input("Rendement par ouvrier et par jour",.01,10000.,t["rate"],step=.1,help="Unité du poste : "+ITEMS[t["item"]][1])
        wage=a.number_input("Salaire d’un ouvrier (FCFA/jour)",0.,1e7,t["wage"],step=500.)
        progress=b.number_input("Avancement déclaré (%)",0.,100.,t["progress"],step=5.)
        predecessors=st.multiselect("Tâches à terminer avant celle-ci",[x for x in task_ids if x!=chosen],default=list(filter(None,t["pred"].split(","))),format_func=lambda identity:labels[identity])
        if st.form_submit_button("Enregistrer cette tâche",type="primary",icon=":material/save:"):
            candidate=copy.deepcopy(p)
            for task in candidate["tasks"]:
                if task["id"]==chosen:task.update(workers=workers,rate=rate,wage=wage,progress=progress,pred=",".join(predecessors))
            try:update_project(candidate)
            except ValueError as exc:st.error(str(exc))
    st.caption("Le tableau ci-dessous permet aussi de modifier plusieurs tâches avant de les enregistrer ensemble.")
    editor_revision=hashlib.sha256(json.dumps(p["tasks"],sort_keys=True).encode()).hexdigest()[:12]
    with st.form("tasks_"+p["id"]):
        df=st.data_editor(task_frame(p),hide_index=True,num_rows="fixed",key="tasks_editor_"+p["id"]+editor_revision,
            disabled=["id","item"],alt="Tableau modifiable des équipes, rendements, salaires, dépendances et avancements",
            column_config={"id":st.column_config.TextColumn("ID"),"name":st.column_config.TextColumn("Tâche",required=True),
            "item":st.column_config.TextColumn("Poste"),"workers":st.column_config.NumberColumn("Effectif",min_value=1,max_value=100,step=1,required=True),
            "rate":st.column_config.NumberColumn("Rendement / ouvrier / jour",min_value=.01,max_value=10000.,required=True),
            "wage":st.column_config.NumberColumn("Salaire FCFA / jour",min_value=0.,max_value=1e7,required=True),
            "pred":st.column_config.TextColumn("Prédécesseurs"),"progress":st.column_config.NumberColumn("Avancement (%)",min_value=0.,max_value=100.,required=True)})
        if st.form_submit_button("Enregistrer les équipes et le planning",type="primary",icon=":material/save:"):
            try:
                candidate=copy.deepcopy(p);candidate["tasks"]=df.to_dict("records");update_project(candidate)
            except ValueError as exc:st.error(str(exc))
    with st.expander("Importer les tâches CSV / Excel"):
        st.download_button("Modèle CSV des tâches",csv_bytes(task_frame(p)),"Civil360_modele_taches.csv","text/csv")
        uploaded=st.file_uploader("Fichier de tâches",type=["csv","xlsx"])
        if st.button("Vérifier et remplacer les tâches",disabled=uploaded is None):
            try:
                candidate=copy.deepcopy(p);candidate["tasks"]=read_tasks(uploaded.getvalue(),uploaded.name.rsplit(".",1)[-1].lower());update_project(candidate)
            except ValueError as exc:st.error(str(exc))
    e=estimate(p);daily=daily_plan(p);start=date.fromisoformat(p["start"])
    metrics([("Durée",str(e["duration"])+" j","Dépendances prises en compte"),("Fin planifiée",(start+timedelta(days=e["duration"]-1)).strftime("%d/%m/%y"),"Date · JJ/MM/AA"),("Effectif maximal",str(int(daily.effectif.max())),"Ouvriers présents simultanément"),("Travail total",str(int(daily.effectif.sum())),"Journées-ouvriers")])
    st.caption("Calendrier : du "+start.strftime("%d/%m/%Y")+" au "+(start+timedelta(days=e["duration"]-1)).strftime("%d/%m/%Y")+", tous les jours travaillés.")
    fig=go.Figure()
    for t in e["tasks"]:
        fig.add_trace(go.Bar(x=[t["duration"]],base=[t["start"]],y=[t["id"]+" · "+t["name"]],orientation="h",name=t["id"],marker_color=PURPLE if t["slack"]==0 else BLUE,hovertemplate=f"Début J{t['start']+1} · fin J{t['end']}<br>{t['workers']} ouvriers · marge {t['slack']} j<extra></extra>"))
    fig.update_yaxes(autorange="reversed");fig.update_xaxes(title="Jour du projet");fig.update_layout(showlegend=False)
    st.subheader("Enchaînement des tâches")
    show_chart(fig,"Gantt : tâches critiques en violet et tâches avec marge en bleu",350)
    st.caption("Marge nulle : une prolongation de cette tâche peut repousser la fin du projet. Les durées nulles sont possibles si une quantité est nulle.")
    fig=go.Figure(go.Scatter(x=daily.jour,y=daily.effectif,mode="lines",line=dict(color=PURPLE,width=3,shape="hv"),fill="tozeroy",fillcolor="rgba(104,69,232,.12)",name="Effectif total"))
    fig.update_xaxes(title="Jour du projet");fig.update_yaxes(title="Nombre d’ouvriers",dtick=1)
    st.subheader("Les équipes au fil du temps")
    show_chart(fig,"Courbe de l’effectif total par jour, somme des équipes parallèles")
    st.dataframe(task_frame(p,True),hide_index=True,alt="Durées et effectifs calculés pour chaque tâche")
    st.download_button("Effectifs et finances CSV",csv_bytes(daily),"Civil360_effectifs_finances.csv","text/csv")


@st.cache_data(max_entries=8,ttl=3600,show_spinner=False)
def cached_simulation(payload,count,seed,prices,yields):
    return simulate(json.loads(payload),count,seed,prices,yields)


def render_risks():
    header("Risques et probabilités", "Et si les prix ou les rendements changeaient ? Explorez les conséquences.")
    p=require_project()
    if not p:return
    hint("Deux facteurs hypothétiques : un facteur de prix commun à tous les postes et un facteur de rendement commun à toutes les équipes. Ils sont indépendants l’un de l’autre. Les bornes sont des hypothèses, pas des observations ni des prévisions locales.")
    with st.form("risk_form_"+p["id"]):
        cols=st.columns(2)
        with cols[0]:
            st.subheader("Facteur de prix")
            price=tuple(st.number_input(label,.1,3.,v,step=.05,key="price_"+key+"_"+p["id"]) for label,v,key in [("Prix minimum",.9,"min"),("Prix le plus plausible",1.,"mode"),("Prix maximum",1.2,"max")])
        with cols[1]:
            st.subheader("Facteur de rendement")
            yields=tuple(st.number_input(label,.1,3.,v,step=.05,key="yield_"+key+"_"+p["id"]) for label,v,key in [("Rendement minimum",.75,"min"),("Rendement le plus plausible",1.,"mode"),("Rendement maximum",1.1,"max")])
        count=st.select_slider("Nombre de simulations",[1000,5000,10000,20000,50000],value=10000)
        seed=st.number_input("Graine aléatoire (reproductibilité)",0,2**32-1,42)
        submitted=st.form_submit_button("Lancer les simulations",type="primary",icon=":material/play_arrow:")
    payload=json.dumps(p,sort_keys=True)
    if submitted:
        try:
            with st.spinner("Calcul des scénarios…"):
                frame=cached_simulation(payload,count,seed,price,yields)
            st.session_state.risk_result=dict(payload=payload,frame=frame,prices=price,yields=yields,seed=seed,count=count)
        except ValueError as exc:st.error(str(exc))
    result=st.session_state.risk_result
    if not result or result["payload"]!=payload:
        st.caption("Lancez une simulation pour ce scénario enregistré. Après une modification du projet, les anciens résultats ne sont plus affichés.")
        return
    frame=result["frame"];n=len(frame)
    cost_hits=int((frame.cout_simule>p["budget_limit"]).sum());time_hits=int((frame.duree_simulee>p["deadline"]).sum())
    prob=cost_hits/n
    metrics([("Dépassement du budget",fmt(100*prob,1)+" %",f"{cost_hits} / {n} tirages"),
             ("Dépassement du délai",fmt(100*time_hits/n,1)+" %",f"{time_hits} / {n} tirages"),
             ("Coût médian",money(frame.cout_simule.median()),"FCFA · coût simulé sans provision"),
             ("Coût au quantile 95 %",money(frame.cout_simule.quantile(.95)),"FCFA · 95 % des tirages en dessous")])
    st.caption(f"Simulation affichée : {result['count']} tirages, graine {result['seed']}. Prix min/mode/max : {result['prices']}. Rendement : {result['yields']}. Ces bornes sont celles du dernier calcul, même si les champs ci-dessus ont ensuite été modifiés.")
    # Wilson reste informatif même si aucun dépassement n'a été observé.
    z=1.96;center=(prob+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(prob*(1-prob)/n+z*z/(4*n*n))/(1+z*z/n)
    st.caption(f"Intervalle de Wilson à 95 % pour la fréquence de dépassement du budget : [{fmt(100*(center-half),2)} % ; {fmt(100*(center+half),2)} %]. Cela mesure l’erreur d’échantillonnage des tirages, pas l’incertitude du modèle.")
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Histogram(x=frame.cout_simule,nbinsx=35,marker_color=PURPLE,name="Coût"));fig.add_vline(x=p["budget_limit"],line_dash="dash",line_color="#965D22");fig.update_xaxes(title="Coût simulé (FCFA)");fig.update_yaxes(title="Nombre de tirages")
        show_chart(fig,"Histogramme des coûts simulés et seuil de budget")
    with b:
        counts=frame.duree_simulee.value_counts().sort_index()
        fig=go.Figure(go.Bar(x=counts.index,y=counts.values,marker_color=BLUE));fig.add_vline(x=p["deadline"],line_dash="dash",line_color="#965D22");fig.update_xaxes(title="Durée simulée (jours)");fig.update_yaxes(title="Nombre de tirages")
        show_chart(fig,"Distribution discrète des durées et délai objectif")
    st.subheader("Sensibilité : modifier une hypothèse à la fois")
    base=estimate(p);labels=[];deltas=[]
    for label,pf,yf in [("Prix +10 %",1.1,1.),("Prix −10 %",.9,1.),("Rendement +10 %",1.,1.1),("Rendement −10 %",1.,.9)]:
        labels.append(label);deltas.append(estimate(p,pf,yf)["direct"]-base["direct"])
    fig=go.Figure(go.Bar(y=labels,x=deltas,orientation="h",marker_color=[PURPLE if v>0 else "#6A922F" for v in deltas]));fig.update_xaxes(title="Écart de coût (FCFA)")
    show_chart(fig,"Sensibilité déterministe du coût direct aux prix et aux rendements",280)
    st.dataframe(pd.DataFrame({"Statistique":["Moyenne","Écart-type (échantillon)","Quantile 5 %","Médiane","Quantile 95 %"],"Coût_FCFA":[frame.cout_simule.mean(),frame.cout_simule.std(ddof=1),frame.cout_simule.quantile(.05),frame.cout_simule.median(),frame.cout_simule.quantile(.95)],"Durée_jours":[frame.duree_simulee.mean(),frame.duree_simulee.std(ddof=1),frame.duree_simulee.quantile(.05),frame.duree_simulee.median(),frame.duree_simulee.quantile(.95)]}),hide_index=True,alt="Statistiques descriptives des résultats simulés")
    st.download_button("Tirages simulés CSV",csv_bytes(frame),"Civil360_simulations.csv","text/csv")
    meta=dict(app="Civil360",type="simulation",project=p,price_bounds=result["prices"],yield_bounds=result["yields"],seed=result["seed"],count=result["count"],budget_exceedances=cost_hits,deadline_exceedances=time_hits)
    st.download_button("Hypothèses du calcul JSON",json.dumps(meta,ensure_ascii=False,indent=2).encode(),"Civil360_hypotheses_simulation.json","application/json")
    with st.expander("Comprendre le modèle"):
        st.markdown("**Loi triangulaire** : définie par un minimum, une valeur la plus plausible et un maximum. **Monte-Carlo** : répéter le calcul avec des facteurs tirés au hasard. **Quantile 95 %** : valeur sous laquelle se trouvent 95 % des tirages. Un quantile de durée interpolé peut être décimal ; pour une échéance entière, arrondissez au supérieur.")
        st.markdown("Le coût simulé comprend fournitures, main-d’œuvre et frais fixes, **sans ajouter la provision comme dépense certaine**. Il est comparé au budget disponible saisi dans Projets. La durée tient compte des dépendances et des arrondis journaliers. Aucun taux obtenu ne représente une conformité ou la sécurité d’un ouvrage.")


def render_reports():
    header("Rapports et sauvegardes", "Emportez vos calculs, conservez vos hypothèses et reprenez votre travail.")
    p=require_project()
    if p:
        st.subheader("Exporter le projet actif")
        with st.container(horizontal=True):
            st.download_button("Note de projet PDF",pdf_bytes(p,st.session_state.demo),"Civil360_note.pdf","application/pdf",icon=":material/picture_as_pdf:")
            st.download_button("Classeur Excel",excel_bytes(p),"Civil360_projet.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",icon=":material/table_view:")
            st.download_button("Budget CSV",csv_bytes(budget_frame(p)),"Civil360_budget.csv","text/csv",icon=":material/download:")
        st.caption("PDF et Excel : scénario de base enregistré. Les simulations et variantes temporaires s’exportent dans leurs rubriques.")
    st.subheader("Sauvegarder tous les projets de cet espace")
    st.download_button("Sauvegarde complète JSON",backup_bytes(projects()),"Civil360_sauvegarde.json","application/json",icon=":material/save:")
    with st.form("restore"):
        upload=st.file_uploader("Sauvegarde Civil360 à restaurer",type="json")
        confirmed=st.checkbox("Ajouter les projets comme copies sans supprimer les projets existants")
        if st.form_submit_button("Importer les copies"):
            try:
                if upload is None:raise ValueError("Choisissez d’abord un fichier de sauvegarde JSON.")
                if not confirmed:raise ValueError("Cochez la confirmation d’ajout des copies.")
                imported=read_backup(upload.getvalue());commit(projects()+imported,imported[0]["id"]);st.rerun()
            except ValueError as exc:st.error(str(exc))
    hint("En local, vos projets sont enregistrés dans SQLite hors du dossier livré. Sur Streamlit Cloud, le disque peut être temporaire et le registre est partagé : utilisez des données pédagogiques et téléchargez vos sauvegardes JSON.")
    st.subheader("Parcours de test guidé")
    st.markdown("1. Chargez **l’exemple fictif**.\n2. Dans **Quantités**, vérifiez **14,72 m³ de béton** et **96 m² de murs**.\n3. Dans **Planning**, vérifiez **26 jours**, un maximum de **6 ouvriers** et **112 journées-ouvriers**.\n4. Dans **Budget**, vérifiez **3 951 360 FCFA** de fournitures, **672 000 FCFA** de main-d’œuvre et **5 360 696 FCFA** provision comprise.\n5. Dans **Risques**, mettez les six facteurs à **1** : coût simulé **4 873 360 FCFA**, durée **26 jours**, aucun dépassement du budget de **9 000 000 FCFA** ni du délai de **45 jours**.\n6. Téléchargez les exports, puis restaurez le JSON comme copie dans l’exemple fictif.")


def main():
    st.set_page_config(page_title="Civil360 · bureau de projet",page_icon=":material/domain:",layout="wide")
    state=st.session_state
    if "projects" not in state:
        try:state.projects,state.revision=load_store()
        except (OSError,sqlite3.Error,ValueError) as exc:
            st.error("Impossible d’ouvrir le registre local. Vérifiez le dossier CIVIL360_DATA_DIR et les droits d’écriture.");st.caption(str(exc));st.stop()
        state.demo=False;state.demo_projects=[];state.selected=None;state.risk_result=None
    state.setdefault("risk_result",None)
    backgrounds={"Brume lilas":"#F4F1FB","Sable clair":"#F7F2E9","Glace bleue":"#EFF4FA"}
    bg=backgrounds.get(state.get("appearance","Brume lilas"),"#F4F1FB")
    st.html(CSS+f'<style>:root{{--c-bg:{bg};}}</style>')
    global PAGES
    PAGES=[st.Page(render_dashboard,title="Vue d’ensemble",icon=":material/dashboard:",default=True,url_path="accueil"),
           st.Page(render_projects,title="Projets",icon=":material/folder_open:",url_path="projets"),
           st.Page(render_quantities,title="Quantités",icon=":material/straighten:",url_path="quantites"),
           st.Page(render_budget,title="Budget et variantes",icon=":material/account_balance_wallet:",url_path="budget"),
           st.Page(render_planning,title="Planning et effectifs",icon=":material/calendar_month:",url_path="planning"),
           st.Page(render_risks,title="Risques et probabilités",icon=":material/query_stats:",url_path="risques"),
           st.Page(render_reports,title="Rapports",icon=":material/description:",url_path="rapports")]
    with st.sidebar:
        st.html('<div class="sidebrand">Civil<b>360</b></div><div class="sidetag">Les décisions se construisent ensemble.</div>')
        st.button("Charger l’exemple fictif",on_click=enter_demo,icon=":material/science:",key="load_demo",width="stretch")
        if state.demo:st.button("Revenir à mes projets",on_click=leave_demo,icon=":material/folder:",width="stretch")
        if current():st.caption("Projet actif : "+current()["name"])
    page=st.navigation(PAGES,position="sidebar")
    with st.sidebar:
        with st.expander("Apparence",icon=":material/palette:"):
            st.radio("Arrière-plan",list(backgrounds),key="appearance")
            st.caption("Palette violette, touches citron et fonds doux. Présentation optimisée en mode clair.")
        if not state.demo and st.button("Recharger le registre",icon=":material/refresh:"):
            try:state.projects,state.revision=load_store();state.risk_result=None;st.rerun()
            except (OSError,sqlite3.Error,ValueError) as exc:st.error(str(exc))
        st.caption("Version "+VERSION+" · Génie civil / pédagogie")
    if state.get("flash"):st.success(state.pop("flash"))
    try:page.run()
    except (ValueError,sqlite3.Error,OSError) as exc:
        st.error("Le calcul ou l’enregistrement n’a pas abouti : "+str(exc))
    st.html('<div class="foot">Civil360 · Étude pédagogique d’un bâtiment de plain-pied. Dimensions, prix et rendements à justifier. Aucune validation d’ouvrage.</div>')


if __name__ == "__main__":
    main()
