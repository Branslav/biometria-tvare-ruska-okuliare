# Rozpoznávanie tváre pri použití rúška alebo okuliarov

Zdrojový kód k praktickej časti semestrálnej práce z predmetu Biometrické
systémy bezpečnosti (TUKE FEI, 2026). Autor: Bc. Branislav Kačmarik.

Systém meria, o koľko rúško a okuliare znižujú presnosť rozpoznávania tváre
založeného na predtrénovanom modeli ArcFace (knižnica InsightFace), a
overuje jednoduché vylepšenie — augmentáciu galériových šablón o syntetické
varianty s rúškom/okuliarmi.

Kompletné zdôvodnenie, metodika a výsledky sú opísané v priloženej
semestrálnej práci (PDF). Tento repozitár slúži ako archivácia a
reprodukovateľný zdrojový kód k nej.

## Obsah repozitára

- `scripts/` — všetky použité Python skripty
- `data/own/` — vlastné testovacie fotografie (originály + reálne
  Photoshop prekrytie rúškom/okuliarmi), použité na kalibráciu veľkosti
  a tvaru syntetického prekrytia
- `results/` — uložené výsledky experimentov (JSON s nameranými
  metrikami, logy behov, náhľady vygenerovaného prekrytia)

Dataset LFW (62 identít) nie je súčasťou repozitára — sťahuje sa
automaticky pri prvom spustení `prepare_dataset.py`.

## Inštalácia

Vyžaduje sa Python 3.11+.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r scripts/requirements.txt
```

Model `buffalo_l` (SCRFD detektor + ArcFace) sa automaticky stiahne pri
prvom spustení ktoréhokoľvek skriptu, ktorý používa InsightFace.

## Spustenie

Všetky skripty sa spúšťajú z priečinka `scripts/`.

1. **Príprava datasetu** (stiahne LFW a uloží snímky do `data/lfw/`):
   ```bash
   cd scripts
   python prepare_dataset.py
   ```

2. **Hlavný experiment** — baseline aj s rúškom/okuliarmi, bez augmentácie:
   ```bash
   python run_experiment.py
   ```
   S navrhnutým vylepšením (augmentovaná galéria):
   ```bash
   python run_experiment.py --augment
   ```
   Výsledky (accuracy, FAR, FRR, EER, priemerné podobnosti) sa vypíšu do
   konzoly a uložia do `results/`.

3. **Demo identifikácia jednej osoby** (1:N, s výpisom TOP 5 zhôd):
   ```bash
   python identify.py --image "../data/lfw/Angelina_Jolie/005.jpg" --occlude mask
   python identify.py --image "../data/lfw/Angelina_Jolie/005.jpg" --occlude glasses --augment
   ```
   Parameter `--occlude` môže byť `mask` alebo `glasses` (voliteľný,
   vynechaním sa použije nezakrytá snímka), `--augment` prepne na
   augmentovanú galériu.

4. **Porovnanie reálneho (Photoshop) a syntetického prekrytia** na
   vlastných fotografiách z `data/own/` — overenie, nakoľko verne
   `overlay.py` simuluje skutočné rúško/okuliare:
   ```bash
   python real_vs_synthetic_test.py
   ```

## Poznámka k reprodukovateľnosti

Experimenty bežali na CPU (bez GPU), beh `run_experiment.py` na 62
identitách trvá rádovo niekoľko minút. Výsledky sú deterministické
(pevný `seed`), takže opakovaný beh by mal dať rovnaké čísla, aké sú
uvedené v práci.
