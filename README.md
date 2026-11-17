# Rozpoznávanie tváre pri použití rúška alebo okuliarov

Zdrojový kód k praktickej časti semestrálnej práce z predmetu Biometrické
systémy bezpečnosti (TUKE FEI, 2026). Autor: Bc. Branislav Kačmarik.

Systém meria, o koľko rúško a okuliare znižujú presnosť rozpoznávania tváre
založeného na predtrénovanom modeli ArcFace (knižnica InsightFace), a
overuje jednoduché vylepšenie — augmentáciu galériových šablón o syntetické
varianty s rúškom/okuliarmi.

Kompletné zdôvodnenie, metodika a výsledky sú opísané v priloženej
semestrálnej práci (PDF).

## Obsah

- `scripts/` — všetky použité Python skripty
- `data/own/` — vlastné testovacie fotografie (originály + reálne
  Photoshop prekrytie rúškom/okuliarmi), použité na kalibráciu veľkosti
  a tvaru syntetického prekrytia (sekcia 2.3.3 práce)
- `results/` — uložené výsledky experimentov (JSON s nameranými
  metrikami, logy behov, náhľady vygenerovaného prekrytia) — presne
  tie, z ktorých vychádzajú čísla a tabuľky v práci

Dataset LFW (62 identít) **nie je súčasťou tohto priečinka** — sťahuje sa
automaticky pri prvom spustení `prepare_dataset.py` (krok 2 nižšie).

## 1. Požiadavky

- **Python 3.11** (odporúčané; 3.10+ by malo tiež fungovať)
- **internetové pripojenie** pri prvom spustení — sťahuje sa dataset LFW
  (cca 60 MB) aj rozpoznávací model `buffalo_l` (cca 280 MB)
- bežný CPU stačí, GPU sa nevyžaduje (celá práca bola robená na CPU)

## 2. Inštalácia

Otvorte príkazový riadok (PowerShell / cmd) v tomto priečinku a spustite:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r scripts/requirements.txt
```

(na Linuxe/macOS namiesto `venv\Scripts\activate` použite `source venv/bin/activate`)

Inštalácia knižníc trvá spravidla 2–5 minút.

## 3. Spustenie — odporúčané poradie

Všetky príkazy sa spúšťajú z priečinka `scripts/`:

```bash
cd scripts
```

### Krok 1 — stiahnutie datasetu (nutné spustiť ako prvé)

```bash
python prepare_dataset.py
```

Stiahne dataset LFW cez `scikit-learn` a uloží fotografie do
`../data/lfw/<meno_osoby>/`. Trvá cca 1–3 minúty podľa rýchlosti
pripojenia. Na konci vypíše počet stiahnutých identít (malo by byť 62).

### Krok 2 — hlavný experiment (baseline aj s vylepšením)

```bash
python run_experiment.py
```

Pri prvom spustení sa automaticky stiahne aj model `buffalo_l`
(InsightFace) — môže to trvať pár minút podľa pripojenia, nabudúce sa už
len načíta z disku. Skript prejde všetky identity, vyhodnotí presnosť
bez prekrytia, s rúškom aj s okuliarmi a vypíše do konzoly Accuracy,
FAR, FRR, EER a priemerné podobnosti pre každú podmienku — presne tie
čísla, ktoré sú v práci v tabuľke 2.3 (sekcia 2.5–2.6). Výsledok sa
zároveň uloží do `../results/baseline.json`.

Celý beh trvá na bežnom notebooku (CPU) rádovo **5–15 minút**.

Následne spustite aj navrhnuté vylepšenie (augmentovaná galéria):

```bash
python run_experiment.py --augment
```

Výsledok sa uloží do `../results/augmented.json` a mal by zodpovedať
tabuľke 2.4 (sekcia 2.8) v práci.

Beh používa pevný `random.seed(42)`, takže namerané čísla by mali byť pri
opakovanom spustení na ľubovoľnom počítači takmer identické s tými v práci
(priemerné podobnosti na stotinu presne) — drobné odchýlky v accuracy/EER
rádovo v treťom desatinnom mieste sú normálne, spôsobené nedeterminizmom
CPU inferencie cez ONNX runtime, nie chybou.

### Krok 3 — rýchle demo na jednej fotografii (najrýchlejší spôsob overenia)

Ak nechcete čakať na celý experiment, toto overí systém na jednej
konkrétnej osobe za pár sekúnd (po stiahnutí datasetu a modelu z
krokov 1–2, resp. aj bez spusteného `run_experiment.py`):

```bash
python identify.py --image "../data/lfw/Angelina_Jolie/005.jpg" --occlude mask
```

Skript vypíše TOP 5 najbližších zhôd a finálnu identifikovanú osobu
(mala by vyjsť `Angelina_Jolie`). Parameter `--occlude` môže byť `mask`
alebo `glasses` (dá sa aj vynechať — vtedy sa použije nezakrytá
fotografia), `--augment` prepne na augmentovanú (vylepšenú) galériu:

```bash
python identify.py --image "../data/lfw/Angelina_Jolie/005.jpg" --occlude glasses --augment
```

Presne tieto príkazy a ich výstup sú opísané aj v sekcii 2.8.1 práce
("Praktické overenie v termináli").

### Krok 4 (voliteľné) — porovnanie reálneho a syntetického prekrytia

Overenie, nakoľko verne vlastný skript na generovanie rúška/okuliarov
(`overlay.py`) simuluje skutočné prekrytie, na vlastných fotografiách
z `../data/own/` (sekcia 2.3.3 práce):

```bash
python real_vs_synthetic_test.py
```

## Riešenie problémov

- **"ModuleNotFoundError"** — nebolo aktivované virtuálne prostredie
  (`venv\Scripts\activate`) alebo preskočený krok `pip install -r
  scripts/requirements.txt`.
- **Prvé spustenie trvá dlho / vyzerá "zaseknuté"** — v pozadí sa sťahuje
  model `buffalo_l` (cca 280 MB), je potrebné internetové pripojenie;
  pri pomalšom pripojení to môže trvať aj niekoľko minút.
- **`FileNotFoundError` pre `data/lfw`** — treba najprv spustiť krok 1
  (`python prepare_dataset.py`), skripty v kroku 2 a 3 dataset sami
  nesťahujú.
- Všetky skripty sa spúšťajú **z priečinka `scripts/`** (používajú
  relatívne cesty `../data`, `../results`).

## Poznámka k reprodukovateľnosti

Experimenty bežali na CPU (bez GPU) s pevným `seed=42`. Reálne otestované
(na čistom prostredí, znovu od nuly) — opakovaný beh dáva priemerné
genuine podobnosti identické s tými v práci, accuracy/EER sa môžu líšiť
v treťom desatinnom mieste kvôli nedeterminizmu CPU inferencie cez ONNX
runtime (nie chyba, bežný jav pri opakovanom behu neurónových sietí na
CPU).
