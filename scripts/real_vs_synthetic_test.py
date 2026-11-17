"""
Porovna REALNE (Photoshop) prekrytie ruskom/okuliarmi na vlastnych fotkach
so SYNTETICKYM prekrytim generovanym skriptom overlay.py (add_mask/add_glasses)
na tych istych fotkach. Cielom je zistit, o kolko % sa lisi pokles cosine
similarity medzi realnym a syntetickym prekrytim.

Spustenie:
    python scripts/real_vs_synthetic_test.py
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import get_app, embed, cosine
from overlay import add_glasses, add_mask

OWN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "own")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "own")

# (nazov_zakladnej_fotky, original, realna_ruska, realne_okuliare, realne_oboje)
PAIRS = [
    ("osoba_A (IMG_8611)", "IMG_8611.JPG", "Fotka s Rúškom 2.jpg",
     "Fotka so slnečnými okuliarmi 2.jpg",
     "Fotka so slnečnými okuliarmi a rúškom 2.jpg"),
    ("osoba_B (IMG_8612)", "IMG_8612.JPG", "Fotka s Rúškom.jpg",
     "Fotka so slnečnými okuliarmi.jpg",
     "Fotka so slnečnými okuliarmi a rúškom.jpg"),
]


def load(name):
    """cv2.imread zlyhava na Windows pri ceste s diakritikou, preto citame
    bajty cez numpy a dekodujeme rucne."""
    path = os.path.join(OWN_DIR, name)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Nepodarilo sa nacitat: {path}")
    return img


def pct_drop(base_sim, occ_sim):
    """O kolko % poklesla similarity oproti idealu (1.0 = sebou samym)."""
    return (base_sim - occ_sim) / base_sim * 100.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app = get_app()

    print(f"{'podmienka':35s} {'realne (Photoshop)':>20s} {'synteticke (skript)':>20s} {'rozdiel (p.b.)':>15s}")
    print("-" * 95)

    summary = {"mask": {"real": [], "synth": []}, "glasses": {"real": [], "synth": []},
               "both": {"real": []}}

    for label, orig_name, real_mask_name, real_glasses_name, real_both_name in PAIRS:
        img_orig = load(orig_name)
        emb_orig, kps_orig = embed(img_orig, app)
        if emb_orig is None:
            print(f"{label}: tvar na originali sa nenasla, preskakujem")
            continue

        # -- realne (Photoshop) varianty --
        real_results = {}
        for cond, fname in (("mask", real_mask_name), ("glasses", real_glasses_name),
                             ("both", real_both_name)):
            img = load(fname)
            emb, _ = embed(img, app)
            if emb is None:
                print(f"{label} + realne {cond}: tvar sa nenasla (detektor zlyhal na {fname})")
                continue
            sim = cosine(emb_orig, emb)
            real_results[cond] = sim

        # -- synteticke varianty (ten isty skript ako na LFW) --
        synth_results = {}
        img_glasses = add_glasses(img_orig, kps_orig)
        img_mask = add_mask(img_orig, kps_orig)
        cv2.imencode(".jpg", img_mask)[1].tofile(
            os.path.join(OUT_DIR, f"{orig_name}_synth_mask.jpg"))
        cv2.imencode(".jpg", img_glasses)[1].tofile(
            os.path.join(OUT_DIR, f"{orig_name}_synth_glasses.jpg"))
        for cond, img_occ in (("mask", img_mask), ("glasses", img_glasses)):
            emb, _ = embed(img_occ, app)
            if emb is None:
                print(f"{label} + synteticke {cond}: tvar sa nenasla po prekryti")
                continue
            synth_results[cond] = cosine(emb_orig, emb)

        print(f"\n=== {label} ===")
        for cond in ("mask", "glasses"):
            r = real_results.get(cond)
            s = synth_results.get(cond)
            r_s = f"{r:.4f}" if r is not None else "n/a"
            s_s = f"{s:.4f}" if s is not None else "n/a"
            diff = f"{(r - s):+.4f}" if (r is not None and s is not None) else "n/a"
            print(f"{cond:35s} {r_s:>20s} {s_s:>20s} {diff:>15s}")
            if r is not None:
                summary[cond]["real"].append(r)
            if s is not None:
                summary[cond]["synth"].append(s)
        if "both" in real_results:
            print(f"{'both (realne, bez synt. ekvivalentu)':35s} {real_results['both']:.4f}")
            summary["both"]["real"].append(real_results["both"])

    print("\n" + "=" * 95)
    print("SUHRN (priemer cez obe osoby, pokles v percentach oproti idealnemu sebaporovnaniu 1.0)")
    print("=" * 95)
    for cond in ("mask", "glasses"):
        reals = summary[cond]["real"]
        synths = summary[cond]["synth"]
        if reals:
            avg_r = sum(reals) / len(reals)
            drop_r = pct_drop(1.0, avg_r)
            print(f"{cond}: realne (Photoshop)   priemer sim={avg_r:.4f}  pokles={drop_r:.1f}%")
        if synths:
            avg_s = sum(synths) / len(synths)
            drop_s = pct_drop(1.0, avg_s)
            print(f"{cond}: synteticke (skript)   priemer sim={avg_s:.4f}  pokles={drop_s:.1f}%")
        if reals and synths:
            print(f"{cond}: synteticke prekrytie je "
                  f"{'MIERNEJSIE' if drop_s < drop_r else 'PRISNEJSIE'} nez realne "
                  f"o {abs(drop_s - drop_r):.1f} percentualneho bodu "
                  f"({'podhodnocuje' if drop_s < drop_r else 'nadhodnocuje'} efekt prekrytia)")
        print()
    if summary["both"]["real"]:
        avg_b = sum(summary["both"]["real"]) / len(summary["both"]["real"])
        print(f"both (okuliare+ruska, len realne): priemer sim={avg_b:.4f}  "
              f"pokles={pct_drop(1.0, avg_b):.1f}%")


if __name__ == "__main__":
    main()
