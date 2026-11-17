"""
Over, ci program rozpozna rovnaku osobu na vlastnej fotke po tom, ako sa jej
na tvar dogeneruju okuliare (resp. ruska). Pouziva ten isty detektor/model
a tie iste funkcie (embed, cosine, add_glasses/add_mask) ako hlavny
experiment na LFW datasete.

Spustenie:
    python scripts/own_photo_test.py
"""
import os
import sys
import glob
import itertools
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from common import get_app, embed, cosine
from overlay import add_glasses, add_mask

OWN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "own")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "own")

# EER threshold z hlavneho LFW experimentu (baseline, bez augmentacie) -
# pouzijeme ho ako referencny prah pre rozhodnutie "rovnaka osoba / cudzia osoba"
EER_THRESHOLD = 0.246  # z results/baseline.json ("orig" -> "threshold")


def load_photos():
    candidates = set(
        glob.glob(os.path.join(OWN_DIR, "*.jpg"))
        + glob.glob(os.path.join(OWN_DIR, "*.JPG"))
        + glob.glob(os.path.join(OWN_DIR, "*.jpeg"))
        + glob.glob(os.path.join(OWN_DIR, "*.png"))
    )
    if not candidates:
        raise SystemExit(f"Ziadna fotka v {OWN_DIR}")
    return sorted(candidates)


def decide(sim):
    return "ZHODA (rovnaka osoba)" if sim >= EER_THRESHOLD else "NEZHODA (cudzia osoba)"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app = get_app()

    entries = []  # (name, img, emb, kps)
    for path in load_photos():
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"{name}: fotku sa nepodarilo nacitat, preskakujem")
            continue
        emb, kps = embed(img, app)
        if emb is None:
            print(f"{name}: na fotke sa nenasla ziadna tvar, preskakujem")
            continue
        entries.append((name, img, emb, kps))
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_original.jpg"), img)
        print(f"Nacitana fotka: {name}")

    if not entries:
        raise SystemExit("Ziadna fotka sa nedala spracovat.")

    # 1) realne genuine porovnanie medzi roznymi fotkami (bez prekrytia)
    if len(entries) >= 2:
        print("\n--- Priame porovnanie roznych fotiek (bez prekrytia) ---")
        for (n1, _, e1, _), (n2, _, e2, _) in itertools.combinations(entries, 2):
            sim = cosine(e1, e2)
            print(f"{n1} vs {n2}: cosine similarity = {sim:.4f}  ->  {decide(sim)} (prah {EER_THRESHOLD:.4f})")

    # 2) efekt syntetickeho prekrytia na kazdej fotke samostatne
    print("\n--- Efekt syntetickeho prekrytia (fotka vs ta ista fotka s okuliarmi/ruskom) ---")
    for name, img, emb_orig, kps in entries:
        img_glasses = add_glasses(img, kps)
        img_mask = add_mask(img, kps)
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_glasses.jpg"), img_glasses)
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_mask.jpg"), img_mask)

        emb_glasses, _ = embed(img_glasses, app)
        emb_mask, _ = embed(img_mask, app)

        for label, emb in [("okuliare", emb_glasses), ("ruska", emb_mask)]:
            if emb is None:
                print(f"{name} + {label}: tvar sa po prekryti nenasla (detektor zlyhal)")
                continue
            sim = cosine(emb_orig, emb)
            print(f"{name} + {label}: cosine similarity = {sim:.4f}  ->  {decide(sim)} (prah {EER_THRESHOLD:.4f})")

    print(f"\nUlozene obrazky v: {OUT_DIR}")


if __name__ == "__main__":
    main()
