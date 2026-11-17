"""
Hlavny experiment: baseline vs. ruska vs. okuliare, s moznostou pouzit
"augmentovanu" galeriu (--augment), kde su pre kazdu identitu v galerii
okrem originalnych snimok aj synteticky prekryte varianty (vylepsenie).

Vystup: JSON s metrikami pre kazdu podmienku, ulozeny do results/.
"""
import os
import json
import random
import argparse
import numpy as np
import cv2

from common import get_app, embed, cosine, list_identities, compute_eer_and_accuracy
from overlay import add_mask, add_glasses

random.seed(42)
np.random.seed(42)

N_GALLERY = 3
N_PROBE_MAX = 8
N_IMPOSTORS_PER_PROBE = 3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _mean_embedding(embeddings):
    e = np.mean(np.stack(embeddings, axis=0), axis=0)
    return e / np.linalg.norm(e)


def build_templates(identities, app, augmented):
    """Vrati dict: identity -> zoznam (condition, embedding) templatov z galerie.
    Kazdy template je PRIEMERNY (mean-pooled) embedding z N_GALLERY snimok danej
    podmienky (standardny pristup pri viacnasobnom zapise do galerie)."""
    templates = {}
    for name, files in identities.items():
        gallery_files = files[:N_GALLERY]
        imgs_kps = []
        orig_embeds = []
        for f in gallery_files:
            img = cv2.imread(f)
            e, kps = embed(img, app)
            if e is None:
                continue
            orig_embeds.append(e)
            imgs_kps.append((img, kps))

        if not orig_embeds:
            continue
        templ_list = [("orig", _mean_embedding(orig_embeds))]

        if augmented:
            for cond, fn in (("mask", add_mask), ("glasses", add_glasses)):
                cond_embeds = []
                for img, kps in imgs_kps:
                    if kps is None:
                        continue
                    occ_img = fn(img, kps)
                    e_occ, _ = embed(occ_img, app)
                    if e_occ is not None:
                        cond_embeds.append(e_occ)
                if cond_embeds:
                    templ_list.append((cond, _mean_embedding(cond_embeds)))

        templates[name] = templ_list
    return templates


def best_match_score(templates_for_identity, probe_embedding):
    return max(cosine(probe_embedding, e) for _, e in templates_for_identity)


def run_condition(identities, templates, app, condition):
    """condition in {'orig','mask','glasses'} aplikovane na probe snimky."""
    overlay_fn = {"orig": None, "mask": add_mask, "glasses": add_glasses}[condition]
    names = list(identities.keys())

    genuine_scores = []
    impostor_scores = []

    for name, files in identities.items():
        if name not in templates:
            continue
        probe_files = files[N_GALLERY:N_GALLERY + N_PROBE_MAX]
        for f in probe_files:
            img = cv2.imread(f)
            e, kps = embed(img, app)
            if e is None:
                continue
            if overlay_fn is not None and kps is not None:
                occ_img = overlay_fn(img, kps)
                e_occ, _ = embed(occ_img, app)
                if e_occ is None:
                    continue
                probe_e = e_occ
            else:
                probe_e = e

            genuine_scores.append(best_match_score(templates[name], probe_e))

            others = [n for n in names if n != name and n in templates]
            sampled = random.sample(others, min(N_IMPOSTORS_PER_PROBE, len(others)))
            for other in sampled:
                impostor_scores.append(best_match_score(templates[other], probe_e))

    return compute_eer_and_accuracy(genuine_scores, impostor_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true",
                         help="pouzit augmentovanu galeriu (vylepsenie)")
    parser.add_argument("--min-images", type=int, default=12)
    args = parser.parse_args()

    app = get_app()
    identities = list_identities(min_images=args.min_images)
    print(f"Pocet identit pouzitych v experimente: {len(identities)}")

    templates = build_templates(identities, app, augmented=args.augment)
    print(f"Galeriove templaty vytvorene pre {len(templates)} identit "
          f"({'augmentovane' if args.augment else 'len original'})")

    results = {}
    for condition in ("orig", "mask", "glasses"):
        print(f"Vyhodnocujem podmienku: {condition} ...")
        res = run_condition(identities, templates, app, condition)
        results[condition] = res
        print(f"  -> Accuracy={res['accuracy']:.4f}  FAR={res['FAR']:.4f}  "
              f"FRR={res['FRR']:.4f}  EER={res['EER']:.4f}  "
              f"(genuine={res['n_genuine']}, impostor={res['n_impostor']})")
        print(f"     genuine_sim={res['genuine_mean']:.4f}+-{res['genuine_std']:.4f}  "
              f"impostor_sim={res['impostor_mean']:.4f}+-{res['impostor_std']:.4f}  "
              f"margin={res['margin']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_name = "augmented.json" if args.augment else "baseline.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("Ulozene do:", out_path)


if __name__ == "__main__":
    main()
