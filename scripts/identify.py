"""
Jednoduchy demo skript: zoberie jednu fotku (probe) a identifikuje osobu
z galerie (tych istych LFW identit, ktore pouziva run_experiment.py).
Volitelne na probe fotku pred rozpoznanim synteticky dogeneruje rusko
alebo okuliare, aby bolo vidiet vplyv prekrytia na % zhody.

Pouzitie:
    python scripts/identify.py --image cesta/k/fotke.jpg
    python scripts/identify.py --image cesta/k/fotke.jpg --occlude mask
    python scripts/identify.py --image cesta/k/fotke.jpg --occlude glasses --augment
"""
import argparse
import sys
import cv2
import numpy as np

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from common import get_app, embed, cosine, list_identities
from overlay import add_mask, add_glasses

N_GALLERY = 3


def _mean_embedding(embeddings):
    e = np.mean(np.stack(embeddings, axis=0), axis=0)
    return e / np.linalg.norm(e)


def build_templates(identities, app, augment):
    """Rovnaka logika ako v run_experiment.py: pre kazdu identitu vytvori
    priemerny (mean-pooled) template z N_GALLERY fotiek, pripadne aj
    augmentovane mask/glasses templaty."""
    templates = {}
    for name, files in identities.items():
        imgs_kps, orig_embeds = [], []
        for f in files[:N_GALLERY]:
            img = cv2.imread(f)
            e, kps = embed(img, app)
            if e is None:
                continue
            orig_embeds.append(e)
            imgs_kps.append((img, kps))
        if not orig_embeds:
            continue
        templ_list = [("orig", _mean_embedding(orig_embeds))]
        if augment:
            for cond, fn in (("mask", add_mask), ("glasses", add_glasses)):
                cond_embeds = []
                for img, kps in imgs_kps:
                    if kps is None:
                        continue
                    e_occ, _ = embed(fn(img, kps), app)
                    if e_occ is not None:
                        cond_embeds.append(e_occ)
                if cond_embeds:
                    templ_list.append((cond, _mean_embedding(cond_embeds)))
        templates[name] = templ_list
    return templates


def best_score(templ_list, probe_e):
    return max(cosine(probe_e, e) for _, e in templ_list)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        # cv2.imread zlyhava na Windows pri diakritike v ceste
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def main():
    parser = argparse.ArgumentParser(description="Identifikacia osoby z galerie na zaklade jednej fotky.")
    parser.add_argument("--image", required=True, help="cesta k probe fotke")
    parser.add_argument("--occlude", choices=["none", "mask", "glasses"], default="none",
                         help="synteticky dogenerovat rusko/okuliare na probe fotku pred rozpoznanim")
    parser.add_argument("--augment", action="store_true",
                         help="pouzit augmentovanu galeriu (vylepsenie navrhnute v praci)")
    parser.add_argument("--min-images", type=int, default=12)
    parser.add_argument("--top", type=int, default=5, help="kolko najlepsich zhod vypisat")
    args = parser.parse_args()

    app = get_app()
    identities = list_identities(min_images=args.min_images)
    print(f"Galeria: {len(identities)} identit (po {N_GALLERY} fotkach kazda)"
          + (", augmentovana (orig+mask+glasses)" if args.augment else ""))

    templates = build_templates(identities, app, args.augment)

    img = load_image(args.image)
    if img is None:
        raise SystemExit(f"Nepodarilo sa nacitat obrazok: {args.image}")

    e, kps = embed(img, app)
    if e is None:
        raise SystemExit("Na fotke sa nepodarila detekcia tvare.")

    if args.occlude != "none":
        if kps is None:
            raise SystemExit("Landmarky na vygenerovanie prekrytia sa nenasli.")
        overlay_fn = {"mask": add_mask, "glasses": add_glasses}[args.occlude]
        occ_img = overlay_fn(img, kps)
        e_occ, _ = embed(occ_img, app)
        if e_occ is None:
            raise SystemExit("Po pridani prekrytia sa tvar uz nepodarilo detegovat.")
        e = e_occ
        out_path = args.image.rsplit(".", 1)[0] + f"_{args.occlude}_preview.jpg"
        cv2.imencode(".jpg", occ_img)[1].tofile(out_path)
        print(f"Syntetické {args.occlude} pridané, náhľad uložený do: {out_path}")

    scores = [(name, best_score(templ, e)) for name, templ in templates.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTOP {args.top} zhoda:")
    for name, score in scores[:args.top]:
        print(f"  {name:30s}  {score * 100:6.2f} % zhoda  (cosine {score:.4f})")

    best_name, best_val = scores[0]
    print(f"\n=> Identifikovaná osoba: {best_name}  ({best_val * 100:.2f} % zhoda)")


if __name__ == "__main__":
    main()
