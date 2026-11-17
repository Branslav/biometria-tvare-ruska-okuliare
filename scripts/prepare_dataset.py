"""
Priprava datasetu: stiahne LFW (cez sklearn), ulozi snimky na disk ako JPG
usporiadane do priecinkov podla identity: data/lfw/<meno_priezvisko>/0001.jpg ...
Pouzivame min_faces_per_person=20, aby kazda identita mala dost snimok na
galeriu aj probe mnozinu.
"""
import os
import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lfw")

def main():
    lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=2.0)
    print("Nacitane snimky:", lfw.images.shape, " / ", len(lfw.target_names), "identit")

    os.makedirs(OUT_DIR, exist_ok=True)
    counters = {}
    for img, label in zip(lfw.images, lfw.target):
        name = lfw.target_names[label].replace(" ", "_")
        person_dir = os.path.join(OUT_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        bgr = (img * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(person_dir, f"{idx:03d}.jpg"), bgr)

    print("Hotovo. Ulozene do:", os.path.abspath(OUT_DIR))
    print("Pocet identit:", len(counters))
    print("Priemer snimok na identitu:", np.mean(list(counters.values())))

if __name__ == "__main__":
    main()
