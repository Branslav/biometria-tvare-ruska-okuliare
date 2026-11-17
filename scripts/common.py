"""Spolocne funkcie pre vsetky experimenty: nacitanie modelu, embedding,
vykreslenie prekrytia a metriky FAR/FRR/EER/Accuracy."""
import os
import glob
import numpy as np
import cv2
import insightface

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lfw")

_APP = None


def get_app(det_size=(320, 320)):
    global _APP
    if _APP is None:
        _APP = insightface.app.FaceAnalysis(name="buffalo_l")
        _APP.prepare(ctx_id=-1, det_size=det_size)
    return _APP


def embed(img_bgr, app=None):
    """Vrati (embedding, kps) pre najvacsiu detegovanu tvar, alebo (None, None)."""
    app = app or get_app()
    faces = app.get(img_bgr)
    if not faces:
        return None, None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    f = faces[0]
    return f.embedding, f.kps


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def list_identities(min_images=10):
    people = sorted(os.listdir(DATA_DIR))
    result = {}
    for p in people:
        files = sorted(glob.glob(os.path.join(DATA_DIR, p, "*.jpg")))
        if len(files) >= min_images:
            result[p] = files
    return result


def compute_eer_and_accuracy(genuine, impostor):
    """genuine, impostor: zoznamy cosine podobnosti. Vrati dict s EER,
    prahom pri EER, accuracy pri tomto prahu, a FAR/FRR pri tomto prahu."""
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    thresholds = np.linspace(-1.0, 1.0, 2001)

    best = None
    for t in thresholds:
        far = float(np.mean(impostor >= t)) if len(impostor) else 0.0
        frr = float(np.mean(genuine < t)) if len(genuine) else 0.0
        diff = abs(far - frr)
        if best is None or diff < best[0]:
            best = (diff, t, far, frr)

    _, eer_threshold, far_at_eer, frr_at_eer = best
    eer = (far_at_eer + frr_at_eer) / 2

    n_g, n_i = len(genuine), len(impostor)
    correct = np.sum(genuine >= eer_threshold) + np.sum(impostor < eer_threshold)
    accuracy = correct / (n_g + n_i)

    return {
        "threshold": eer_threshold,
        "EER": eer,
        "FAR": far_at_eer,
        "FRR": frr_at_eer,
        "accuracy": accuracy,
        "n_genuine": n_g,
        "n_impostor": n_i,
        "genuine_mean": float(np.mean(genuine)) if n_g else None,
        "genuine_std": float(np.std(genuine)) if n_g else None,
        "impostor_mean": float(np.mean(impostor)) if n_i else None,
        "impostor_std": float(np.std(impostor)) if n_i else None,
        "margin": float(np.mean(genuine) - np.mean(impostor)) if n_g and n_i else None,
    }
