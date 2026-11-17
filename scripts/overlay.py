"""
Syntetizuje prekrytie tvare ruskom alebo okuliarmi na zaklade 5-bodovych
landmarkov z InsightFace detektora (lave oko, prave oko, nos, lavy kutik
ust, pravy kutik ust).
"""
import cv2
import numpy as np


def add_mask(img, landmarks):
    """Prekryje spodnu polovicu tvare (nos dole) jednoduchou chirurgickou
    ruskou - modry elipsovity tvar siahajuci od nosa po bradu.

    Velkost je o 15% vacsia nez povodna verzia (porovnanie s realnym
    Photoshop prekrytim na vlastnych fotkach ukazalo, ze povodna maska bola
    prilis mala a prilis posunuta k brade - nezakryvala nos)."""
    out = img.copy()
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    SCALE = 1.15
    eye_dist = np.linalg.norm(right_eye - left_eye)
    mouth_center = (left_mouth + right_mouth) / 2
    chin_estimate = mouth_center + (mouth_center - nose) * 1.6
    nose_to_chin = np.linalg.norm(chin_estimate - nose)

    # stred masky presne medzi nosom a bradou (50/50), s rovnakym presahom
    # na obe strany - tym padom horny okraj siaha nad spicku nosa (zakryje
    # aj mostik nosa), dolny okraj mierne pod bradu
    center = (nose + chin_estimate) / 2
    width = eye_dist * 1.5 * SCALE
    height = nose_to_chin * 1.3 * SCALE

    axes = (int(width / 2), int(height / 2))
    angle = 0
    color = (190, 170, 150)  # modro-sedy odtien ruska (BGR)
    cv2.ellipse(out, (int(center[0]), int(center[1])), axes, angle,
                0, 360, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.ellipse(out, (int(center[0]), int(center[1])), axes, angle,
                0, 360, (120, 110, 100), thickness=2, lineType=cv2.LINE_AA)
    return out


def add_glasses(img, landmarks):
    """Prekryje oblast oci slnecnymi okuliarmi - JEDEN spojity zaobleny
    obdlznikovy tvar (ramik) cez obe oci, nie dve samostatne elipsy s
    tenkym mostikom.

    Tvar bol prekreslene podla porovnania s realnymi (Photoshop) slnecnymi
    okuliarmi na vlastnych fotkach: tie maju spojity ramik bez medzery pri
    nose, su sirsie (siahaju takmer po spanky) a vyssie (od obocia po
    hornu cast lica), nie okruhle elipsy so samostatnymi sosovkami."""
    out = img.copy()
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    eye_dist = np.linalg.norm(right_eye - left_eye)
    eye_vec = right_eye - left_eye
    angle = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))
    eye_center = (left_eye + right_eye) / 2
    # stred ramu mierne pod stredom oci (sosovky siahaju aj pod ocnu liniu)
    direction = eye_vec / eye_dist
    normal = np.array([-direction[1], direction[0]])
    center = eye_center + normal * (eye_dist * 0.08)

    width = eye_dist * 1.9
    height = eye_dist * 0.52

    lens_color = (20, 20, 20)
    frame_color = (8, 8, 8)

    rect = ((float(center[0]), float(center[1])), (float(width), float(height)), float(angle))
    box_i = np.intp(cv2.boxPoints(rect))

    cv2.fillConvexPoly(out, box_i, lens_color, lineType=cv2.LINE_AA)
    cv2.polylines(out, [box_i], isClosed=True, color=frame_color,
                  thickness=max(2, int(eye_dist * 0.05)), lineType=cv2.LINE_AA)
    return out
