import cv2
import numpy as np

def create_curved_view_full_auto(panorama_image, curvature_factor=0.2, pad_ratio=0.3):
    """
    Erstellt eine stark nach innen gewölbte zylindrische Ansicht,
    ohne dass die Seiten oder Ecken abgeschnitten werden.

    Args:
        panorama_image (np.ndarray): Eingabebild (Panorama)
        curvature_factor (float): Steuert die Wölbung (kleiner = stärker)
        pad_ratio (float): Schwarzer Randanteil um das Originalbild (0.2–0.4 empfohlen)
    """
    # --- Padding hinzufügen ---
    h, w, _ = panorama_image.shape
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    padded = cv2.copyMakeBorder(
        panorama_image, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    h, w, _ = padded.shape

    # --- Focal Length bestimmen (je kleiner, desto stärker die Wölbung) ---
    focal_length = w * curvature_factor

    # --- 🧮 Automatische Skalierung ---
    # Damit die Seiten vollständig sichtbar bleiben
    # (abhängig vom maximalen Winkel, den die Projektion erzeugt)
    max_theta = np.arctan((w / 2) / focal_length)
    safe_scale = (2 * focal_length * max_theta) / w * 1.6  # 1.6 = Sicherheitsfaktor

    output_w = int(w * safe_scale)
    output_h = int(h * safe_scale)

    # --- Mapping-Matrizen vorbereiten ---
    map_x = np.zeros((output_h, output_w), dtype=np.float32)
    map_y = np.zeros((output_h, output_w), dtype=np.float32)

    x_center = output_w / 2
    y_center = output_h / 2

    # --- Zylindrische Projektion (nach innen gewölbt) ---
    for y in range(output_h):
        for x in range(output_w):
            x_norm = (x - x_center) / focal_length
            y_norm = (y - y_center) / focal_length

            theta = np.arctan(x_norm)
            z = np.sqrt(x_norm**2 + 1)

            src_x = focal_length * theta + w / 2
            src_y = focal_length * y_norm / z + h / 2

            if 0 <= src_x < w and 0 <= src_y < h:
                map_x[y, x] = src_x
                map_y[y, x] = src_y
            else:
                map_x[y, x] = -1
                map_y[y, x] = -1

    # --- Remapping anwenden ---
    curved_image = cv2.remap(
        padded, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    return curved_image
