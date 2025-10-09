import cv2 as cv
import os

def stitch_images(image_folder: str, output_path: str = "result.jpg", mode: int = cv.Stitcher_PANORAMA, max_images: int = 4):
    """
    Erstellt ein Panorama aus mehreren Bildern in einem Ordner.

    Args:
        image_folder (str): Pfad zum Ordner mit den Bildern.
        output_path (str): Speicherort für das Ergebnisbild.
        mode (int): Stitcher-Modus (cv.Stitcher_PANORAMA oder cv.Stitcher_SCANS).
        max_images (int): Anzahl der zu verarbeitenden Bilder (Standard: 4).
    """
    # Bilder laden
    imgs = []
    for fname in sorted(os.listdir(image_folder))[:max_images]:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, fname)
            img = cv.imread(img_path)
            if img is not None:
                imgs.append(img)
    
    if len(imgs) < 2:
        print("⚠️  Mindestens zwei Bilder werden benötigt, um ein Panorama zu erstellen.")
        return None
    
    print(f"📸 {len(imgs)} Bilder geladen. Starte Stitching...")

    # Stitcher initialisieren
    stitcher = cv.Stitcher.create(mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print(f"❌ Stitching fehlgeschlagen. Fehlercode = {status}")
        return None
    
    # Ergebnis speichern
    cv.imwrite(output_path, pano)
    print(f"✅ Stitching erfolgreich! Ergebnis gespeichert unter: {output_path}")
    
    return pano
