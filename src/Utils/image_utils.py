import cv2
import numpy as np

def apply_barrel_distortion(image, strength=0.3):
    """
    Wendet eine nach innen gewölbte (Barrel) Verzerrung auf ein Bild an.
    
    Args:
        image: Eingabebild (numpy array)
        strength: Stärke der Verzerrung (0.0 - 1.0, empfohlen: 0.2-0.5)
    
    Returns:
        Verzerrtes Bild mit schwarzen Rändern
    """
    h, w = image.shape[:2]
    
    # Erstelle Koordinaten-Mesh
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalisiere Koordinaten auf [-1, 1]
    x_norm = (2 * x - w) / w
    y_norm = (2 * y - h) / h
    
    # Berechne Distanz vom Zentrum
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Barrel-Distortion Formel (nach innen wölben)
    # Je größer strength, desto stärker die Wölbung
    r_distorted = r * (1 + strength * r**2)
    
    # Berechne neue Koordinaten
    theta = np.arctan2(y_norm, x_norm)
    x_distorted = r_distorted * np.cos(theta)
    y_distorted = r_distorted * np.sin(theta)
    
    # Denormalisiere zurück zu Pixel-Koordinaten
    map_x = ((x_distorted + 1) * w / 2).astype(np.float32)
    map_y = ((y_distorted + 1) * h / 2).astype(np.float32)
    
    # Wende Remap an mit schwarzem Hintergrund
    distorted = cv2.remap(image, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))
    
    return distorted


def process_multiple_images(image_paths, output_paths, strength=0.3):
    """
    Verarbeitet mehrere Bilder mit der gleichen Verzerrung.
    
    Args:
        image_paths: Liste von Pfaden zu Eingabebildern
        output_paths: Liste von Pfaden für Ausgabebilder
        strength: Stärke der Verzerrung
    """
    for i, (input_path, output_path) in enumerate(zip(image_paths, output_paths)):
        # Lade Bild
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"Fehler beim Laden von {input_path}")
            continue
        
        # Wende Verzerrung an
        distorted = apply_barrel_distortion(img, strength)
        
        # Speichere Ergebnis
        cv2.imwrite(output_path, distorted)
        print(f"Verarbeitet: {input_path} -> {output_path}")


# Beispiel-Verwendung
if __name__ == "__main__":
    # Einzelnes Bild
    img = cv2.imread('input.jpg')
    
    if img is not None:
        # Verschiedene Stärken testen
        result_weak = apply_barrel_distortion(img, strength=0.2)
        result_medium = apply_barrel_distortion(img, strength=0.3)
        result_strong = apply_barrel_distortion(img, strength=0.5)
        
        cv2.imwrite('output_weak.jpg', result_weak)
        cv2.imwrite('output_medium.jpg', result_medium)
        cv2.imwrite('output_strong.jpg', result_strong)
        print("Einzelbild verarbeitet")
    
    # Mehrere Bilder
    input_images = ['bild1.jpg', 'bild2.jpg', 'bild3.jpg']
    output_images = ['bild1_gewolbt.jpg', 'bild2_gewolbt.jpg', 'bild3_gewolbt.jpg']
    
    process_multiple_images(input_images, output_images, strength=0.3)