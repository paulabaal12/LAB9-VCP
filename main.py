import cv2
import time
import numpy as np
import argparse
from ultralytics import YOLO

CONF_DEFAULT = 0.40
IOU_DEFAULT  = 0.45
IMG_SIZE     = 640
MODEL_NAME   = "yolov8n.pt"  # se descarga automaticamente si no existe

np.random.seed(42)
COLORES = np.random.randint(50, 230, size=(80, 3), dtype=np.uint8)



# Calcula FPS en tiempo real
class CalculadorFPS:
    def __init__(self, ventana=30):
        self.ventana   = ventana
        self.tiempos   = []
        self.fps_actual = 0.0
    def tick(self):
        ahora = time.perf_counter()
        self.tiempos.append(ahora)
        if len(self.tiempos) > self.ventana:
            self.tiempos.pop(0)
        if len(self.tiempos) >= 2:
            delta = self.tiempos[-1] - self.tiempos[0]
            if delta > 0:
                self.fps_actual = (len(self.tiempos) - 1) / delta
        return self.fps_actual


# Dibuja bounding boxes y etiquetas usando primitivas OpenCV 
def dibujar_detecciones(frame, resultados, class_names):
    num_det = 0
    boxes   = resultados[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame, 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confianza       = float(box.conf[0])
        id_clase        = int(box.cls[0])
        nombre          = class_names.get(id_clase, "?")
        color = tuple(int(c) for c in COLORES[id_clase % len(COLORES)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        etiqueta = f"{nombre} {confianza:.0%}"
        (w_txt, h_txt), baseline = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_base = max(y1 - 4, h_txt + 4)
        cv2.rectangle(frame, (x1, y_base - h_txt - 4), (x1 + w_txt + 4, y_base + baseline), color, cv2.FILLED)
        cv2.putText(frame, etiqueta, (x1 + 2, y_base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        num_det += 1
    return frame, num_det



# Dibuja HUD con FPS y configuración activa
def dibujar_hud(frame, fps, num_det, conf, iou):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (265, 115), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    if fps >= 25:
        color_fps = (0, 220, 0)
    elif fps >= 15:
        color_fps = (0, 200, 220)
    else:
        color_fps = (0, 60, 220)
    cv2.putText(frame, f"FPS: {fps:.1f}", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_fps, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detectados: {num_det}", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"conf={conf:.2f}  iou={iou:.2f}", (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "POKEDEX", (14, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 200, 255), 1, cv2.LINE_AA)
    return frame



# Script principal POKEDEX
def main():
    parser = argparse.ArgumentParser(description="POKEDEX en tiempo real")
    parser.add_argument("--video", type=str, default=None, help="Ruta al video mp4. Si no se pasa, usa la camara web (indice 0).")
    parser.add_argument("--conf", type=float, default=CONF_DEFAULT, help=f"Umbral de confianza (default: {CONF_DEFAULT})")
    parser.add_argument("--iou", type=float, default=IOU_DEFAULT, help=f"Umbral IoU para NMS (default: {IOU_DEFAULT})")
    args = parser.parse_args()
    fuente = args.video if args.video else 0
    print(f"  Modelo:  {MODEL_NAME}")
    print(f"  Fuente:  {'camara web' if fuente == 0 else args.video}")
    print(f"  conf:    {args.conf}")
    print(f"  iou:     {args.iou}")
    print(f"  Presiona 'q' para salir")
    print("=" * 55)
    print("\nCargando YOLOv8n...")
    # Instancia modelo YOLOv8n (Task 3, punto 2)
    model = YOLO(MODEL_NAME)
    class_names = model.names
    print(f"Modelo listo. {len(class_names)} clases COCO disponibles.\n")
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir la fuente '{fuente}'")
        return
    calc_fps = CalculadorFPS(ventana=30)
    print("Iniciando inferencia en tiempo real...")
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Error leyendo camara.")
                break
        # Inferencia continua con YOLO (Task 3, punto 2 y 3)
        resultados = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=IMG_SIZE, verbose=False)
        frame, num_det = dibujar_detecciones(frame, resultados, class_names)
        fps = calc_fps.tick()
        frame = dibujar_hud(frame, fps, num_det, args.conf, args.iou)
        cv2.imshow("POKEDEX | q = salir", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nSaliendo...")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"FPS promedio final: {calc_fps.fps_actual:.1f}")
    print("Listo.")

if __name__ == "__main__":
    main()
