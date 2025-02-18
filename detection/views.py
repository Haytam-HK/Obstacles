from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
import cv2
import threading
import time
from ultralytics import YOLO

# Variables globales
camera = None
detected_objects = []
lock = threading.Lock()
model = None

# Dictionnaire de traduction des étiquettes en français
LABELS_FR = {
    "person": "personne",
    "car": "voiture",
    "bicycle": "vélo",
    "dog": "chien",
    "cat": "chat",
    "chair": "chaise",
    "bottle": "bouteille",
    "bird": "oiseau",
    "airplane": "avion",
    "helicopter": "hélicoptère",
    "drone": "drone",
    "tree": "arbre",
    "traffic light": "feu de circulation",
    "pole": "poteau",
    "building": "bâtiment",
    "bridge": "pont",
    "antenna": "antenne",
    "kite": "cerf-volant",
    "parachute": "parachute",
    "power line": "ligne électrique",
    "balloon": "ballon",
}


def index(request):
    stop_camera()
    return render(request, 'index.html')

def Temp_reel(request):
    return render(request, 'Temp_reel.html')

def start_camera():
    global camera, detected_objects, lock, model
    
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            raise ValueError("Erreur: Impossible d'ouvrir la caméra.")
        print("Caméra ouverte avec succès.")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        model = YOLO('yolov5su.pt')
        
        detection_thread = threading.Thread(target=detect_objects, args=(camera, model, detected_objects, lock))
        detection_thread.daemon = True
        detection_thread.start()

def stop_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        print("Caméra libérée.")

def detect_objects(camera, model, detected_objects, lock):
    while camera.isOpened():
        success, image = camera.read()
        if not success:
            print("Erreur: Impossible de lire le flux vidéo.")
            time.sleep(0.1)
            continue

        try:
            results = model(image)
            with lock:
                detected_objects.clear()

                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    if conf < 0.5:  # Ignorer les détections de faible confiance
                        continue
                    
                    cls = box.cls[0].item()
                    label = results[0].names[int(cls)]
                    label_fr = LABELS_FR.get(label, label)
                    
                    detected_objects.append({
                        "label": label_fr,
                        "confidence": float(conf)
                    })

        except Exception as e:
            print(f"Erreur lors de la détection d'objets : {e}")

        time.sleep(0.2)

def gen():
    global camera
    while True:
        if camera is None or not camera.isOpened():
            break

        success, image = camera.read()
        if not success:
            print("Erreur: Impossible de lire le flux vidéo.")
            break

        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            print("Erreur: Impossible d'encoder l'image en JPEG.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def video_reel(request):
    try:
        start_camera()
        return StreamingHttpResponse(gen(),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except ValueError as e:
        return HttpResponse(str(e), status=500)
    except Exception as e:
        return HttpResponse(f"Erreur inattendue : {e}", status=500)

def get_detected_objects(request):
    global detected_objects, lock
    try:
        with lock:
            return JsonResponse({"objects": detected_objects})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
