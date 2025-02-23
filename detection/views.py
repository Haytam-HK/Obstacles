from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import threading
import time
from ultralytics import YOLO
import numpy as np
from pathlib import Path

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

# --- Vues de base ---

def index(request):
    """Vue de la page d'accueil"""
    stop_camera()
    return render(request, 'index.html')

def Temp_reel(request):
    """Vue de la page temps réel"""
    return render(request, 'Temp_reel.html')

def start_camera():
    """Initialise et démarre la caméra"""
    global camera, model
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            raise ValueError("Erreur: Impossible d'ouvrir la caméra.")
        print("Caméra ouverte avec succès.")
        # Configuration de la résolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Chargement du modèle YOLO
        model = YOLO('yolov5su.pt')
        # Démarrage du thread de détection
        detection_thread = threading.Thread(target=detect_objects)
        detection_thread.daemon = True
        detection_thread.start()

def stop_camera():
    """Arrête la caméra et libère les ressources"""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        print("Caméra libérée.")

def detect_objects():
    """Fonction principale de détection d'objets"""
    global camera, detected_objects, lock, model
    while camera and camera.isOpened():
        success, image = camera.read()
        if not success:
            print("Erreur: Impossible de lire le flux vidéo.")
            time.sleep(0.1)
            continue
        try:
            # Détection des objets avec YOLO
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
                    # Obtenir les coordonnées de la boîte englobante
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Calculer le centre de l'objet
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    detected_objects.append({
                        "label": label_fr,
                        "confidence": float(conf),
                        "position": {
                            "center_x": center_x,
                            "center_y": center_y,
                            "box": [int(x1), int(y1), int(x2), int(y2)]
                        }
                    })
        except Exception as e:
            print(f"Erreur lors de la détection d'objets : {e}")
        time.sleep(0.1)  # Petit délai pour éviter une utilisation excessive du CPU

def calculate_movement_suggestion(objects):
    """Calcule la suggestion de mouvement basée sur les objets détectés"""
    if not objects:
        return {
            "movement": "AVANCER",
            "reason": "Aucun obstacle détecté",
            "confidence": 1.0
        }
    # Dimensions de l'image
    SCREEN_WIDTH = 640
    SCREEN_CENTER = SCREEN_WIDTH / 2
    SAFE_DISTANCE = 200  # Distance de sécurité en pixels
    # Sélectionner l'objet avec la plus haute confiance
    highest_confidence_obj = max(objects, key=lambda x: x['confidence'])
    obj_x = highest_confidence_obj['position']['center_x']
    # Calculer la distance par rapport au centre
    distance_from_center = abs(obj_x - SCREEN_CENTER)
    if distance_from_center < SAFE_DISTANCE:
        if obj_x < SCREEN_CENTER:
            return {
                "movement": "DROITE",
                "reason": f"Évitement de {highest_confidence_obj['label']} par la droite",
                "confidence": highest_confidence_obj['confidence']
            }
        else:
            return {
                "movement": "GAUCHE",
                "reason": f"Évitement de {highest_confidence_obj['label']} par la gauche",
                "confidence": highest_confidence_obj['confidence']
            }
    else:
        return {
            "movement": "AVANCER",
            "reason": f"{highest_confidence_obj['label']} détecté mais assez loin",
            "confidence": highest_confidence_obj['confidence']
        }

# --- Générateurs de streaming ---

def gen_camera_stream():
    """Générateur pour le streaming vidéo de la caméra du PC"""
    global camera, detected_objects, lock
    while camera and camera.isOpened():
        success, image = camera.read()
        if not success:
            print("Erreur: Impossible de lire le flux vidéo.")
            break
        # Dessiner les boîtes englobantes et les étiquettes
        with lock:
            for obj in detected_objects:
                box = obj["position"]["box"]
                label = f"{obj['label']} {obj['confidence']:.2f}"
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Encoder l'image en JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            print("Erreur: Impossible d'encoder l'image en JPEG.")
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def video_reel(request):
    """Vue pour le flux vidéo en temps réel depuis la caméra du PC"""
    try:
        start_camera()
        return StreamingHttpResponse(gen_camera_stream(),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except ValueError as e:
        return HttpResponse(str(e), status=500)
    except Exception as e:
        return HttpResponse(f"Erreur inattendue : {e}", status=500)

def get_detected_objects(request):
    """API pour obtenir les objets détectés et les suggestions de mouvement"""
    global detected_objects, lock
    try:
        with lock:
            objects_copy = detected_objects.copy()
            suggestion = calculate_movement_suggestion(objects_copy)
            return JsonResponse({
                "objects": objects_copy,
                "suggestion": suggestion
            })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# --- Classe et générateur pour le streaming de vidéo via fichier ou source alternative ---

class VideoCamera:
    def __init__(self, video_path=None):
        if video_path:
            self.video = cv2.VideoCapture(video_path)
        else:
            self.video = cv2.VideoCapture(0)
        # Charger le modèle YOLOv5
        self.model = YOLO('yolov5s.pt')
        self.last_suggestion = "FORWARD"  # Par défaut, avancer
        self.suggestion_confidence = 0.9
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.obstacles = []  # Pour suivre les obstacles détectés
        self.destination_reached = False
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None, None
        # Faire la détection avec YOLOv5
        results = self.model(image)
        # Identifier les obstacles et générer une suggestion
        self.obstacles = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = results[0].names[int(cls)]
            # Considérer la plupart des objets comme des obstacles à éviter
            if conf > 0.5 and label not in ['road', 'sky', 'grass', 'pavement']:
                self.obstacles.append({
                    'label': label,
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'size': (x2 - x1) * (y2 - y1) / (self.frame_width * self.frame_height),
                    'confidence': float(conf)
                })
                # Dessiner les boîtes de détection (obstacles en rouge)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Générer une suggestion de mouvement basée sur les obstacles
        self._generate_obstacle_avoidance_suggestion()
        # Ajouter la suggestion de mouvement à l'image
        cv2.putText(image, f'Suggestion: {self.last_suggestion} ({self.suggestion_confidence:.2f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        # Ajouter des informations supplémentaires
        cv2.putText(image, f'Obstacles: {len(self.obstacles)}', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # Si la destination est atteinte
        if self.destination_reached:
            cv2.putText(image, 'DESTINATION REACHED!', 
                        (int(self.frame_width/2) - 150, int(self.frame_height/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # Encoder l'image en JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), self.last_suggestion

    def _generate_obstacle_avoidance_suggestion(self):
        """
        Génère des suggestions pour éviter les obstacles et optimiser le chemin
        """
        if not self.obstacles:
            self.last_suggestion = "FORWARD"
            self.suggestion_confidence = 0.9
            return
        dangerous_obstacles = [o for o in self.obstacles if o['size'] > 0.1]
        if dangerous_obstacles:
            biggest_obstacle = max(dangerous_obstacles, key=lambda o: o['size'])
            center_x, _ = biggest_obstacle['center']
            rel_x = center_x / self.frame_width
            if rel_x < 0.5:
                self.last_suggestion = "RIGHT"
                self.suggestion_confidence = 0.85
            else:
                self.last_suggestion = "LEFT"
                self.suggestion_confidence = 0.85
            if biggest_obstacle['size'] > 0.3:
                self.last_suggestion = "STOP"
                self.suggestion_confidence = 0.95
            return
        medium_obstacles = [o for o in self.obstacles if 0.05 < o['size'] < 0.1]
        if medium_obstacles:
            left_count = sum(1 for o in medium_obstacles if o['center'][0] < self.frame_width/2)
            right_count = len(medium_obstacles) - left_count
            if left_count > right_count:
                self.last_suggestion = "RIGHT"
                self.suggestion_confidence = 0.75
            elif right_count > left_count:
                self.last_suggestion = "LEFT"
                self.suggestion_confidence = 0.75
            else:
                self.last_suggestion = "SLOW"
                self.suggestion_confidence = 0.7
            return
        self.last_suggestion = "FORWARD"
        self.suggestion_confidence = 0.8
        
    def check_destination_reached(self, destination_markers):
        """
        Vérifie si le drone a atteint sa destination
        """
        for obstacle in self.obstacles:
            if obstacle['label'] in destination_markers and obstacle['size'] > 0.2:
                self.destination_reached = True
                self.last_suggestion = "LAND"
                self.suggestion_confidence = 1.0
                return True
        return False

def gen_video_stream(camera1):
    """Générateur pour le streaming vidéo depuis une source alternative (fichier, etc.)"""
    destination_markers = ['helipad', 'landing zone', 'H', 'parking']
    while True:
        frame, suggestion = camera1.get_frame()
        if frame is not None:
            camera1.check_destination_reached(destination_markers)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

def video_feed(request):
    """Vue pour le flux vidéo depuis un fichier ou une autre source"""
    video_path = request.session.get('video_path', None)
    return StreamingHttpResponse(gen_video_stream(VideoCamera(video_path)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def get_suggestion(request):
    """
    API pour obtenir la dernière suggestion de mouvement
    """
    video_path = request.session.get('video_path', None)
    camera1 = VideoCamera(video_path)
    _, suggestion = camera1.get_frame()
    return JsonResponse({'suggestion': suggestion})

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        # Créer un dossier 'uploads' s'il n'existe pas
        upload_dir = Path('media/uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarder le fichier
        filename = fs.save(f'uploads/{video_file.name}', video_file)
        uploaded_file_path = fs.path(filename)
        # Stocker le chemin dans la session
        request.session['video_path'] = uploaded_file_path
        return render(request, 'video.html')
    return render(request, 'upload.html')
