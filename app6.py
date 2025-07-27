import os
import cv2
from dotenv import load_dotenv
from flask import Flask, render_template, request
from deepface import DeepFace
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
from io import BytesIO

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/videos'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB

ALLOWED_EXTENSIONS = {'mp4'}
NOTIFICACAO_FOLDER = 'notificacao'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pil_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def is_valid_face(face_img):
    if face_img.size == 0:
        return False
    mean_pixel = np.mean(face_img)
    return mean_pixel >= 0.01

def save_temp_image(img_array, prefix):
    temp_path = os.path.join('temp', f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    Image.fromarray((img_array * 255).astype(np.uint8)).save(temp_path)
    return temp_path

def load_notificacao_faces():
    notificacoes = []
    if os.path.exists(NOTIFICACAO_FOLDER):
        for filename in os.listdir(NOTIFICACAO_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                notificacoes.append(os.path.join(NOTIFICACAO_FOLDER, filename))
    return notificacoes

@app.route('/deep/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files.get('video')
        photo = request.files.get('photo')
        top_matches = []
        other_matches = []

        if video and allowed_file(video.filename) and photo:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = secure_filename(video.filename)
            photo_filename = secure_filename(photo.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            photo_path = os.path.join('temp', photo_filename)
            os.makedirs('temp', exist_ok=True)
            video.save(video_path)
            photo.save(photo_path)

            print(f"✅ Vídeo salvo em: {video_path}", flush=True)
            print(f"✅ Foto de referência salva localmente: {photo_path}", flush=True)

            ref_faces = DeepFace.extract_faces(photo_path, enforce_detection=False)
            ref_face_data = []
            for idx, ref in enumerate(ref_faces):
                ref_img = ref['face']
                if not is_valid_face(ref_img):
                    continue
                ref_pil = Image.fromarray((ref_img * 255).astype(np.uint8))
                ref_base64 = pil_image_to_base64(ref_pil)
                ref_face_data.append({
                    'ref_name': f"ref_face_{idx}",
                    'ref_img': ref_img,
                    'ref_base64': ref_base64,
                    'confidence': ref['confidence'],
                    'matches': [],
                    'observacao': False
                })

            notificacao_paths = load_notificacao_faces()
            for ref_face in ref_face_data:
                temp_ref_path = save_temp_image(ref_face['ref_img'], f"check_ref_{ref_face['ref_name']}")
                for notif_path in notificacao_paths:
                    try:
                        result = DeepFace.verify(img1_path=temp_ref_path, img2_path=notif_path, model_name='ArcFace', enforce_detection=False)
                        if result.get('verified') and result.get('distance') < 0.60:
                            ref_face['observacao'] = True
                            break
                    except Exception as e:
                        print(f"Erro verificando notação para {ref_face['ref_name']}: {e}")
                os.remove(temp_ref_path)

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step = 200
            frame_count = 0

            while cap.isOpened() and frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    results = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
                    if results:
                        for result in results:
                            face_img = result['face']
                            confidence = result['confidence']
                            if confidence < 0.6:
                                continue
                            if not is_valid_face(face_img):
                                continue
                            face_pil = Image.fromarray((face_img * 255).astype(np.uint8))
                            face_base64 = pil_image_to_base64(face_pil)
                            face_temp_path = save_temp_image(face_img, f"temp_face_{frame_count}")
                            for ref_face in ref_face_data:
                                try:
                                    ref_temp_path = save_temp_image(ref_face['ref_img'], f"temp_ref_{ref_face['ref_name']}")
                                    result = DeepFace.verify(img1_path=face_temp_path, img2_path=ref_temp_path, model_name='ArcFace', enforce_detection=False)
                                    if 'distance' not in result:
                                        print(f"⚠️ Nenhum 'distance' retornado para frame {frame_count} com {ref_face['ref_name']}")
                                        continue
                                    score = result['distance']
                                    if not (0 <= score <= 1.04):
                                        print(f"⚠️ Score inválido ({score}) para frame {frame_count} com {ref_face['ref_name']}")
                                        continue
                                    similarity = (1 - score) * 100
                                    print(f"Debug: Score = {score:.4f}, Similarity = {similarity:.2f}%")
                                    ref_face['matches'].append({
                                        'video_base64': face_base64,
                                        'score': score,
                                        'similarity': similarity,
                                        'frame': frame_count,
                                        'confidence': confidence
                                    })
                                    print(f"📊 Distância: {score:.4f} (Similaridade: {similarity:.2f}%) para frame {frame_count} com {ref_face['ref_name']}")
                                except Exception as e:
                                    print(f"❌ Erro na verificação de similaridade para frame {frame_count}: {e}")
                                finally:
                                    if os.path.exists(ref_temp_path):
                                        os.remove(ref_temp_path)
                            if os.path.exists(face_temp_path):
                                os.remove(face_temp_path)
                except Exception as e:
                    print(f"❌ Erro na extração de faces do frame {frame_count}: {e}")
                frame_count += frame_step
                if frame_count >= total_frames:
                    break
            cap.release()
            os.remove(photo_path)
            os.remove(video_path)

            for ref_face in ref_face_data:
                sorted_matches = sorted(ref_face['matches'], key=lambda x: x['score'])
                if sorted_matches:
                    top = sorted_matches[0]
                    top['ref_name'] = ref_face['ref_name']
                    top['ref_base64'] = ref_face['ref_base64']
                    top['observacao'] = ref_face['observacao']
                    if top['score'] < 0.60:
                        top_matches.append(top)
                    for other in sorted_matches[1:]:
                        other['ref_name'] = ref_face['ref_name']
                        other['ref_base64'] = ref_face['ref_base64']
                        other['observacao'] = ref_face['observacao']
                        if other['score'] < 0.60:
                            top_matches.append(other)
                        else:
                            other_matches.append(other)

            return render_template("index.html", top_faces=top_matches, other_faces=other_matches)

    return render_template("index.html", top_faces=[], other_faces=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060, debug=True)
 
 
