import os
import cv2
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response, stream_with_context
from deepface import DeepFace
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
from io import BytesIO
import json

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/videos'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4'}
NOTIFICACAO_FOLDER = 'notificacao'
cancel_flag = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pil_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def is_valid_face(face_img):
    return face_img.size > 0 and np.mean(face_img) >= 0.01

def save_temp_image(img_array, prefix):
    os.makedirs('temp', exist_ok=True)
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

@app.route('/deep/', methods=['GET'])
def form_page():
    return render_template("index.html")

@app.route('/deep/cancel', methods=['POST'])
def cancel_processing():
    global cancel_flag
    cancel_flag = True
    return "Cancelado", 200

@app.route('/deep/process', methods=['POST'])
def process_video():
    global cancel_flag
    cancel_flag = False

    video = request.files.get('video')
    photo = request.files.get('photo')

    if not video or not allowed_file(video.filename) or not photo:
        return "Erro: Arquivos inválidos", 400

    video_filename = secure_filename(video.filename)
    photo_filename = secure_filename(photo.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    photo_path = os.path.join('temp', photo_filename)
    os.makedirs('temp', exist_ok=True)
    video.save(video_path)
    photo.save(photo_path)

    ref_faces = DeepFace.extract_faces(photo_path, enforce_detection=False)
    if not ref_faces:
        os.remove(photo_path)
        os.remove(video_path)
        return Response(
            f"data: {json.dumps({'progress': '❌ Nenhuma face encontrada na foto de referência'})}\n\n",
            mimetype='text/event-stream'
        )

    ref_face_data = []
    for idx, ref in enumerate(ref_faces):
        ref_img = ref['face']
        if not is_valid_face(ref_img):
            continue
        ref_face_data.append({
            'ref_name': f"ref_face_{idx}",
            'ref_img': ref_img,
            'ref_base64': pil_image_to_base64(Image.fromarray((ref_img * 255).astype(np.uint8))),
            'confidence': ref['confidence'],
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
                app.logger.error(f"Erro notificação: {e}")
        os.remove(temp_ref_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = 200
    block_size = total_frames // 10 if total_frames >= 10 else total_frames

    @stream_with_context
    def generate():
        global cancel_flag
        frame_count = 0
        block_idx = 1

        while frame_count < total_frames:
            if cancel_flag:
                cap.release()
                os.remove(photo_path)
                os.remove(video_path)
                cancel_flag = False
                yield f"data: {json.dumps({'progress': '❌ Processamento cancelado'})}\n\n"
                return

            block_end = min(frame_count + block_size, total_frames)

            while frame_count < block_end:
                if cancel_flag:
                    cap.release()
                    os.remove(photo_path)
                    os.remove(video_path)
                    cancel_flag = False
                    yield f"data: {json.dumps({'progress': '❌ Processamento cancelado'})}\n\n"
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    results = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
                    for result in results:
                        face_img = result['face']
                        conf = result['confidence']
                        if conf < 0.6 or not is_valid_face(face_img):
                            continue
                        face_temp = save_temp_image(face_img, f"face_{frame_count}")
                        for ref_face in ref_face_data:
                            ref_temp = save_temp_image(ref_face['ref_img'], f"ref_{ref_face['ref_name']}")
                            try:
                                result = DeepFace.verify(img1_path=face_temp, img2_path=ref_temp, model_name='ArcFace', enforce_detection=False)
                                if 'distance' in result and 0 <= result['distance'] <= 1.04 and result['distance'] < 0.60:
                                    sim = (1 - result['distance']) * 100
                                    match_data = {
                                        'video_base64': pil_image_to_base64(Image.fromarray((face_img * 255).astype(np.uint8))),
                                        'score': result['distance'],
                                        'similarity': sim,
                                        'frame': frame_count,
                                        'confidence': conf,
                                        'ref_name': ref_face['ref_name'],
                                        'ref_base64': ref_face['ref_base64'],
                                        'observacao': ref_face['observacao']
                                    }
                                    # ✅ Envia resultado imediatamente
                                    yield f"data: {json.dumps({'progress': f'Processando bloco {block_idx} de 10', 'matches': [match_data]})}\n\n"
                            finally:
                                if os.path.exists(ref_temp):
                                    os.remove(ref_temp)
                        if os.path.exists(face_temp):
                            os.remove(face_temp)
                except Exception as e:
                    app.logger.error(f"Erro frame {frame_count}: {e}")

                frame_count += frame_step

            # ✅ Final do bloco: informa progresso mesmo sem matches
            yield f"data: {json.dumps({'progress': f'✅ Bloco {block_idx} concluído'})}\n\n"
            block_idx += 1

        cap.release()
        os.remove(photo_path)
        os.remove(video_path)
        cancel_flag = False
        yield f"data: {json.dumps({'progress': '✅ Finalizado'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060, debug=True)
 
