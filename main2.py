import gradio as gr
import cv2
import os
import uuid
from ultralytics import YOLO

model = YOLO("/home/user/myenv/YoloV8_test/runs/segment/final10/weights/best.pt")  # 원하는 모델로 교체 가능
# model = YOLO("/home/user/myenv/YoloV8_test/runs/detect/final32/weights/best.pt")

IMAGE_DIR = "/home/user/myenv/YoloV8_test/temp/test_img"
VIDEO_DIR = "/home/user/myenv/YoloV8_test/temp/test_vid"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

def convert_to_browser_friendly(input_path, output_path):
    os.system(f"ffmpeg -y -i {input_path} -vcodec libx264 -crf 23 {output_path}")

def process_video(video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0 or width == 0 or height == 0:
        return None, "⚠️ 비디오 정보를 읽을 수 없습니다."

    os.makedirs("temp", exist_ok=True)
    raw_output_path = f"temp/output_raw_{uuid.uuid4().hex[:8]}.mp4"
    final_output_path = f"temp/output_final_{uuid.uuid4().hex[:8]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    convert_to_browser_friendly(raw_output_path, final_output_path)

    return final_output_path, f"✅ 추론 완료! ({frame_count} 프레임)"

def process_image(image, conf_threshold):
    if image is None:
        return None, "⚠️ 이미지를 업로드하세요."
    results = model(image, conf=conf_threshold)
    annotated = results[0].plot()
    return annotated, "✅ 이미지 추론 완료!"

def list_files(folder, exts):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

image_files = list_files(IMAGE_DIR, (".jpg", ".jpeg", ".png"))
video_files = list_files(VIDEO_DIR, (".mp4", ".avi", ".mov", ".mkv"))

initial_preview_image = os.path.join(IMAGE_DIR, image_files[0]) if image_files else None
initial_preview_video = os.path.join(VIDEO_DIR, video_files[0]) if video_files else None

def get_file_path(file_type, uploaded_file, selected_file):
    if uploaded_file is not None:
        # gr.File 객체는 임시경로(uploaded_file.name) 제공
        return uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    if selected_file:
        return os.path.join(IMAGE_DIR if file_type=="Image" else VIDEO_DIR, selected_file)
    return None

def preview_file(file_type, uploaded_file, selected_img, selected_vid):
    path = get_file_path(file_type, uploaded_file, selected_img if file_type=="Image" else selected_vid)
    if path is None:
        return None, None
    if file_type == "Image":
        return path, None
    else:
        return None, path

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 & 🖼️ YOLO 비디오/이미지 객체 감지 - 업로드 및 서버파일 선택 + 미리보기")

    conf_slider = gr.Slider(0.1, 1.0, value=0.3, label="Confidence Threshold")
    btn = gr.Button("추론 시작")

    file_type = gr.Radio(["Image", "Video"], value="Image", label="파일 유형 선택")

    uploaded_file = gr.File(label="파일 업로드", file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"])

    image_dropdown = gr.Dropdown(choices=image_files, label="서버 이미지 선택", value=image_files[0] if image_files else None, visible=True)
    preview_image = gr.Image(label="미리보기 이미지", interactive=False, visible=True, value=initial_preview_image, elem_id="preview_image", scale=0.3)

    video_dropdown = gr.Dropdown(choices=video_files, label="서버 비디오 선택", value=video_files[0] if video_files else None, visible=False)
    preview_video = gr.Video(label="미리보기 비디오", interactive=False, visible=False, value=initial_preview_video)

    output_image = gr.Image(label="추론 결과 이미지", visible=True)
    output_video = gr.Video(label="추론 결과 비디오", visible=False)

    status_text = gr.Textbox(label="상태 메시지")

    def toggle_ui(ft):
        return (
            gr.update(visible=ft=="Image"),    # image_dropdown
            gr.update(visible=ft=="Video"),    # video_dropdown
            gr.update(visible=ft=="Image"),    # preview_image
            gr.update(visible=ft=="Video"),    # preview_video
            gr.update(visible=ft=="Image"),    # output_image
            gr.update(visible=ft=="Video"),    # output_video
            gr.update(file_types=[".jpg", ".jpeg", ".png"] if ft=="Image" else [".mp4", ".avi", ".mov", ".mkv"]) # uploaded_file file_types
        )

    file_type.change(
        toggle_ui,
        inputs=file_type,
        outputs=[image_dropdown, video_dropdown, preview_image, preview_video, output_image, output_video, uploaded_file]
    )

    def update_preview(file_type, uploaded_file, selected_img, selected_vid):
        img_path, vid_path = preview_file(file_type, uploaded_file, selected_img, selected_vid)
        return img_path, vid_path


    uploaded_file.change(
        update_preview,
        inputs=[file_type, uploaded_file, image_dropdown, video_dropdown],
        outputs=[preview_image, preview_video]
    )
    image_dropdown.change(
        update_preview,
        inputs=[file_type, uploaded_file, image_dropdown, video_dropdown],
        outputs=[preview_image, preview_video]
    )
    video_dropdown.change(
        update_preview,
        inputs=[file_type, uploaded_file, image_dropdown, video_dropdown],
        outputs=[preview_image, preview_video]
    )

    def process_output(file_type, uploaded_file, selected_img, selected_vid, conf):
        path = get_file_path(file_type, uploaded_file, selected_img if file_type=="Image" else selected_vid)
        if path is None:
            return None, None, "⚠️ 파일을 업로드하거나 선택하세요."
        if file_type == "Image":
            img, status = process_image(path, conf)
            return img, None, status
        else:
            vid, status = process_video(path, conf)
            return None, vid, status

    
    btn.click(
        process_output,
        inputs=[file_type, uploaded_file, image_dropdown, video_dropdown, conf_slider],
        outputs=[output_image, output_video, status_text]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
