import json
from pathlib import Path

# COCO JSON PATH
coco_path = "result_coco.json"  # 수정
output_dir = Path("/datasets/labels")  # YOLOv8 라벨 저장 위치
output_dir.mkdir(parents=True, exist_ok=True)

# COCO JSON 로딩
with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# image id → image 정보 매핑
images = {img["id"]: img for img in coco["images"]}

labels = {}

for ann in coco["annotations"]:
    image = images[ann["image_id"]]
    w, h = image["width"], image["height"]
    filename = Path(image["file_name"]).with_suffix(".txt").name

    class_id = ann["category_id"] - 1

    for seg in ann["segmentation"]:
        if len(seg) < 6:
            continue

        norm_seg = []
        for i in range(0, len(seg), 2):
            x = seg[i] / w
            y = seg[i + 1] / h
            norm_seg.append(f"{x:.6f} {y:.6f}")

        line = f"{class_id} " + " ".join(norm_seg)
        labels.setdefault(filename, []).append(line)

# 저장
for filename, lines in labels.items():
    label_path = output_dir / filename
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
