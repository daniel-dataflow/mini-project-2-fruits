from common_imports import *
from config import *


def load_and_split_data():
    jsons = list(JSON_DIR.glob("*.json"))
    if not jsons:
        print("✗ JSON 파일을 찾을 수 없습니다!")
        return {}, []

    train, temp = train_test_split(jsons, train_size=TRAIN_CONFIG['train_split'],
                                   random_state=TRAIN_CONFIG['random_seed'])
    val, test = train_test_split(temp, train_size=TRAIN_CONFIG['val_split'],
                                 random_state=TRAIN_CONFIG['random_seed'])

    splits = {'train': [], 'val': [], 'test': []}
    classes, class_to_idx = [], {}

    for split, files in zip(['train', 'val', 'test'], [train, val, test]):
        for j in tqdm(files, desc=f"Loading {split}"):
            with open(j, 'r', encoding='utf-8') as f:
                d = json.load(f)

            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                p = IMG_DIR / f"{j.stem}{ext}"
                if p.exists():
                    img_path = str(p)
                    break
            if not img_path:
                continue

            name = f"{d['cate1']}_{d['cate3']}"
            if name not in classes:
                class_to_idx[name] = len(classes)
                classes.append(name)

            bbox = d['bndbox']
            splits[split].append({
                'image': img_path,
                'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                'label': class_to_idx[name],
                'json_stem': j.stem
            })

    print(f"\\n 데이터 로딩 완료:")
    print(f"  - Train: {len(splits['train'])}개")
    print(f"  - Val: {len(splits['val'])}개")
    print(f"  - Test: {len(splits['test'])}개")
    print(f"  - Classes: {len(classes)}개")

    return splits, classes


def prepare_yolo_format(splits, classes):
    print("\\n YOLO 데이터셋 생성 중...")

    for split, items in splits.items():
        for item in tqdm(items, desc=f"YOLO {split}"):
            with open(item['image'], 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]

            img_save = DATASET_YOLO / 'images' / split / f"{item['json_stem']}.jpg"
            cv2.imwrite(str(img_save), img)

            bbox = item['bbox']
            x_center = (bbox[0] + bbox[2]) / 2 / w
            y_center = (bbox[1] + bbox[3]) / 2 / h
            width = (bbox[2] - bbox[0]) / w
            height = (bbox[3] - bbox[1]) / h

            label_save = DATASET_YOLO / 'labels' / split / f"{item['json_stem']}.txt"
            with open(label_save, 'w') as f:
                f.write(f"{item['label']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")

    data_yaml = {
        'path': str(DATASET_YOLO),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }

    with open(DATASET_YOLO / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    print(f" YOLO 데이터셋 생성 완료: {DATASET_YOLO}")


def create_coco_annotations(splits, classes):
    print("\\n COCO annotations 생성 중")

    coco_categories = [
        {"id": idx, "name": name, "supercategory": "freshness"}
        for idx, name in enumerate(classes)
    ]

    for split in ['train', 'val', 'test']:
        coco_data = {
            'info': {
                "description": f"Custom Dataset - {split} Set",
                "version": "1.0",
                "year": 2025,
            },
            'licenses': [{"id": 0, "name": "Unknown", "url": ""}],
            'categories': coco_categories,
            'images': [],
            'annotations': []
        }

        ann_id = 0

        for img_id, item in enumerate(splits[split]):
            try:
                with open(str(item['image']), 'rb') as f:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
            except Exception as e:
                print(f"✗ 이미지 로드 실패 {item['image']}: {e}")
                continue

            coco_data['images'].append({
                'id': img_id,
                'file_name': str(item['image']),
                'width': w,
                'height': h
            })

            bbox = item['bbox']
            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': item['label'],
                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'iscrowd': 0
            })
            ann_id += 1

        anno_path = DATASET_EFFDET / f'coco_{split}.json'
        with open(anno_path, 'w') as f:
            json.dump(coco_data, f, indent=4)

        print(f"  {split}: {anno_path}")

    return DATASET_EFFDET / 'coco_test.json'


if __name__ == "__main__":
    from config import create_directories
    create_directories()

    splits, classes = load_and_split_data()
    if classes:
        prepare_yolo_format(splits, classes)
        create_coco_annotations(splits, classes)

