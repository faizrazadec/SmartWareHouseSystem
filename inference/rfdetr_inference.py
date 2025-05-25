from rfdetr import RFDETRBase
# import supervision as sv
# from PIL import Image

# # ✅ Load the trained model
# model = RFDETRBase(
#     pretrain_weights="/home/faizraza/Personal/Projects/Smart-warehouse-system11/models/rfdetr_model/checkpoint_best_total (1).pth"
# )

# # ✅ Path to test image
# image_path = "/home/faizraza/Personal/Projects/Smart-warehouse-system11/dataset/istockphoto-1215484506-612x612.jpg"
# image = Image.open(image_path)

# # ✅ Run inference
# detections = model.predict(image, threshold=0.5)

# # ✅ Calculate annotation settings
# text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
# thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

# # ✅ Create annotators
# bbox_annotator = sv.BoxAnnotator(thickness=thickness)
# label_annotator = sv.LabelAnnotator(
#     text_color=sv.Color.BLACK,
#     text_scale=text_scale,
#     text_thickness=thickness,
#     smart_position=True
# )

# # ✅ Generate labels
# detections_labels = [
#     f"{model.classes[class_id]} {confidence:.2f}"
#     for class_id, confidence in zip(detections.class_id, detections.confidence)
# ]

# # ✅ Annotate image
# annotated_image = image.copy()
# annotated_image = bbox_annotator.annotate(annotated_image, detections)
# annotated_image = label_annotator.annotate(annotated_image, detections, detections_labels)

# # ✅ Display result
# # sv.plot_image(annotated_image)
# # ✅ Save result
# output_path = "/home/faizraza/Personal/Projects/Smart-warehouse-system11/dataset/output.jpg"
# annotated_image.save(output_path)


# from rfdetr import RFDETRBase

# model = RFDETRBase(pretrain_weights="/home/faizraza/Personal/Projects/Smart-warehouse-system11/models/rfdetr_model/checkpoint_best_total(2).pth")

# detections = model.predict("/home/faizraza/Personal/Projects/Smart-warehouse-system11/dataset/box_dataset/test/-2023-01-13-10-45-52_png.rf.cb3b877fe5534c50303eb8d55fd0d861.jpg")

# print(detections)

# import supervision as sv
# from rfdetr import RFDETRBase
# from PIL import Image

# import supervision as sv



# path, image, annotations = ds[i]
# image = Image.open(path)

# detections = model.predict(image, threshold=0.5)

# text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
# thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

# bbox_annotator = sv.BoxAnnotator(thickness=thickness)
# label_annotator = sv.LabelAnnotator(
#     text_color=sv.Color.BLACK,
#     text_scale=text_scale,
#     text_thickness=thickness,
#     smart_position=True)

# detections_labels = [
#     f"{ds.classes[class_id]} {confidence:.2f}"
#     for class_id, confidence
#     in zip(detections.class_id, detections.confidence)
# ]

# detections_image = image.copy()
# detections_image = bbox_annotator.annotate(detections_image, detections)
# detections_image = label_annotator.annotate(detections_image, detections, detections_labels)


# sv.plot_images_grid(images=detections_image, grid_size=(3, 3), size=(12, 12))

from rfdetr import RFDETRBase
import cv2

def main():
    # 1. load model
    model = RFDETRBase(
        pretrain_weights="/home/faizraza/Personal/Projects/Smart-warehouse-system11/models/"
                        "rfdetr_model/checkpoint_best_total(2).pth"
    )

    # 2. run inference
    img_path = "/home/faizraza/Personal/Projects/Smart-warehouse-system11/dataset/box_dataset/test/" \
               "-2023-01-13-10-45-52_png.rf.cb3b877fe5534c50303eb8d55fd0d861.jpg"
    detections = model.predict(img_path)

    # 3. load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")

    # 4. draw boxes + confidence
    for (x1, y1, x2, y2), score in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    # 5. save output
    out_path = "/home/faizraza/Personal/Projects/Smart-warehouse-system11/detected.jpg"
    cv2.imwrite(out_path, img)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()