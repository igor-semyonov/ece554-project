python yolo_quant_flow.py --data data/coco.yaml --cfg models/yolov5s.yaml --ckpt-path weights/yolov5s.pt --hyp data/hyp.qat.yaml --skip-layers \
    --calib-batch-size 128 \
    --batch-size-train 128 \
    --batch-size-test 128 \
    --batch-size-onnx 1
