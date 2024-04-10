python yolo_quant_flow.py \
    --data cifar10/data.yaml \
    --model-name yolov5s \
    --cfg models/yolov5s.yaml \
    --img-size 32 \
    --batch-size-train 10000 \
    --batch-size-test 10000 \
    --batch-size-onnx 10000 \
    --device 0 \
    --seed 42 \
    --calib-batch-size 10000 \
    --num-calib-batch 2 \
    --num-finetune-epochs 10 \
    --calibrator max 

