trtexec --onnx=build/fcn_resnet101_onnx_model.onnx \
    --fp16 \
    --workspace=64 \
    --minShapes=input:1x3x256x256 \
    --optShapes=input:1x3x1026x1282 \
    --maxShapes=input:1x3x1440x2560 \
    --buildOnly \
    --saveEngine=build/fcn_resnet101_engine.trt
