import numpy as np
from onnx_helper import ONNXClassifierWrapper


def main():
    N_CLASSES = 1000  # Our ResNet-50 is trained on a 1000 class ImageNet task
    BATCH_SIZE = 32
    PRECISION = np.float32

    trt_model = ONNXClassifierWrapper(
        "build/resnet_engine.trt",
        [BATCH_SIZE, N_CLASSES],
        target_dtype=PRECISION
    )
    dummy_input_batch = np.random.normal(
        size=(BATCH_SIZE, 224, 224, 3)).astype(PRECISION)
    predictions = trt_model.predict(dummy_input_batch)
    print(predictions.argmax(axis=1))


if __name__ == '__main__':
    main()
