import torch
import torch.onnx
import torchvision.models as models


def main():
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

    BATCH_SIZE = 32
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)

    torch.onnx.export(
        resnext50_32x4d, dummy_input,
        "build/resnet50_onnx_model.onnx",
        input_names=["input"],
        output_names=["output"],
        verbose=False)


if __name__ == '__main__':
    main()
