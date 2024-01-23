import torch
import torch.onnx
import torchvision.models as models


def main():
    fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True)

    BATCH_SIZE = 4
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    torch.onnx.export(
        fcn_resnet101,
        dummy_input,
        "build/fcn_resnet101_onnx_model.onnx",
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        verbose=False
    )


if __name__ == '__main__':
    main()
