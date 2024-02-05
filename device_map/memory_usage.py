# https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map
import requests
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def make_device_map(model_id):
    with init_empty_weights():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
        )
        print("device", model.device)  # type: ignore

    device_map = infer_auto_device_map(
        model,  # type: ignore
        # no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16,
        max_memory={0: "10GiB", 1: "25GiB"}
    )
    return device_map


def infer(model_id, prompt, image_file, device_map):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(model_id)
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(
        dtype=torch.float16, device='cuda')
    output = model.generate(  # type: ignore
        **inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True)


def main():
    model_id = "llava-hf/llava-1.5-13b-hf"
    prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

    device_map = make_device_map(model_id)
    output = infer(model_id, prompt, image_file, device_map)
    print(output)


if __name__ == "__main__":
    main()
