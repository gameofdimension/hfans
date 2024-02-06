from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def hello_world(accelerator):
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)


def prepare(accelerator):
    # 10*10 Prompts.
    # Source:
    # https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
    prompts_all = [
        "The King is dead. Long live the Queen.",
        "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",  # noqa
        "The story so far: in the beginning, the universe was created.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",  # noqa
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",  # noqa
        "The sweat wis lashing oafay Sick Boy; he wis trembling.",
        "124 was spiteful. Full of Baby's venom.",
        "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",  # noqa
        "I write this sitting in the kitchen sink.",
        "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",  # noqa
    ] * 10

    # load a base model and tokenizer
    model_path = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return prompts_all, model, tokenizer

# batch, left pad (for inference), and tokenize


def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches = [prompts[i:i + batch_size]
               for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok


def batch_infer(accelerator, prompts_all, model, tokenizer):
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=16)

        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(
                **prompts_tokenized, max_new_tokens=100)

            # remove prompt from gen. tokens
            outputs_tokenized = [
                tok_out[len(tok_in):]
                for tok_in, tok_out in zip(
                    prompts_tokenized["input_ids"], outputs_tokenized)]

            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizer.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)  # type: ignore
            results["num_tokens"] += num_tokens  # type: ignore

        # transform to list, otherwise gather_object()
        # will not collect correctly
        results = [results]

    # collect results from all the GPUs
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time()-start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])

        print(
            f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, " +
            "fnum_tokens {num_tokens}")


def main():
    '''
    使用 Accelerator 来拉起协调多个 gpu 做 inference
    一个 batch 的推理在一个 gpu 上进行，并没有分散到各个 gpu 上
    '''
    accelerator = Accelerator()
    # hello_world(accelerator)

    prompts_all, model, tokenizer = prepare(accelerator)
    batch_infer(accelerator, prompts_all, model, tokenizer)


if __name__ == '__main__':
    # https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db
    # accelerate launch --num_processes=2 cluster_infer.py
    main()
