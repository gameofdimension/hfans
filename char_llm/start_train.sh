set -xeuo pipefail

if [ "$1" = "ddp" ];then
    # ddp
    exec torchrun --nnodes=1 --nproc_per_node=2 --master_addr=localhost --master_port=30601 --node_rank=0 -m char_llm.ddp_train --data_file=./char_llm/shakespeare.txt --batch_size=2 --wandb_project=gpt-train-lab --max_epoch=500
elif [ "$1" = "fsdp" ];then
    # fsdp
    exec torchrun --nnodes=1 --nproc_per_node="$2" --master_addr=localhost --master_port=30601 --node_rank=0 -m char_llm.fsdp_train --data_file=./char_llm/shakespeare.txt --batch_size=2 --wandb_project=gpt-train-lab --max_epoch=500
else
    # vanilla
    exec python -m char_llm.train --data_file=./char_llm/shakespeare.txt --batch_size=64 --model_type=gpt --wandb_project=gpt-train-lab
fi;