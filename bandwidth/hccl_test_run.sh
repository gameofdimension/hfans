set -xeuo pipefail

command=$1  # all_gather_test/all_reduce_test
nnpu=$2

export INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/mpich-3.2.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich-3.2.1/lib:${INSTALL_DIR}:$LD_LIBRARY_PATH


# https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/HCCLpertest_16_0003.html
mpirun -n $nnpu "$INSTALL_DIR"/tools/hccl_test/bin/$command -b 8K -e 4096M -f 2 -d fp32 -p $nnpu -c 0
