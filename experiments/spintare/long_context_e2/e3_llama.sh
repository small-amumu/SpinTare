# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
. "$(cd "$(dirname "$0")/../.." && pwd)/scripts/env.sh"  # KVREL_* paths (see docs/setup.md)
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/common.sh"
export LD_LIBRARY_PATH=$KVREL_LD_PATH
sed -n "/^e3()/,/^}/p" $HERE/e3_gpu1.sh > $KB/e3fn.sh
GPU=1; cd $KB; source $KB/e3fn.sh
waitfree(){ local f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | tr -d " ")
  [ $f -ge 22000 ] && { echo $f; return; }; sleep 120; done; }
L8=${KVREL_MODELS}/Meta-Llama-3.1-8B-Instruct; mkdir -p /dev/shm/e3
e3 llama31_8b $L8 auto ref - ref
e3 llama31_8b $L8 int4_per_token_head int4 -
e3 llama31_8b $L8 fp8 fp8 -
rm -rf /dev/shm/e3/llama31_8b
runlog E3 llama31_8b - 0 "E3 Llama rerun done (GPU1 alone)"
