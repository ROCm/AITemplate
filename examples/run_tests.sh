#!/bin/bash 
#
# this is a script to run tests during ROCM CI
# input argument:
# Hugging Face token

export HF_TOKEN=$1
export TRANSFORMERS_CACHE=/.cache/huggingface/hub
export NUM_BUILDERS=$(($(nproc)/4))

function print_log_header(){
	rm -f $1;
    export GIT_BRANCH=${GITHUB_BASE_REF:-${GITHUB_REF#refs/heads/}}
    echo -n "hostname: " >$1; hostname >> $1;
    echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1
    rocminfo | grep "Compute Unit:" >> $1
    echo "git_branch: $GIT_BRANCH" >> $1
    git show --summary | grep commit >> $1
    /opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1
}

echo "Running RESNET50 tests"
cd 01_resnet-50
print_log_header resnet50.log
#HIP_VISIBLE_DEVICES=0 python3 benchmark_ait.py 2>&1 | tee -a resnet50.log

echo "Running BERT tests"
cd ../03_bert
print_log_header bert.log
#for sq in 64 128 384 512 1024
#do
#    HIP_VISIBLE_DEVICES=0,1 python3 benchmark_ait.py --seq-length $sq 2>&1 | tee -a bert.log
#done

echo "Running VIT tests"
cd ../04_vit
print_log_header vit.log
#HIP_VISIBLE_DEVICES=0,1 python3 benchmark_ait.py 2>&1 | tee -a vit.log
# test 2 gcd
#for BATCH_SIZE in 1 2 4 8 16 32 64 128 256
#do
#    HIP_VISIBLE_DEVICES=0 python3 benchmark_ait.py --batch-size $BATCH_SIZE 2>&1 | tee -a vit.log &
#    HIP_VISIBLE_DEVICES=1 python3 benchmark_ait.py --batch-size $BATCH_SIZE 2>&1 | tee -a vit.log
#done

echo "Running Stable Diffusion tests"
cd ../05_stable_diffusion
print_log_header sdiff.log
HIP_VISIBLE_DEVICES=0 python3 compile.py --token $HF_TOKEN 2>&1 | tee -a sdiff.log
HIP_VISIBLE_DEVICES=0 python3 demo.py --token $HF_TOKEN --benchmark 1 2>&1 | tee -a sdiff.log
