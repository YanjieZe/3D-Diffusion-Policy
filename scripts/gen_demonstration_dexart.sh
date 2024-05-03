# bash scripts/gen_demonstration_dexart.sh laptop
# bash scripts/gen_demonstration_dexart.sh faucet
# bash scripts/gen_demonstration_dexart.sh bucket
# bash scripts/gen_demonstration_dexart.sh toilet


cd third_party/dexart-release

task_name=${1}
num_episodes=100
root_dir=../../3D-Diffusion-Policy/data/

CUDA_VISIBLE_DEVICES=2 python examples/gen_demonstration_expert.py --task_name=${task_name} \
            --checkpoint_path assets/rl_checkpoints/${task_name}/${task_name}_nopretrain_0.zip \
            --num_episodes $num_episodes \
            --root_dir $root_dir \
            --img_size 84 \
            --num_points 1024
