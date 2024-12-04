# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}

# set the method name
method=${1} # PERACT_BC or BIMANUAL_PERACT

# set the seed number
seed="0"

train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

# cur_dir=$(pwd)
train_demo_path="path/to/your/dataset"
# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
logdir="path/to/your/logdir"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 1s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

#######
# override hyper-params in config.yaml
#######
batch_size=2
anybimanual=True
augmentation_type="ab" # "standard"
timesteps=1
tasks=[handover,pick_in_one,pick_in_two,lift,press,clothes,pingpong,toothbrush,type,pour]
demo=30
episode_length=10
save_freq=10000
log_freq=1000
task_folder="multi"
replay_path="replay"
training_iterations=100001
wandb_project="AnyBimanual_real"
apply_se3=True
aug_rpy=[0.0,0.0,45.0]


tmux select-pane -t 0 
tmux send-keys "conda activate per_real; 
CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        rlbench.episode_length=${episode_length} \
        framework.save_freq=${save_freq} \
        framework.load_existing_weights=${load_existing_weights} \
        framework.wandb_project=${wandb_project} \
        framework.use_pretrained=${use_pre} \
        framework.use_skill=${skill_predictor} \
        replay.path=${replay_path} \
        replay.task_folder=${task_folder} \
        rlbench.instructions=${instructions} \
        framework.log_freq=${log_freq} \
        replay.timesteps=${timesteps} \
        framework.use_prefix=${prefix} \
        framework.training_iterations=${training_iterations} \
        framework.frozen=${frozen} \
        method.transform_augmentation.apply_se3=${apply_se3} \
        method.transform_augmentation.aug_rpy=${aug_rpy}
"
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}