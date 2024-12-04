import re
from pathlib import Path
import pickle
import tap
import transformers
from tqdm.auto import tqdm
import torch
from typing import Dict, List, Tuple, Literal, Optional
from utils.utils_with_rlbench import RLBenchEnv, task_file_to_task_class

TextEncoder = Literal["bert", "clip"]

class Arguments(tap.Tap):
    tasks: Tuple[str, ...] = ['bimanual_pick_laptop','bimanual_pick_plate','bimanual_straighten_rope',
    'coordinated_lift_ball','coordinated_lift_tray','coordinated_push_box','coordinated_put_bottle_in_fridge','dual_push_buttons',
    'handover_item','bimanual_sweep_to_dustpan','coordinated_take_tray_out_of_oven','handover_item_easy']
    output: Path = 'instructions.pkl'
    batch_size: int = 10
    encoder: TextEncoder = "clip"
    model_max_length: int = 77
    variations: Tuple[int, ...] = (1,)
    device: str = "cuda"
    train_dir: Path = Path("train_dir")
    zero: bool = False
    verbose: bool = False

def parse_int(s):
    return int(re.findall(r"\d+", s)[0])

def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model

def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer

def count_and_set_variations(task_dir: Path) -> List[int]:
    variations = []
    
    for variation_dir in task_dir.iterdir():
        if variation_dir.is_dir() and variation_dir.name.startswith('variation'):
            variation_number = int(variation_dir.name.replace('variation', ''))
            variations.append(variation_number)
    
    return sorted(variations)

def load_instructions_from_train(train_dir: Path, tasks: Tuple[str, ...], variations: Tuple[int, ...]) -> Dict[str, Dict[int, List[str]]]:
    instructions = {}
    
    for task in tasks:
        task_dir = train_dir / task
        variations = count_and_set_variations(task_dir)
        print("variations: ",variations)
        if not variations:
            print(f"No variations found for task {task}")
            continue
        instructions[task] = {}
        
        for variation in variations:
            variation_dir = task_dir / f"variation{variation}"
            pkl_file = variation_dir / "variation_descriptions.pkl"
            
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    variation_instructions = pickle.load(f)
                    print("variation_instructions: ",variation_instructions)
                instructions[task][variation] = variation_instructions

    return instructions

if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder)
    model = model.to(args.device)

    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"),
        headless=True,
    )

    instructions: Dict[str, Dict[int, torch.Tensor]] = {}
    tasks = set(args.tasks)
    loaded_instructions = load_instructions_from_train(args.train_dir, tasks, args.variations)

    for task in tqdm(tasks):
        # task_type = task_file_to_task_class(task)
        # task_inst = env.env.get_task(task_type)._task
        # task_inst.init_task()

        instructions[task] = {}
        task_dir = args.train_dir / task
        variations = count_and_set_variations(task_dir)
        for variation in variations:
            instr = loaded_instructions.get(task, {}).get(variation)

            if instr is None:
                print(f"No instructions found for task {task} variation {variation}")
                continue

            if args.verbose:
                print(task, variation, instr)

            tokens = tokenizer(instr, padding="max_length", truncation=True)["input_ids"]
            lengths = [len(t) for t in tokens]
            if any(l > args.model_max_length for l in lengths):
                raise RuntimeError(f"Too long instructions: {lengths}")

            tokens = torch.tensor(tokens).to(args.device)
            with torch.no_grad():
                pred = model(tokens).last_hidden_state
            instructions[task][variation] = pred.cpu()

    if args.zero:
        for instr_task in instructions.values():
            for variation, instr_var in instr_task.items():
                instr_task[variation].fill_(0)

    print("Instructions:", sum(len(inst) for inst in instructions.values()))

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(instructions, f)