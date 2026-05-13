import re

with open('/data/hxz/WXY/CurvBench/main.py', 'r') as f:
    code = f.read()

# Add to BASELINE_MAP
baseline_map_patch = """    "samgpt": {
        "dir": "gfm_baselines/SAMGPT/src",
        "script": "execute.py"
    },
    "graphglue": {
        "dir": "gfm_baselines/GraphGlue",
        "script": "main.py"
    },
    "sa2gfm": {
        "dir": "gfm_baselines/SA2GFM",
        "script": "main.py"
    },"""
code = re.sub(r'    "samgpt": \{(.*?)\},', baseline_map_patch, code, flags=re.DOTALL)

# Add run_type for graphglue and sa2gfm
run_type_patch = """        if args.model in ["graphglue", "sa2gfm"]:
            command.extend(["--run_type", "adapt", "--data_name", args.dataset])
        else:
            command.extend(["--dataset", args.dataset])"""
code = code.replace('command.extend([\n            "--dataset", args.dataset\n        ])', run_type_patch)
code = re.sub(r'command\.extend\(\[\n\s+"--dataset", args\.dataset\n\s+\]\)', run_type_patch, code)
code = code.replace('command.extend(["--dataset", args.dataset])', run_type_patch)

# Add shot nums forwarding for graphglue and sa2gfm
shot_num_patch = """        # Forward shot_num to baselines that might support it (mdgfm, gcope, mdgpt, samgpt, graphglue, sa2gfm)
        if args.model in ["gcope", "mdgfm", "mdgpt", "samgpt"]:
            command.extend(["--shot_num", str(args.shot_num)])
        elif args.model in ["graphglue", "sa2gfm"]:
            command.extend(["--k_shot", str(args.shot_num)])"""
code = re.sub(r'        # Forward shot_num.*?command.extend\(\["--shot_num", str\(args.shot_num\)\]\)', shot_num_patch, code, flags=re.DOTALL)

with open('/data/hxz/WXY/CurvBench/main.py', 'w') as f:
    f.write(code)
