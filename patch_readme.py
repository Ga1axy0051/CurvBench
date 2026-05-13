import re

with open('/data/hxz/WXY/CurvBench/README.md', 'r') as f:
    code = f.read()

tree_patch = """│   ├── mdgpt/              # MDGPT
│   ├── SAMGPT/             # SAMGPT (Newly integrated)
│   ├── GraphGlue/          # GraphGlue (Newly integrated)
│   └── SA2GFM/             # SA2GFM (Newly integrated)"""
code = code.replace("│   ├── mdgpt/              # MDGPT\n│   └── SAMGPT/             # SAMGPT (Newly integrated)", tree_patch)

sh_patch = """python main.py --model samgpt --dataset f1 --shot_num 1
python main.py --model graphglue --dataset cora --shot_num 5
python main.py --model sa2gfm --dataset citeseer --shot_num 1"""
code = code.replace("python main.py --model samgpt --dataset f1 --shot_num 1", sh_patch)

models_patch = "- **Graph Foundation Models:** `gcope`, `mdgfm`, `mdgpt`, `samgpt`, `graphglue`, `sa2gfm`"
code = code.replace("- **Graph Foundation Models:** `gcope`, `mdgfm`, `mdgpt`, `samgpt`", models_patch)

with open('/data/hxz/WXY/CurvBench/README.md', 'w') as f:
    f.write(code)
