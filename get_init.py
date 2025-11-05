# gen_init.py
import os
import re

folder = os.path.dirname(os.path.abspath(__file__))
files = [f for f in os.listdir(folder)
         if f.endswith('.py') and f != '__init__.py' and f != 'gen_init.py']

class_pattern = re.compile(r'^\s*class\s+([A-Za-z0-9_]+)\s*(\(|:)')
imports = []
mapping = {}
display = {}

for fname in files:
    modname = fname[:-3]
    path = os.path.join(folder, fname)
    found = False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = class_pattern.match(line)
            if m:
                classname = m.group(1)
                print(f"Found class '{classname}' in file '{fname}'")
                imports.append(f"from .{modname} import {classname}")
                mapping[classname] = classname
                # Generate a display name
                display[classname] = re.sub(r'(?<=.)([A-Z])', r' \1', classname)
                found = True
                break
    if not found:
        print(f"WARNING: No class definition found in '{fname}'")

init_path = os.path.join(folder, '__init__.py')
with open(init_path, 'w', encoding='utf-8') as f:
    f.write('# Auto-generated init for custom nodes\n\n')
    for imp in imports:
        f.write(imp + '\n')
    f.write('\nNODE_CLASS_MAPPINGS = {\n')
    for cls in mapping:
        f.write(f"    '{cls}': {cls},\n")
    f.write('}\n\n')
    f.write('NODE_DISPLAY_NAME_MAPPINGS = {\n')
    for cls, dname in display.items():
        f.write(f"    '{cls}': '{dname}',\n")
    f.write('}\n\n')
    f.write("__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']\n")

print(f"Generated __init__.py with {len(mapping)} node entries.")
