"""Static analysis test for TTM nodes - ASCII only."""

import ast
import os

print("=" * 60)
print("TTM Nodes - Syntax and Structure Check")
print("=" * 60)

# Test 1: File existence
print("\n[1] File Existence")
files = [
    "__init__.py",
    "nodes_ttm.py",
    "ttm_conditioning.py",
    "ttm_sampler.py",
    "requirements.txt",
]
for f in files:
    status = "OK  " if os.path.exists(f) else "FAIL"
    print(f"  [{status}] {f}")

# Test 2: Syntax check
print("\n[2] Python Syntax")
for f in ['__init__.py', 'nodes_ttm.py', 'ttm_conditioning.py', 'ttm_sampler.py']:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            ast.parse(file.read())
        print(f"  [OK  ] {f}")
    except SyntaxError as e:
        print(f"  [FAIL] {f}: {e}")

# Test 3: Class structure
print("\n[3] Node Classes")

def has_method(filepath, classname, methodname):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return any(m.name == methodname for m in node.body if isinstance(m, ast.FunctionDef))
    return False

# Check WanTTMConditioning
has_schema1 = has_method('ttm_conditioning.py', 'WanTTMConditioning', 'define_schema')
has_execute1 = has_method('ttm_conditioning.py', 'WanTTMConditioning', 'execute')
print(f"  WanTTMConditioning:")
print(f"    define_schema: {'YES' if has_schema1 else 'NO'}")
print(f"    execute:       {'YES' if has_execute1 else 'NO'}")

# Check WanTTMSamplerComplete
has_schema2 = has_method('ttm_sampler.py', 'WanTTMSamplerComplete', 'define_schema')
has_execute2 = has_method('ttm_sampler.py', 'WanTTMSamplerComplete', 'execute')
print(f"  WanTTMSamplerComplete:")
print(f"    define_schema: {'YES' if has_schema2 else 'NO'}")
print(f"    execute:       {'YES' if has_execute2 else 'NO'}")

# Test 4: Extension structure
print("\n[4] Extension Structure")
with open('nodes_ttm.py', 'r', encoding='utf-8') as f:
    code = f.read()
    has_ext = 'class TTMExtension' in code
    has_entry = 'async def comfy_entrypoint' in code
    has_cond = 'WanTTMConditioning' in code
    has_samp = 'WanTTMSampler' in code
    print(f"  TTMExtension:        {'YES' if has_ext else 'NO'}")
    print(f"  comfy_entrypoint:    {'YES' if has_entry else 'NO'}")
    print(f"  Imports Conditioning: {'YES' if has_cond else 'NO'}")
    print(f"  Imports Sampler:      {'YES' if has_samp else 'NO'}")

# Test 5: __init__.py
print("\n[5] Extension Entry Point")
with open('__init__.py', 'r', encoding='utf-8') as f:
    init_code = f.read()
    has_import = 'comfy_entrypoint' in init_code
    has_all = '__all__' in init_code
    print(f"  Exports comfy_entrypoint: {'YES' if has_import else 'NO'}")
    print(f"  Defines __all__:          {'YES' if has_all else 'NO'}")

# Summary
print("\n" + "=" * 60)
all_good = all([
    has_schema1, has_execute1,
    has_schema2, has_execute2,
    has_ext, has_entry, has_cond, has_samp,
    has_import, has_all
])

if all_good:
    print("RESULT: ALL CHECKS PASSED")
    print("=" * 60)
    print("\nNodes are properly structured:")
    print("  - WanTTMConditioning (conditioning)")
    print("  - WanTTMSampler (sampling)")
    print("\nRestart ComfyUI to load the nodes.")
else:
    print("RESULT: SOME CHECKS FAILED")
    print("Review the output above for details.")

print("=" * 60)
