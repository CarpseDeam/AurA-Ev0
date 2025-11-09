import ast
from pathlib import Path

source = Path("src/aura/tools/tool_manager.py").read_text()
tree = ast.parse(source)
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ToolManager")

print("Public methods:")
for node in cls.body:
    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
        args = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            ann = ast.unparse(arg.annotation) if arg.annotation else "None"
            args.append(f"{arg.arg}:{ann}")
        if node.args.vararg:
            ann = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else "None"
            args.append(f"*{node.args.vararg.arg}:{ann}")
        if node.args.kwarg:
            ann = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else "None"
            args.append(f"**{node.args.kwarg.arg}:{ann}")
        ret = ast.unparse(node.returns) if node.returns else "None"
        print(f"- {node.name} (line {node.lineno}) -> {ret}")
        if args:
            print("    " + ", ".join(args))
        else:
            print("    (no args)")
