from pathlib import Path

prompts_dir = Path("src/operate_ai/prompts")

print((prompts_dir / "task_spec.md").read_text())
