from pathlib import Path

thread_dir = "workspaces/1/threads/1"

thread_dir = Path(thread_dir)


thread_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = thread_dir.parent.parent.expanduser().resolve()
data_dir = workspace_dir / "data"
analysis_dir = thread_dir / "analysis"
results_dir = thread_dir / "results"
data_dir.mkdir(parents=True, exist_ok=True)
analysis_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

