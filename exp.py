from pathlib import Path

state_path = Path(
    "/Users/hamza/dev/operate-ai/src/operate_ai/workspaces/8c5d7d8e-0aa8-4f0e-afcc-50f940c308f8/threads/e5b95cc5-45d8-4595-a52f-e0874fd0be2b/states/19-05-2025_21-17-56.json"
)

prev_thread_id = state_path.parent.parent.name

prev_analysis_dir = state_path.parent.parent / "analysis"

print(prev_analysis_dir)
