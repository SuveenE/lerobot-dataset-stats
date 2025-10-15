import os
from typing import List

import gradio as gr
from huggingface_hub import HfApi

from get_dataset_stats import (
    get_dataset_stats,
    format_stats_display,
    compare_metadata_with_actual,
)


def search_datasets_fn(query: str) -> List[str]:
    """Search for datasets on HuggingFace"""
    api = HfApi()
    try:
        items = api.list_datasets(search=(query or "").strip() or None)
        repo_ids = [getattr(d, "id", None) or getattr(d, "repo_id", None) for d in items]
        repo_ids = [r for r in repo_ids if r]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for r in repo_ids:
            if r not in seen:
                unique.append(r)
                seen.add(r)
        return unique[:500]
    except Exception as e:
        print(f"Error searching datasets: {e}")
        return []


def fetch_stats_fn(repo_id: str, progress=gr.Progress()):
    """Fetch dataset statistics"""
    if not repo_id:
        return ""
    
    try:
        progress(0.3, desc="Fetching dataset info...")
        token = os.environ.get("HF_TOKEN")
        
        progress(0.5, desc="Analyzing files...")
        stats = get_dataset_stats(repo_id, hf_token=token)
        
        progress(0.8, desc="Formatting results...")
        
        # Format main stats display
        stats_display = format_stats_display(stats)
        
        # Format comparison if metadata exists
        comparison_display = ""
        if stats.get("info_metadata"):
            comparison_display = "\n\n" + compare_metadata_with_actual(stats)
        
        # Format episode list (if not too many)
        episodes_list = ""
        if stats["episode_numbers"]:
            episodes = stats["episode_numbers"]
            if len(episodes) <= 100:
                episodes_list = f"\n\n**Episode Numbers:** {', '.join(map(str, episodes))}"
            else:
                episodes_list = f"\n\n**Episode Numbers:** {', '.join(map(str, episodes[:50]))}... (showing first 50 of {len(episodes)})"
        
        progress(1.0, desc="Complete!")
        
        # Combine all into one output
        full_output = stats_display + comparison_display + episodes_list
        return full_output
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error fetching stats: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


# Build the Gradio interface
with gr.Blocks(title="LeRobot Dataset Stats Viewer") as demo:
    gr.Markdown("**View statistics for Hugging Face datasets (LeRobot format).**")
    
    # Load initial datasets
    _initial_choices = search_datasets_fn("griffinlabs-cortex")
    
    with gr.Row():
        org_input = gr.Textbox(
            label="Organization or keyword",
            value="griffinlabs-cortex",
            placeholder="e.g., lerobot, griffinlabs-cortex"
        )
        load_btn = gr.Button("Load Datasets")
    
    dataset_dropdown = gr.Dropdown(
        label="Select dataset",
        choices=_initial_choices,
        interactive=True,
    )
    
    stats_output = gr.Markdown(
        label="Dataset Statistics",
        value="Select a dataset to view statistics"
    )
    
    # Event handlers
    def load_datasets_from_org(org_name):
        results = search_datasets_fn(org_name)
        return gr.update(choices=results, value=None)
    
    load_btn.click(
        load_datasets_from_org,
        inputs=org_input,
        outputs=dataset_dropdown,
    )
    
    dataset_dropdown.change(
        fetch_stats_fn,
        inputs=dataset_dropdown,
        outputs=stats_output,
    )


if __name__ == "__main__":
    demo.launch()
