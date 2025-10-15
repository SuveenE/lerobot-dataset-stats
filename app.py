import os
from typing import List

import gradio as gr
from huggingface_hub import HfApi

from get_dataset_stats import get_dataset_stats


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


def fetch_stats_for_selected(selected_datasets: List[str], progress=gr.Progress()):
    """Fetch statistics for selected datasets"""
    if not selected_datasets:
        return "Please select at least one dataset"
    
    token = os.environ.get("HF_TOKEN")
    results = []
    total_episodes = 0
    
    for i, repo_id in enumerate(selected_datasets):
        try:
            progress((i + 1) / len(selected_datasets), desc=f"Processing {repo_id}...")
            stats = get_dataset_stats(repo_id, hf_token=token)
            
            if stats.get("error"):
                results.append(f"❌ {repo_id}: Error - {stats['error']}")
            else:
                episodes = stats['total_episodes']
                total_episodes += episodes
                results.append(f"{repo_id}: {episodes} episodes")
            
        except Exception as e:
            results.append(f"❌ {repo_id}: Error - {str(e)}")
    
    # Build output with total at top
    output = [f"**Total Episodes: {total_episodes}**\n"]
    output.extend(results)
    
    return "\n".join(output)


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
    
    dataset_checkboxes = gr.CheckboxGroup(
        label="Select datasets",
        choices=_initial_choices,
        interactive=True,
    )
    
    fetch_btn = gr.Button("Fetch Statistics", variant="primary")
    
    stats_output = gr.Markdown(
        label="Dataset Statistics",
        value="Select datasets and click 'Fetch Statistics'"
    )
    
    # Event handlers
    def load_datasets_from_org(org_name):
        results = search_datasets_fn(org_name)
        return gr.update(choices=results, value=[])
    
    load_btn.click(
        load_datasets_from_org,
        inputs=org_input,
        outputs=dataset_checkboxes,
    )
    
    fetch_btn.click(
        fetch_stats_for_selected,
        inputs=dataset_checkboxes,
        outputs=stats_output,
    )


if __name__ == "__main__":
    demo.launch()
