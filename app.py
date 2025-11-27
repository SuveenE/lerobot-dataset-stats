import os
from typing import List

import gradio as gr
from huggingface_hub import HfApi

from get_dataset_stats import get_dataset_stats


def get_user_organizations() -> List[str]:
    """Get organizations the user is part of"""
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    try:
        # Get the user's info
        user_info = api.whoami(token=token)
        username = user_info["name"]
        
        # Get organizations the user is part of
        orgs = [username]  # Include user's own namespace
        if "orgs" in user_info:
            orgs.extend([org["name"] for org in user_info["orgs"]])
        
        return orgs
    except Exception as e:
        print(f"Error getting user organizations: {e}")
        return []


def search_datasets_fn(org_name: str) -> List[str]:
    """Search for datasets from a specific organization"""
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    try:
        if not org_name:
            return []
        
        # List datasets for the specific organization/user
        items = api.list_datasets(author=org_name, token=token)
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
                results.append(f"{repo_id}: **{episodes}** episodes")
            
        except Exception as e:
            results.append(f"❌ {repo_id}: Error - {str(e)}")
    
    # Build output with total at top (larger font)
    output = [f"## Total Episodes: {total_episodes}\n"]
    output.extend(results)
    
    return "\n\n".join(output)


# Build the Gradio interface
with gr.Blocks(title="LeRobot Dataset Stats Viewer") as demo:
    gr.Markdown("**View statistics for Hugging Face datasets (LeRobot format).**")
    
    # Get user's organizations
    _user_orgs = get_user_organizations()
    _initial_datasets = search_datasets_fn(_user_orgs[0]) if _user_orgs else []
    
    with gr.Row():
        org_dropdown = gr.Dropdown(
            label="Select Organization",
            choices=_user_orgs,
            value=_user_orgs[0] if _user_orgs else None,
            interactive=True,
        )
        load_btn = gr.Button("Load Datasets", variant="secondary")
    
    dataset_checkboxes = gr.CheckboxGroup(
        label="Select datasets",
        choices=_initial_datasets,
        interactive=True,
    )
    
    with gr.Row():
        select_all_btn = gr.Button("Select All", size="sm")
        deselect_all_btn = gr.Button("Deselect All", size="sm")
    
    fetch_btn = gr.Button("Fetch Statistics", variant="primary")
    
    stats_output = gr.Markdown(
        label="Dataset Statistics",
        value="Select datasets and click 'Fetch Statistics'"
    )
    
    # Event handlers
    def load_datasets_from_org(org_name):
        results = search_datasets_fn(org_name)
        return gr.update(choices=results, value=[])
    
    def select_all_datasets(current_choices):
        # Get all available choices from the checkbox group
        return current_choices
    
    def deselect_all_datasets():
        return []
    
    # Load datasets on button click or dropdown change
    load_btn.click(
        load_datasets_from_org,
        inputs=org_dropdown,
        outputs=dataset_checkboxes,
    )
    
    org_dropdown.change(
        load_datasets_from_org,
        inputs=org_dropdown,
        outputs=dataset_checkboxes,
    )
    
    # Select/Deselect all buttons
    select_all_btn.click(
        lambda choices: gr.update(value=choices),
        inputs=dataset_checkboxes,
        outputs=dataset_checkboxes,
    )
    
    deselect_all_btn.click(
        lambda: gr.update(value=[]),
        outputs=dataset_checkboxes,
    )
    
    fetch_btn.click(
        fetch_stats_for_selected,
        inputs=dataset_checkboxes,
        outputs=stats_output,
    )


if __name__ == "__main__":
    demo.launch()
