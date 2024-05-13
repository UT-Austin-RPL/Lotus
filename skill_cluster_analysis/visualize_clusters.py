import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import imageio
import io
import os
import h5py

import base64

from PIL import Image
from libero.libero import benchmark, get_libero_path
from streamlit_plotly_events import plotly_events

# configure streamlit layout
st.set_page_config(layout="wide")

data_name = 'skill_discovery_data/LIBERO-Object/dinov2_libero_object_image_only_10/skill_data/saved_feature_data.hdf5'
tsne_data_name = data_name.replace('.hdf5', '_tsne.hdf5')
datasets_default_path = get_libero_path("datasets")
benchmark_dict = benchmark.get_benchmark_dict()
benchmark_instance = benchmark_dict["libero_object"]()
num_tasks = benchmark_instance.get_num_tasks()
demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]

# load data
with h5py.File(data_name, 'r') as f:
    skill_ids = f['cluster_labels'][...]
    demo_indices = f['demo_indices'][...]
    embeddings = f['embeddings'][...]
    task_ids = f['task_ids'][...]
    seg_start = f['seg_start'][...]
    seg_end = f['seg_end'][...]    

with h5py.File(tsne_data_name, 'r') as f:
    tsne_results = f['tsne_results'][...]

def generate_single_colored_image(index):
    # Convert hex to RGB tuple
    hex_colors = ['#007FA1', 
               '#7BDCB5',
               '#FF5A5F',
               ]
    color = tuple(int(hex_colors[index%3].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Create a new image with the specified color and size
    img = np.array(Image.new("RGB", (100, 100), color))
    return img

def images_to_gif(images, duration=500):
    gif = io.BytesIO()
    images[0].save(gif, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)
    return gif.getvalue()

def images_to_mp4(images, duration=500):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, layers = images[0].shape
    size = (width, height)
    
    mp4 = io.BytesIO()
    with imageio.get_writer(mp4, format='mp4', fps=10) as writer:
        for img in images:
            writer.append_data(np.array(img))
    mp4.seek(0)
    return mp4.getvalue()

data_dict = {}
data_list = []

max_num_skills = len(np.unique(skill_ids))

for i in range(len(tsne_results)):
    point = {
        'x': tsne_results[i][0],
        'y': tsne_results[i][1],
        'task_id': task_ids[i],
        'skill_index': skill_ids[i],
        'demo_index': demo_indices[i],
        'seg_start': seg_start[i],
        'seg_end': seg_end[i],
    }
    data_list.append(point)
    data_dict[tsne_results[i][0], tsne_results[i][1]] = point

colors = ['#007FA1', '#7BDCB5', '#FF5A5F', '#00D084', '#B25116', '#00BEFF', '#02A4D3']

# add a slider
cols = st.columns(2)
base_num_tasks = 6
max_num_tasks = 10

skills = []
for i in range(max_num_skills):
    skills.append(
        {"id": i, "name": f"Skill {i}"}
    )
selected_skill_id = st.session_state.get("selected_skill_id", "All")

with cols[0]:
    selected_num_tasks = st.slider('Lifelong Learning Tasks', min_value=base_num_tasks, max_value=max_num_tasks, value=base_num_tasks, step=1)
    highlight_new_points = st.checkbox('Highlight New Points', value=False)

outer_columns = st.columns((5, 5))
with outer_columns[0]:
    cols = st.columns(1 + len(skills))
    with cols[0]:
        if st.button("Unselect Skill"):
            selected_skill_id = "All"

    for i, skill in enumerate(skills):
        if cols[i+1].button(skill["name"]):
        # if st.button(skill["name"]):
            selected_skill_id = skill["id"]

st.session_state["selected_skill_id"] = selected_skill_id

st.session_state["selected_skill_id"] = selected_skill_id
fig = go.Figure()

if not highlight_new_points or selected_num_tasks == base_num_tasks:

    if st.session_state["selected_skill_id"] == "All":
        for skill_id in np.unique(skill_ids):
            task_indices = np.where(task_ids < selected_num_tasks)[0]
            skill_indices = np.where(skill_ids == skill_id)[0]
            # Find the intersection of the two indices
            skill_indices = np.intersect1d(task_indices, skill_indices)
            # print(skill_indices)
            fig.add_trace(
                go.Scatter(
                    x=tsne_results[skill_indices, 0],
                    y=tsne_results[skill_indices, 1],
                    mode='markers',
                    hoverinfo='text',
                    # text=[f"Task ID: {point_task_id}<br>Skill Index: {point_skill_id}" for (point_task_id, point_skill_id) in zip(task_ids, skill_ids)],            
                    marker=dict(size=7, color=colors[skill_id])
                )
            )
    else:
        skill_id = st.session_state["selected_skill_id"]
        task_indices = np.where(task_ids < selected_num_tasks)[0]
        skill_indices = np.where(skill_ids == skill_id)[0]
        # Find the intersection of the two indices
        skill_indices = np.intersect1d(task_indices, skill_indices)
        # print(skill_indices)
        fig.add_trace(
            go.Scatter(
                x=tsne_results[skill_indices, 0],
                y=tsne_results[skill_indices, 1],
                mode='markers',
                hoverinfo='text',
                # text=[f"Task ID: {point_task_id}<br>Skill Index: {point_skill_id}" for (point_task_id, point_skill_id) in zip(task_ids, skill_ids)],            
                marker=dict(size=7, color=colors[skill_id])
            )
        )        
else:
    old_task_indices = np.where(task_ids < selected_num_tasks-1)[0]
    new_task_indices = np.where(task_ids == selected_num_tasks-1)[0]

    if selected_skill_id != "All":
        skill_indices = np.where(skill_ids == selected_skill_id)[0]
        old_task_indices = np.intersect1d(old_task_indices, skill_indices)
        new_task_indices = np.intersect1d(new_task_indices, skill_indices)

    # Visualize old points
    fig.add_trace(
                go.Scatter(
                    x=tsne_results[old_task_indices, 0],
                    y=tsne_results[old_task_indices, 1],
                    mode='markers',
                    hoverinfo='text',
                    # text=[f"Task ID: {point_task_id}<br>Skill Index: {point_skill_id}" for (point_task_id, point_skill_id) in zip(task_ids, skill_ids)],            
                    marker=dict(size=7, color="#b3cde0")
                )
            )
    # Visualize new points
    fig.add_trace(
                go.Scatter(
                    x=tsne_results[new_task_indices, 0],
                    y=tsne_results[new_task_indices, 1],
                    mode='markers',
                    hoverinfo='text',
                    # text=[f"Task ID: {point_task_id}<br>Skill Index: {point_skill_id}" for (point_task_id, point_skill_id) in zip(task_ids, skill_ids)],            
                    marker=dict(size=7, color="#8C76FF")
                )
            )
fig.update_xaxes(range=[-50, 50])
fig.update_yaxes(range=[-50, 50])

selected_points = plotly_events(fig, click_event=True, hover_event=True)
cols = st.columns(12)

# print(data_dict)
with cols[0]:
    if selected_points:
        x, y = selected_points[0]['x'], selected_points[0]['y']

        x = round(x, 3)
        y = round(y, 3)
        # print(data_dict.keys())
        selected_data = data_dict.get((x, y))
        for key in data_dict.keys():
            if abs(key[0] - x) < 0.001 and abs(key[1] - y) < 0.001:
                selected_data = data_dict[key]
                print(selected_data)
                break
        # st.write("Selected points:", selected_data)
        selected_task_id = selected_data['task_id']
        selected_skill_index = selected_data['skill_index']
        # Highlight the information
    else:
        selected_task_id = None
        selected_skill_index = None
    st.metric("Task ID", selected_task_id)
    st.metric("Skill Index", selected_skill_index)   

with cols[2]:
    st.markdown('<h3>Skills</h3>', unsafe_allow_html=True)
    for i in range(0, max_num_skills, 5):
        # Begin the container div for each row
        row_blocks = "<div style='width: 200px;'>"
        
        for skill_id in np.arange(i, i+5):
            if skill_id < max_num_skills:
                if selected_points and skill_id == selected_skill_index:
                    # The selected task gets highlighted (for instance with a red background)
                    row_blocks += f"<div style='background-color: #7BDCB5;height: 50px; width: 30px; margin: 5px; text-align: center; display: inline-block;'>{skill_id}</div>"
                else:
                    # Other tasks have a neutral background (for instance gray)
                    row_blocks += f"<div style='background-color: gray; height: 50px; width: 30px; margin: 5px; text-align: center; display: inline-block;'>{skill_id}</div>"

        # End the container div for each row
        row_blocks += "</div>"
        
        st.markdown(row_blocks, unsafe_allow_html=True)

if selected_points:    
    st.write(f"Showing images for Task ID: {selected_task_id}")
    demo_index = selected_data['demo_index']

    demo_file = demo_files[selected_task_id]
    start_idx = selected_data['seg_start']
    end_idx = selected_data['seg_end']
    with h5py.File(demo_file, "r") as f:
        images = f[f"data/demo_{demo_index}/obs/agentview_rgb"][()][start_idx:end_idx, ::-1]

# with cols[2]:
    
#         mp4 = images_to_mp4(images)
#         st.video(mp4, format='video/mp4', start_time=0)

# cols = st.columns(12)

image_indices = [0, (end_idx - start_idx) // 2, end_idx - start_idx - 1]
for i in range(len(image_indices)):
    cols[6 + i].image(images[image_indices[i]], caption=f"Image {i+1}", use_column_width=False, width=128)
    # cols[i].image(image_path, caption=f"Image {i+1}", use_column_width=True)
