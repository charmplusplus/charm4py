import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import glob
import re

def create_gif(input_dir, output_gif_filename, qoi_index):
    """
    Creates a GIF from .npz simulation output files from multiple chares.

    Args:
        input_dir (str): Directory containing the .npz files (e.g., data_iter_XXXXXX_chare_YYY.npz).
        output_gif_filename (str): Name of the output GIF file.
        qoi_index (int): Index of the Quantity of Interest to visualize.
    """
    
    search_pattern = os.path.join(input_dir, "data_iter_*_chare_*.npz")
    all_npz_files = sorted(glob.glob(search_pattern))

    if not all_npz_files:
        print(f"No .npz files found in {input_dir} matching the pattern {search_pattern}")
        return
    print(f"Found {len(all_npz_files)} total chare .npz files to process.")

    iteration_files_metadata = {} 
    filename_pattern = re.compile(r"data_iter_(\d+)_chare_(\d+)_(\d+)\.npz")

    for file_path in all_npz_files:
        basename = os.path.basename(file_path)
        match = filename_pattern.match(basename)
        if not match:
            print(f"Warning: Filename {basename} does not match expected pattern data_iter_XXXXXX_chare_YYY_ZZZ.npz. Skipping.")
            continue
        
        iter_num = int(match.group(1))

        try:
            with np.load(file_path) as data_archive:
                required_keys = ['state', 'etime', 'chare_nx', 'chare_i_beg', 'chare_nz', 'chare_k_beg']
                if not all(key in data_archive for key in required_keys):
                    print(f"Warning: File {file_path} is missing one or more required keys ({', '.join(required_keys)}). Skipping.")
                    continue

                meta = {
                    'path': file_path,
                    'etime': float(data_archive['etime']),
                    'chare_nx': int(data_archive['chare_nx']),
                    'chare_i_beg': int(data_archive['chare_i_beg']),
                    'chare_nz': int(data_archive['chare_nz']),
                    'chare_k_beg': int(data_archive['chare_k_beg'])
                }
            
            if iter_num not in iteration_files_metadata:
                iteration_files_metadata[iter_num] = []
            iteration_files_metadata[iter_num].append(meta)
        except Exception as e:
            print(f"Could not load metadata from {file_path}: {e}")
            continue
            
    if not iteration_files_metadata:
        print("No valid iteration data could be processed from file metadata.")
        return

    sorted_iter_nums = sorted(iteration_files_metadata.keys())
    
    reconstructed_frames_info = [] 
    num_vars_global = None

    print("Reconstructing data for each iteration...")
    for iter_idx, iter_num in enumerate(sorted_iter_nums):
        chare_metas_for_iter = iteration_files_metadata[iter_num]
        if not chare_metas_for_iter: continue

        current_global_nx = 0
        current_global_nz = 0
        sim_time_for_iter = chare_metas_for_iter[0]['etime'] 
        
        temp_chare_data_for_iter = []

        valid_iter = True
        for chare_meta in chare_metas_for_iter:
            current_global_nx = max(current_global_nx, chare_meta['chare_i_beg'] + chare_meta['chare_nx'])
            current_global_nz = max(current_global_nz, chare_meta['chare_k_beg'] + chare_meta['chare_nz'])
            
            try:
                with np.load(chare_meta['path']) as data_archive:
                    state_data_chare = data_archive['state']
                
                if num_vars_global is None:
                    num_vars_global = state_data_chare.shape[0]
                    if qoi_index >= num_vars_global:
                        print(f"Error: QoI index {qoi_index} is out of bounds for data (num_vars={num_vars_global}). Max valid QoI index is {num_vars_global - 1}.")
                        return # Critical error, stop processing
                elif state_data_chare.shape[0] != num_vars_global:
                    print(f"Warning: Inconsistent number of variables in {chare_meta['path']} ({state_data_chare.shape[0]} vs {num_vars_global}). Skipping iteration {iter_num}.")
                    valid_iter = False
                    break 
                
                temp_chare_data_for_iter.append({**chare_meta, 'state': state_data_chare})

            except Exception as e:
                print(f"Could not load state from {chare_meta['path']} for iter {iter_num}: {e}")
                valid_iter = False
                break
        
        if not valid_iter or not temp_chare_data_for_iter:
            print(f"Warning: Skipping iteration {iter_num} due to data loading issues or inconsistencies.")
            continue
        
        if num_vars_global is None: # Should be set if at least one chare was processed
            print(f"Warning: Number of variables could not be determined for iteration {iter_num}. Skipping.")
            continue

        full_state_np = np.zeros((num_vars_global, current_global_nz, current_global_nx), dtype=np.float64)
            
        for data_loaded in temp_chare_data_for_iter:
            s = data_loaded['state']
            i_beg, i_len = data_loaded['chare_i_beg'], data_loaded['chare_nx']
            k_beg, k_len = data_loaded['chare_k_beg'], data_loaded['chare_nz']
            full_state_np[:, k_beg:k_beg+k_len, i_beg:i_beg+i_len] = s
            
        qoi_slice = full_state_np[qoi_index, :, :]
        reconstructed_frames_info.append({
            'iter_num': iter_num,
            'sim_time': sim_time_for_iter,
            'qoi_data': qoi_slice 
        })
        if (iter_idx + 1) % 10 == 0 or (iter_idx + 1) == len(sorted_iter_nums) or len(sorted_iter_nums) < 10 :
             print(f"  Reconstructed data for iteration {iter_num} ({iter_idx+1}/{len(sorted_iter_nums)})")

    if not reconstructed_frames_info:
        print("No simulation frames could be reconstructed. GIF creation aborted.")
        return
        
    vmin, vmax = None, None
    print("Determining color scale from reconstructed data...")
    for i, frame_info in enumerate(reconstructed_frames_info):
        qoi_data = frame_info['qoi_data']
        if i == 0:
            vmin = np.min(qoi_data)
            vmax = np.max(qoi_data)
        else:
            vmin = min(vmin, np.min(qoi_data))
            vmax = max(vmax, np.max(qoi_data))
            
    if vmin is None or vmax is None:
        print("Could not determine color scale from reconstructed data. No valid data files processed or QoI data was empty.")
        return
        
    print(f"Color scale determined: vmin={vmin:.2e}, vmax={vmax:.2e}")

    images = []
    print("Generating images for GIF...")
    for i, frame_info in enumerate(reconstructed_frames_info):
        if (i + 1) % 10 == 0 or (i + 1) == len(reconstructed_frames_info) or len(reconstructed_frames_info) < 10:
            print(f"Processing frame {i+1}/{len(reconstructed_frames_info)} for iter {frame_info['iter_num']}")
        
        qoi_data_to_plot = frame_info['qoi_data']
        sim_time = frame_info['sim_time']
        iter_num_for_title = frame_info['iter_num']

        nz_dim, nx_dim = qoi_data_to_plot.shape
        aspect_ratio = nx_dim / nz_dim if nz_dim > 0 else 1.0

        base_fig_height = 5 
        fig_width = base_fig_height * aspect_ratio
        max_fig_width = 10 
        if fig_width > max_fig_width:
            fig_width = max_fig_width
            base_fig_height = fig_width / aspect_ratio if aspect_ratio > 0 else base_fig_height

        fig, ax = plt.subplots(figsize=(fig_width, base_fig_height))
        im = ax.imshow(qoi_data_to_plot, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f"QoI {qoi_index}")
        
        ax.set_title(f"Sim Time: {sim_time:.3f}s (Iter: {iter_num_for_title}) - QoI {qoi_index}")
        ax.set_xlabel("Global X-index")
        ax.set_ylabel("Global Z-index")

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba() 
        image_rgba = np.frombuffer(buf, dtype=np.uint8)
        canvas_width, canvas_height = fig.canvas.get_width_height() 
        image_rgba = image_rgba.reshape(canvas_height, canvas_width, 4) 
        images.append(image_rgba[:, :, :3]) 

        plt.close(fig)

    if not images:
        print("No images were generated for GIF. Aborting.")
        return

    print(f"Saving GIF to {output_gif_filename}...")
    try:
        imageio.mimsave(output_gif_filename, images, fps=10)
        print(f"Successfully created {output_gif_filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from MiniWeather multi-chare simulation output .npz files.")
    parser.add_argument("input_dir", type=str, help="Directory containing the .npz simulation output files (e.g., data_iter_*_chare_*.npz).")
    parser.add_argument("--out", type=str, default="simulation_qoi0.gif", help="Output GIF filename (default: simulation_qoi0.gif).")
    parser.add_argument("--qoi", type=int, default=0, help="Index of the Quantity of Interest to visualize (default: 0, e.g., density).")
    
    args = parser.parse_args()

    output_filename = args.out
    if args.qoi != 0 and args.out == "simulation_qoi0.gif": 
        output_filename = f"simulation_qoi{args.qoi}.gif"
        
    create_gif(args.input_dir, output_filename, args.qoi) 