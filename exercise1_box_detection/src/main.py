from pathlib import Path
import scipy.io as sio
import numpy as np
from datetime import datetime
from scipy.ndimage import binary_opening, binary_closing
from visualization import extract_data, show_all, show_allalg
from ransac import ransac_plane
from mlesac import mlesac_plane
from calculate import calculate_box_dimensions
from preemptive_ransac import preemptive_ransac_plane


def main():

    algorithms = {
        "RANSAC":{
            "func": ransac_plane,
            "floor_params":{"threshold":0.05, "max_iterations":1500},
            "top_params":{"threshold":0.01, "max_iterations":3000},
            },
        "MLESAC":{
            "func": mlesac_plane,
            "floor_params":{"threshold":0.05,"gamma":0.03, "max_iterations":1500},
            "top_params":{"threshold":0.01, "gamma":0.03, "max_iterations":3000},
        },
        "Preemptive_ransac":{
            "func": preemptive_ransac_plane,
            "floor_params":{"M":100, "B":100, "threshold":0.05, "gamma":0.05},
            "top_params":{"M":100, "B":100, "threshold":0.05, "gamma":0.05},
        },
    }

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent.parent # exercise1_box_detection 
    results_dir = base_dir / "results" /run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    open(results_dir / "results.txt", "w").close()

    with open(results_dir / "results.txt", "a") as f:
        f.write("===== Parameters =====\n")

        for name, cfg in algorithms.items():
            f.write(f"Algorithm: {name}\n")
            f.write(f"Floor params:{cfg['floor_params']}\n")
            f.write("Morphology: opening=3x3, closing=3x3\n")
            f.write(f"Top_params:{cfg['top_params']}\n\n\n")
        
        f.write("======================\n\n")


    for i in range(1, 5):
        save_path = results_dir / f"example{i}.png"
        #print("save_path:", save_path)

        path = base_dir / "data" /f"example{i}kinect.mat"
        #print("data path:",path)
        A, D, PC = extract_data(path)
        print("\nShapes:")
        print("A shape:", A.shape)
        print("D shape:", D.shape)
        print("PC shape:", PC.shape)


        valid_mask = PC[:,:,2]!=0
        points = PC[valid_mask] 
        H, W, _ = PC.shape

        results = {}

        for name, cfg, in algorithms.items():
            print(f"running {name}........")

            # floor plane
            floor_normal, floor_d, floor_inlier_mask = cfg["func"](points, **cfg["floor_params"])

            floor_mask = np.zeros((H,W), dtype=np.uint8)
            floor_mask[valid_mask] = floor_inlier_mask.astype(np.uint8)

            # filter floor mask
            mask_open = binary_opening(floor_mask.astype(bool), structure=np.ones((3,3)))
            mask_clean = binary_closing(mask_open, structure=np.ones((3,3)))

            #  non-floor points
            non_floor_mask = (mask_clean == 0)& valid_mask
            non_floor_points = PC[non_floor_mask]

            # top floor plane
            top_normal, top_d, top_inlier_mask = cfg["func"](non_floor_points, **cfg["floor_params"])

            top_mask = np.zeros((H,W), dtype=np.uint8)
            top_mask[non_floor_mask] = top_inlier_mask.astype(np.uint8)

            
            results[name] = {
                "floor_normal": floor_normal,
                "floor_d":floor_d,
                "floor_inlier_count": floor_inlier_mask.sum(),
                "floor_mask":floor_mask,
                "mask_clean":mask_clean,
                "non_floor_mask":non_floor_mask,
                "top_normal":top_normal, 
                "top_d":top_d,
                "top_inlier_count":top_inlier_mask.sum(),
                "top_mask":top_mask,
            }
            length, width, height = calculate_box_dimensions(non_floor_points, top_normal, top_d)
            # calculate box dimension using ... algorithm
            with open(results_dir / "results.txt", "a") as f:
                f.write(
                    f"Box Dimensions of Example {i} using {name} : \n"
                    f"Length={length:.4f}, "
                    f"Width={width:.4f}, "
                    f"Height={height:.4f}\n\n"
                )
        with open(results_dir / "results.txt", "a") as f:
            f.write("----------------------------\n\n")


        # visulization
        # produce list of all mask and their titles
        algs = ["RANSAC", "MLESAC", "Preemptive_ransac"]
        keys = ["floor_mask", "mask_clean", "top_mask"]

        all_masks = [results[a][k] for a in algs for k in keys]
        all_titles = [f"{a}_{k}" for a in algs for k in keys]

        show_allalg(A, D, PC,all_masks, all_titles, th=i, stride=10, save_path=save_path)



if __name__== "__main__":
    main()


