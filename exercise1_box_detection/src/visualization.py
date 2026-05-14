import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy.io as sio
from pathlib import Path
import re

def extract_data(data_path):
    print("data_path", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
        
    data_path = Path(data_path)
    
    match = re.search(r"\d+", data_path.stem)
    number = int(match.group()) if match else None
    
    if number is None:
        raise FileNotFoundError(f"File not found: {data_path}")

    # read data
    data = sio.loadmat(data_path)

    # extract 
    A = data[f"amplitudes{number}"]   # amplitude image
    D = data[f"distances{number}"]   # distance image
    PC = data[f"cloud{number}"] # point cloud
    
    return A, D, PC

    


def show_all(A, D, PC, floor_mask, mask_clean,top_mask, stride=10, save_path=None):
    """
    Show amplitude image, distance image, and point cloud in one figure.
    """
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)


    # --- 1. Amplitude ---
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(A, cmap="gray")
    ax1.set_title("Amplitude Image (A)")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # --- 2. Distance ---
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(D, cmap="jet")
    ax2.set_title("Distance Image (D)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    # --- 3. Point Cloud ---
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")

    points = PC.reshape(-1, 3)
    points = points[points[:, 2] != 0]   # remove invalid points
    points = points[::stride]            # subsample

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax3.scatter(x, y, z, s=1)

    ax3.set_title("Point Cloud")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # --- 4. floor mask ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(floor_mask, cmap="gray")
    ax4.set_title("Floor")
    ax4.axis("off")

    # --- 5. floor mask clean ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(mask_clean, cmap="gray")
    ax5.set_title("Filtered")
    ax5.axis("off")

    # --- 6. top mask ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(top_mask, cmap="gray")
    ax6.set_title("Top")
    ax6.axis("off")

    # save result
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    #plt.tight_layout()
    plt.show()



def show_allalg(A, D, PC, masks_lists, titles_lists, th=1, stride=10, save_path=None):
    """
    Show amplitude image, distance image, and point cloud 
    and floor_mask, filtered_mask, top_mask using different algoruthms
    in one figure.
    """
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)

    fig.canvas.manager.set_window_title(f"results of example{th}kinect.mat")

    # --- 1. Amplitude ---
    ax1 = fig.add_subplot(4, 3, 1)
    im1 = ax1.imshow(A, cmap="gray")
    ax1.set_title("Amplitude Image (A)")
    ax1.axis("off")
    #fig.colorbar(im1, ax=ax1, fraction=0.046)

    # --- 2. Distance ---
    ax2 = fig.add_subplot(4, 3, 2)
    im2 = ax2.imshow(D, cmap="jet")
    ax2.set_title("Distance Image (D)")
    ax2.axis("off")
    #fig.colorbar(im2, ax=ax2, fraction=0.046)

    # --- 3. Point Cloud ---
    ax3 = fig.add_subplot(4, 3, 3, projection="3d")

    points = PC.reshape(-1, 3)
    points = points[points[:, 2] != 0]   # remove invalid points
    points = points[::stride]            # subsample

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax3.scatter(x, y, z, s=1)

    ax3.set_title("Point Cloud")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # plane_mask , filtered_mask top_mask using ransac, mlesac, preemptive_ransac
    for i, (mask, title) in enumerate(zip(masks_lists, titles_lists)):
        ax = fig.add_subplot(4, 3, 4+i)
        ax.imshow(mask, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # save result
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    #plt.tight_layout()
    plt.show()