import utils.mesh_tools as mt
import utils.custom_voronoi as cv
import torch
import os
import argparse
from time import time
import trimesh
import numpy as np


DEFAULTS = {
    "grid_n": 32,
    "samples_fac": 150,
    "lr": 5e-3,
    "epochs": 400,
    "random_mask": .2
}


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Creates a VoroMesh from a given mesh. The user can choose any grid size, the other hyperparameters will adjust to it')
    parser.add_argument('shape_path', type=str, help='path to input mesh')
    parser.add_argument('--output_name', type=str,
                        default='voromesh', help='name of output mesh')
    parser.add_argument('--grid_n', type=int,
                        default=DEFAULTS["grid_n"], help='grid_size')
    parser.add_argument('--samples_fac', type=int, default=DEFAULTS["samples_fac"],
                        help='total number of samples: grid_n**2*samples_fac')
    parser.add_argument('--lr', type=float,
                        default=DEFAULTS["lr"], help='learning rate')
    parser.add_argument('--epochs', type=int,
                        default=DEFAULTS["epochs"], help='epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--random_mask', type=float, default=DEFAULTS["random_mask"],
                        help='proportion of subsampled points for each epoch')
    parser.add_argument('--lr_scheduler', type=bool,
                        default=True, help='decaying lr')
    return parser


def create_voromesh(samples, v, f, grid_n=DEFAULTS["grid_n"], lr=DEFAULTS["lr"], epochs=DEFAULTS["epochs"], device='cuda', random_mask=DEFAULTS["random_mask"], lr_scheduler=True, return_time=False):
    mgrid = mt.mesh_grid(grid_n, True)
    V = cv.VoronoiValues(mgrid)
    # subselect points close to the samples
    V.subselect_cells(cv.mask_relevant_voxels(grid_n, samples))
    tensor_points = torch.tensor(samples, dtype=torch.float32).to(device)
    V.to(device)
    optimizer = torch.optim.Adam(V.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 150, 200, 250], gamma=0.5)
    t0 = time()
    for _ in (range(epochs)):
        cv.train_voronoi(V, tensor_points, optimizer, random_mask)
        if lr_scheduler:
            scheduler.step()
    t0 = time()-t0
    V.set_values_winding(v, f, True)
    if return_time:
        return V, t0
    return V

def import_mesh(fname, device, scale=0.8):
    if fname.endswith(".obj"):
        target_mesh = trimesh.load_mesh(fname, 'obj')
    elif fname.endswith(".stl"):
        target_mesh = trimesh.load_mesh(fname, 'stl')    
    elif fname.endswith(".ply"):
        target_mesh = trimesh.load_mesh(fname, 'ply')
    else:
        raise ValueError(f"unknown mesh file type: {fname}")

    target_vertices, target_faces = target_mesh.vertices,target_mesh.faces
    target_vertices, target_faces = \
        torch.tensor(target_vertices, dtype=torch.float32, device=device), \
        torch.tensor(target_faces, dtype=torch.long, device=device)
    
    # normalize to fit mesh into a sphere of radius [scale];
    if scale > 0:
        target_vertices = target_vertices - target_vertices.mean(dim=0, keepdim=True)
        max_norm = torch.max(torch.norm(target_vertices, dim=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale

    return target_vertices, target_faces

if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    print("Sampling input shape...")

    # rescale;
    target_vertices, target_faces = import_mesh(args.shape_path, args.device, scale=1.0)
    v, f = target_vertices.cpu().numpy(), target_faces.cpu().numpy()
    
    ref_mesh = trimesh.Trimesh(v, f)
    samples, _ = trimesh.sample.sample_surface_even(ref_mesh, 100000)
    samples = np.array(samples)

    v = v.astype(np.float64)

    print("Optimizing VoroMesh...")
    V, comp_time = create_voromesh(samples, v, f, args.grid_n, args.lr,
                        args.epochs, args.device, args.random_mask, args.lr_scheduler, return_time=True)
    print("Exporting VoroMesh...")

    # precise mesh extraction using CGAL
    mt.export_obj(*V.to_mesh(), args.output_name)

    logdir = os.path.join("logs", args.output_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(os.path.join(logdir, "comp_time.txt"), "w") as f:
        f.write(str(comp_time))