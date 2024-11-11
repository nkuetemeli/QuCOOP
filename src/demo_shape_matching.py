import numpy as np
import scipy.io as sio
import trimesh
import io
from PIL import Image
import matplotlib.pyplot as plt
from QuCOOP.src import qap, utils
from QuCOOP.Qmatch import MatchingFramework, Wsampling
from scipy.optimize import linear_sum_assignment


def load_shape(file):
    shape = sio.loadmat(file)
    faces = shape['S']['TRIV'][0][0] - 1
    vertices = shape['S']['VERT'][0][0]
    geodesics = shape['geodesics']
    descriptors = shape['hks']
    return faces, vertices, geodesics, descriptors



def mesh_to_image(vertices, faces, data, cmap='GnBu_r'):
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Normalize data to the range [0, 1]
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Apply color map
    colormap = plt.get_cmap(cmap)
    vertex_colors = (colormap(norm_data)[:, :3] * 255).astype(np.uint8)  # RGBA to RGB

    # Option 1: Use the color of the first vertex of each face for uniform face color
    face_colors = vertex_colors[mesh.faces[:, 0]]

    # Expand the vertex array so each face has its own unique vertices
    expanded_vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    expanded_faces = np.arange(len(expanded_vertices)).reshape((-1, 3))

    # Create a new mesh with expanded vertices
    mesh = trimesh.Trimesh(vertices=expanded_vertices, faces=expanded_faces, process=False)

    # Assign the face colors to the expanded vertices
    mesh.visual.vertex_colors = np.repeat(face_colors, 3, axis=0)
    # mesh.show()

    # Create a scene
    scene = mesh.scene()

    # Render the scene to an image
    data = scene.save_image(resolution=(300, 400))
    img = np.array(Image.open(io.BytesIO(data)))

    return img


def render(Xfaces, Yfaces, Xvertices, Yvertices, P_list):
    n = Xvertices.shape[0]

    data_base = np.sum(Xvertices, axis=1)

    data = np.eye(n) @ data_base
    ref = mesh_to_image(Xvertices, Xfaces, data, cmap='GnBu' + '_r')

    data = P_list[0] @ Xvertices[:, -1]
    tmp_qmatch = mesh_to_image(Yvertices, Yfaces, data, cmap='GnBu' + '_r')

    data = P_list[1] @ Xvertices[:, -1]
    tmp_ours = mesh_to_image(Yvertices, Yfaces, data, cmap='GnBu' + '_r')

    ###############
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8, 8)
    fig.tight_layout(pad=-3)
    ax[0].imshow(ref)
    ax[0].set_title('Reference')
    ax[0].axis(False)
    ax[1].imshow(tmp_ours)
    ax[1].set_title('Ours')
    ax[1].axis(False)
    ax[2].imshow(tmp_qmatch)
    ax[2].set_title('Q-Match')
    ax[2].axis(False)
    plt.show()
    return


if __name__ == '__main__':
    Xpb_instance, Ypb_instance = ('../data/FaustDS/tr_reg_000.mat',
                                  '../data/FaustDS/tr_reg_001.mat')

    Xfaces, Xvertices, Xgeodesics, Xdescriptors = load_shape(Xpb_instance)
    Yfaces, Yvertices, Ygeodesics, Ydescriptors = load_shape(Ypb_instance)

    n = 502
    Permutation1 = np.eye(n)
    Permutation2 = np.eye(n)
    np.random.shuffle(Permutation2)

    Xvertices = Permutation1 @ Xvertices
    Xgeodesics[:, :] = Permutation1 @ Xgeodesics @ Permutation1.T
    Xdescriptors[:, :] = Permutation1 @ Xdescriptors
    Xfaces = np.dstack([((Permutation1.T @ np.arange(n))[i]).astype(np.int32) *
                        (Xfaces == i).astype(np.int32) for i in np.arange(n)]).sum(axis=2)

    Yvertices = Permutation2 @ Yvertices
    Ygeodesics[:, :] = Permutation2 @ Ygeodesics @ Permutation2.T
    Ydescriptors[:, :] = Permutation2 @ Ydescriptors
    Yfaces = np.dstack([((Permutation2.T @ np.arange(n))[i]).astype(np.int32) *
                        (Yfaces == i).astype(np.int32) for i in np.arange(n)]).sum(axis=2)

    P_star = Permutation2 @ Permutation1.T

    # initial matching by descriptor comparison
    softC = Xdescriptors.dot(Ydescriptors.transpose())
    rows, cols = linear_sum_assignment(-softC)
    C = np.array([rows, cols]).transpose()
    P_init = utils.permutation_idx2mat(C[:, 1]).T

    # Geodesic error
    def geodesics_error(P):
        geodesics_errors = np.array([Ygeodesics[int(i), int(j)] for (i, j) in zip(P @ np.arange(n), P_star @ np.arange(n))]) / np.max(Ygeodesics)
        mean = np.mean(geodesics_errors)
        thresh = np.linspace(0, 1, 100)
        percentage = [100 * np.sum(geodesics_errors <= threshold) / n for threshold in thresh]
        return geodesics_errors, thresh, percentage, mean

    # Set number of worst vertices
    n_worst = 40

    # Q-Match matching
    [C_qmatch, e_qmatch] = MatchingFramework.match(Xgeodesics, Ygeodesics, Xdescriptors, Ydescriptors, n_worst, None)
    P_qmatch = utils.permutation_idx2mat(C_qmatch[:, 1]).T

    # Record results
    geodesics_error_qmatch, thresh_qmatch, percentage, mean = geodesics_error(P_qmatch)

    # My matching
    C_ours = C
    e_ours = []
    for iter in range(30):
        worst_matches, scores = MatchingFramework.evaluateCorrespondences(C_ours, Xgeodesics, Ygeodesics, n_worst)
        worst_vertices = C_ours[worst_matches, :]

        # generate subQ
        Q = Wsampling.flattenW(Wsampling.subproblemW(C_ours, worst_vertices[:, 0], worst_vertices[:, 1], Xgeodesics, Ygeodesics))
        alpha = np.min(np.linalg.eigvalsh(Q)) * 10

        # Start Solver
        qap_instance = qap.QAP(Q, alpha, P_star=None)
        history = qap_instance.solve()

        # update C accordingly
        C_ours[worst_matches, 1] = C_ours[[worst_matches[k] for k in (qap_instance.P_sol @ np.arange(n_worst)).astype(np.int32)], 1]
        e_ours.append(scores.sum())

        # Print loss
        P_ours = utils.permutation_idx2mat(C_ours[:, 1]).T
        print(f'----------------------------------------\n'
              f'Round {iter}: Scores = {e_ours[-1]}, '
              f'Objective = {np.linalg.norm(Xgeodesics - P_ours.T @ Ygeodesics @ P_ours, ord="fro")}\n')
        if iter > 0 and e_ours[-1] == e_ours[-2]:
            break

    # Record results
    geodesics_error_ours, thresh_ours, percentage, mean = geodesics_error(P_ours)

    plt.plot(thresh_ours, percentage, label='Ours')
    plt.plot(thresh_qmatch, percentage, label='Q-Match')
    plt.xlabel('Geodesic Error Threshold')
    plt.ylabel('% Correct Matches')
    plt.legend()
    plt.ylim([-5, 105])
    plt.show()

    # Render
    render(Xfaces, Yfaces, Xvertices, Yvertices,
           [P_qmatch, P_ours],)
