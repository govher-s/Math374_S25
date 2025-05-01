'''
****************************************************************
*** Name       : Govher Sapardurdyyeva                         *
*** Assignment : Project 3                                     *
*** Due Date   : May 9                                         *
*** Description: This project solves systems of equations using*
                Gaussian Elimination with Partial Pivoting for *
                nxn matrices                                   *
****************************************************************
'''
import numpy as np
import matplotlib.pyplot as plt

#display the matrices
matrix_snapshots = []
snapshot_titles = []

def capture_matrix_step(matrix, title):
    matrix_snapshots.append(matrix.copy())
    snapshot_titles.append(title)

#reset the snapshots for the next matrix
def reset_snapshots():
    matrix_snapshots.clear()
    snapshot_titles.clear()

def show_matrix_snapshots_grid(title_prefix, save_path=None):
    num_snapshots = len(matrix_snapshots)
    cols = 3
    rows = (num_snapshots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = axes.flatten()

    for idx, (matrix, title) in enumerate(zip(matrix_snapshots, snapshot_titles)):
        ax = axes[idx]

        n_rows, n_cols = matrix.shape
        cell_colors = np.full((n_rows, n_cols), '#4169e1', dtype=object)
        for i in range(n_rows):
            for j in range(n_cols - 1):
                if abs(matrix[i, j]) < 1e-10:
                    cell_colors[i, j] = '#ffff00'

        for i in range(n_rows):
            for j in range(n_cols):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cell_colors[i, j]))
                ax.text(j + 0.5, i + 0.5, f'{matrix[i, j]:.1f}', ha='center', va='center', fontsize=9, color='black')

        ax.text(-0.7, n_rows / 2, f"{title_prefix} - {title}", va='center', ha='right',
                fontsize=9, rotation=90, color='black', fontweight='bold')

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_aspect('equal')

    for i in range(num_snapshots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=3.5)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

#algorithm for partial pivot
def gaussian_elimination(A, b):
    reset_snapshots()
    n = len(A)
    Ab = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    scale = np.max(np.abs(A), axis=1)

    print("Initial Matrix:")
    print(Ab)
    capture_matrix_step(Ab, "Initial")
# find the pivot
    for k in range(n - 1):
        ratios = np.abs(Ab[k:n, k]) / scale[k:n]
        max_row = np.argmax(ratios) + k
        if Ab[max_row, k] == 0: #if 0, matrix has no unique solution
            raise ValueError("Matrix is singular.")
        if max_row != k: #if not continue
            print(f"\nSwap R{k+1} <-> R{max_row+1}")
            Ab[[k, max_row]] = Ab[[max_row, k]]
            scale[[k, max_row]] = scale[[max_row, k]]
            print(Ab)
            capture_matrix_step(Ab, f"Swap R{k+1}<->{max_row+1}")
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            print(f"R{i+1} = R{i+1} - ({factor:.2f}) * R{k+1}")
            Ab[i, k:] -= factor * Ab[k, k:]
            Ab[i, k] = 0
            print(Ab)
            capture_matrix_step(Ab, f"Elim A[{i+1},{k+1}]")

    return Ab

def back_substitution_verbose(Ab):
    n = Ab.shape[0]
    x = np.zeros(n)
    steps = []

    print("\n*** Back Substitution ***")
    for i in range(n - 1, -1, -1):
        rhs_expr = f"{Ab[i, -1]:.1f}"
        for j in range(i + 1, n):
            rhs_expr += f" - ({Ab[i, j]:.1f} * x{j+1})"
        if i == n - 1:
            rhs_val = Ab[i, -1]
        else:
            rhs_val = Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:])
        x[i] = rhs_val / Ab[i, i]
        step = f"x{i+1} = ({rhs_expr}) / {Ab[i, i]:.1f} = {x[i]:.2f}"
        print(step)
        steps.append(step)

    fig, ax = plt.subplots(figsize=(8, 1.5 * n))
    ax.axis('off')
    for idx, step in enumerate(reversed(steps)):
        ax.text(0.05, 1 - idx * 0.15, step, fontsize=14, va='top')
    plt.tight_layout()
    plt.show()

    return x

# Systems of equations to solve
if __name__ == "__main__":
    # 4x4 system
    A1 = np.array([
        [3, -13, 9, 3],
        [-6, 4, 1, -18],
        [6, -2, 2, 4],
        [12, -8, 6, 10]
    ])
    b1 = np.array([-19, -34, 16, 26])

    Ab1 = gaussian_elimination(A1, b1)
    show_matrix_snapshots_grid("4x4")
    x1 = back_substitution_verbose(Ab1)

    # 3x3 system
    A2 = np.array([
        [1, -2, 3],
        [-1, 3, 0],
        [2, -5, 5]
    ])
    b2 = np.array([9, -4, 17])

    Ab2 = gaussian_elimination(A2, b2)
    show_matrix_snapshots_grid("3x3")
    x2 = back_substitution_verbose(Ab2)
