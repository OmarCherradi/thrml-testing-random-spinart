import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def make_spin_art(size=12, beta_val=1.2, weight=1.0):
    # 1. Create a grid of spin nodes
    nodes_2d = [[SpinNode() for _ in range(size)] for _ in range(size)]
    flat_nodes = [n for row in nodes_2d for n in row]

    # 2. Connect neighbors on a 2D grid
    edges = []
    for i in range(size):
        for j in range(size):
            if i + 1 < size:
                edges.append((nodes_2d[i][j], nodes_2d[i + 1][j]))
            if j + 1 < size:
                edges.append((nodes_2d[i][j], nodes_2d[i][j + 1]))

    # 3. Ising model parameters
    biases = jnp.zeros((len(flat_nodes),))
    weights = jnp.ones((len(edges),)) * weight
    beta = jnp.array(beta_val)

    model = IsingEBM(flat_nodes, edges, biases, weights, beta)

    # 4. Checkerboard free blocks (even / odd spins)
    free_blocks = [Block(flat_nodes[::2]), Block(flat_nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    # 5. Run sampler
    key = jax.random.key(0)
    k_init, k_sample = jax.random.split(key, 2)

    init_state = hinton_init(k_init, model, free_blocks, ())

    schedule = SamplingSchedule(
        n_warmup=100,
        n_samples=1,
        steps_per_sample=3,
    )

    # state_clamp is [] (no clamped blocks)
    # nodes_to_sample is [Block(flat_nodes)] (collect all spins)
    samples = sample_states(
        k_sample,
        program,
        schedule,
        init_state,
        [],                   # state_clamp
        [Block(flat_nodes)],  # nodes_to_sample
    )

    # samples shape: (n_samples, n_nodes) → (size, size)
    grid = samples[0].reshape(size, size)
    return grid


def print_spin_art(grid):
    for row in grid:
        line = "".join("⬜" if int(x) == 1 else "⬛" for x in row)
        print(line)


if __name__ == "__main__":
    art = make_spin_art(size=12)
    print_spin_art(art)
