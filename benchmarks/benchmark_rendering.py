import time
import numpy as np
import matplotlib.pyplot as plt

from src.animation import AnimationEngine


def run_benchmark():

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(13, 9))

    fig.subplots_adjust(bottom=0.34)

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)

    X, Y = np.meshgrid(x, y)

    U = -Y
    V = X

    ax.streamplot(
        X,
        Y,
        U,
        V,
        color=np.hypot(U, V),
        cmap="turbo",
        density=1.5
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    ax.set_title("Rendering Performance Benchmark")

    particle, = ax.plot(
        [0],
        [0],
        "o",
        color="#a3e635",
        markersize=8
    )

    frames = 120

    angles = np.linspace(
        0,
        2 * np.pi,
        frames,
        endpoint=False
    )

    plt.show(block=False)

    fig.canvas.draw()
    fig.canvas.flush_events()

    if not fig.canvas.supports_blit:

        plt.close(fig)

        raise RuntimeError(
            "The current backend does not support blitting"
        )

    print("Running full rendering benchmark...")

    start = time.perf_counter()

    for angle in angles:

        if not plt.fignum_exists(fig.number):

            raise RuntimeError(
                "Benchmark window was closed"
            )

        particle.set_data(
            [3 * np.cos(angle)],
            [3 * np.sin(angle)]
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

    full_time = time.perf_counter() - start

    if not plt.fignum_exists(fig.number):

        raise RuntimeError(
            "Benchmark window was closed"
        )

    print("Running blitting benchmark...")

    animation = AnimationEngine(fig)

    animation.set_artists(particle)

    fig.canvas.draw()
    fig.canvas.flush_events()

    animation.reset_fps()

    start = time.perf_counter()

    for angle in angles:

        if animation.closed or not plt.fignum_exists(fig.number):

            raise RuntimeError(
                "Benchmark window was closed"
            )

        particle.set_data(
            [3 * np.cos(angle)],
            [3 * np.sin(angle)]
        )

        animation.render()

        fig.canvas.flush_events()

    blit_time = time.perf_counter() - start

    statistics = animation.get_statistics()

    if animation.closed or not plt.fignum_exists(fig.number):

        raise RuntimeError(
            "Benchmark window was closed"
        )

    if statistics["blit_draws"] != frames:

        print()
        print("BENCHMARK VALIDATION FAILED")
        print(f"Expected blit draws: {frames}")
        print(f"Actual blit draws: {statistics['blit_draws']}")
        print(f"Full draws: {statistics['full_draws']}")

        plt.close(fig)

        raise RuntimeError(
            "Not all frames were rendered using blitting"
        )

    full_fps = frames / full_time

    blit_fps = frames / blit_time

    speedup = full_time / blit_time

    full_frame_time = 1000 * full_time / frames

    blit_frame_time = 1000 * blit_time / frames

    print()
    print("RENDERING BENCHMARK")
    print("-" * 45)

    print(f"Backend: {plt.get_backend()}")
    print(f"Frames: {frames}")
    print(f"Blitting supported: {fig.canvas.supports_blit}")

    print()
    print("FULL RENDERING")
    print(f"Total time: {full_time:.4f} s")
    print(f"Frame time: {full_frame_time:.4f} ms")
    print(f"FPS: {full_fps:.2f}")

    print()
    print("BLITTING")
    print(f"Total time: {blit_time:.4f} s")
    print(f"Frame time: {blit_frame_time:.4f} ms")
    print(f"FPS: {blit_fps:.2f}")

    print()
    print("VALIDATION")
    print(f"Expected frames: {frames}")
    print(f"Blit draws: {statistics['blit_draws']}")
    print(f"Full draws: {statistics['full_draws']}")

    print()
    print(f"Speedup: {speedup:.2f}x")
    print("Benchmark validation: PASSED")

    print("-" * 45)

    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()