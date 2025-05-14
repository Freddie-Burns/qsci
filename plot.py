from matplotlib import pyplot as plt

def plot_energy_ladders(
        energy_ladders: list[list[float]],
        labels: list[str] = None,
    ):
    level_width = 0.9  # Width of each energy line
    spacing = 1.0  # Horizontal spacing between columns
    margin = 0.2
    colors = "black", "blue", "green"
    labels = "FCI", "Garnet 1", "Garnet 2"

    fig, ax = plt.subplots()
    for col_index, levels in enumerate(energy_ladders):
        x_center = col_index * spacing
        for energy in levels:
            ax.hlines(
                y=energy,
                xmin=x_center - level_width / 2,
                xmax=x_center + level_width / 2,
                color=colors[col_index],
                linewidth=2
            )
        ax.text(x_center, 2.7, f"{labels[col_index]}", ha='center')

    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'LiH Energy Levels')
    ax.set_xlim(-spacing / 2 - margin,
                spacing * (len(energy_ladders) - 0.5) + margin)
    ax.set_ylim(3, 7)
    ax.set_xticks([])
    ax.grid(True)
    plt.show()


def plot_error_line_graph(self):
    pass