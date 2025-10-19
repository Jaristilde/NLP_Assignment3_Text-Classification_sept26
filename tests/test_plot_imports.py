import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def test_plot_creation(tmp_path):
    # Create a simple plot and save it to a temporary file
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_title('test plot')
    out = tmp_path / "plot.png"
    fig.savefig(out)
    plt.close(fig)

    assert out.exists()
