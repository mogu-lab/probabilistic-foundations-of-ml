import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from matplotlib_venn import venn2


def display_joint_probability_illustration():
    fig, ax = plt.subplots(1, 1, figsize=(2, 4))
    
    out = venn2(
        subsets=(2, 2, 1),
        set_labels=('$A = a$', '$B = b$', '$A = a, B = b$'),
        ax=ax,
    )
    
    for idx, subset in enumerate(out.subset_labels):
        out.subset_labels[idx].set_visible(False)
        
    plt.show()

    
