import os

import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from matplotlib.animation import ArtistAnimation
import jax
import jax.numpy as jnp


OUTPUT_DIR = 'figs'


def animate_gradient_descent(
    fn, 
    start_x=-2.0, 
    x_domain=(-2.0, 2.0, 0.01), 
    iterations=1000, 
    lr=0.1, 
    precision=None,
    tangent_length=1.0,
    history_length=5,
    figsize=(5, 4),
    annotation_loc=(0.05, 0.7),
    name='anim.gif', 
    fps=5,
):
    '''
    Adapted from: https://www.kaggle.com/code/trolukovich/animating-gradien-descent
    '''

    grad_fn = jax.grad(fn)
    
    images = []    
    fig, ax = plt.subplots(figsize=figsize)
    x = jnp.arange(*x_domain)
    trajectory = [float(start_x)]

    # Function plot
    f = plt.plot(x, fn(x), color='blue', label=r'$\mathcal{L}(\theta)$')

    init = plt.scatter(
        [start_x], [fn(start_x)],
        c='black', marker='*', zorder=3.0, label=r'Initial $\theta$',
    )

    for frame in range(iterations):
        # Plot point to track
        fn_at_y = fn(trajectory[-1]) # Y coordinate of point    

        num_points = min(len(trajectory), history_length)
        history = []
        for h in range(num_points):
            plot_cur_legend = (h == num_points - 1 and frame == 1) 
            history.append(plt.scatter(
                trajectory[-num_points + h + 1], fn(trajectory[-num_points + h + 1]), 
                color='r', zorder=2.5, alpha=(h + 1.0) / float(num_points),
                **(dict(label=r'Current $\theta$') if plot_cur_legend else {}),
            ))

        slope = grad_fn(trajectory[-1])
        y_intercept = fn_at_y - slope * trajectory[-1]
        tx = jnp.arange(trajectory[-1] - tangent_length, trajectory[-1] + tangent_length, 2) # X coordinates of tangent line
        ty = slope * tx + y_intercept # Y coordinates of tangent line
        tangent = plt.plot(tx, ty, 'r--')    

        trajectory.append(trajectory[-1] - lr * slope)
        step = abs(trajectory[-1] - trajectory[-2])

        # Plot text info
        bbox_args = dict(boxstyle='round', fc='0.8')
        #arrow_args = dict(arrowstyle = '->', color = 'b', linewidth = 1)
        text = f'Iteration: {frame}\nPoint: ({trajectory[-2]:.2f}, {fn_at_y:.2f})\nSlope: {slope:.2f}\nStep: {step:.4f}'
        text = ax.annotate(text, xy=(trajectory[-2], fn_at_y), xytext=annotation_loc, textcoords='axes fraction', bbox=bbox_args, fontsize=12)

        plt.title('Animation of Gradient Descent')    
        plt.legend()
        
        # Stopping algorithm if desired precision have been met
        if precision is not None and step <= precision:
            text2 = plt.text(0.7, 0.1, 'Local minimum found', fontsize=12, transform=ax.transAxes)
            images.append([f[0], init, tangent[0], text, text2] + history)
            break

        images.append([f[0], init, tangent[0], text] + history)

    anim = ArtistAnimation(fig, images) 
    anim.save(name, writer='imagemagic', fps=fps)


def main():    
    animate_gradient_descent(
        lambda theta: theta ** 2.0,
        start_x=-2.0, 
        x_domain=(-2.0, 2.0, 0.01), 
        iterations=50, 
        lr=0.1, 
        tangent_length=1.0,
        history_length=5,        
        figsize=(5, 4),
        annotation_loc=(0.05, 0.7),
        name=os.path.join(OUTPUT_DIR, 'gradient_descent_quadratic_fn.gif'), 
        fps=5,
    )

    
if __name__ == '__main__':
    main()

