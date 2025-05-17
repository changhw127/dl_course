import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_detection(image, results):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label}: {score:.2f}", color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    plt.show()


