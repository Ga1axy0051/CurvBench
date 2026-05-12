import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
import os

def plot_curves(save_dir):
    """
    加载保存的预测值和标签，并绘制 ROC 和 PR 曲线。
    """
    
    preds_path = os.path.join(save_dir, 'predictions.npy')
    labels_path = os.path.join(save_dir, 'labels.npy')

    
    if not os.path.exists(preds_path) or not os.path.exists(labels_path):
        print(f"Error: predictions.npy or labels.npy not found in {save_dir}")
        return

    
    predictions = np.load(preds_path)
    labels = np.load(labels_path)

    
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    ap = average_precision_score(labels, predictions)
    
    plt.subplot(1, 2, 2)
    plt.step(recall, precision, where='post', color='b', alpha=0.7, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    
    plt.suptitle(f'Evaluation Curves from {os.path.basename(save_dir)}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(save_dir, 'evaluation_curves.png')
    plt.savefig(output_path)
    print(f"Saved evaluation curves to {output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot evaluation curves.')
    parser.add_argument('save_dir', type=str, help='The directory where predictions.npy and labels.npy are saved.')
    
    args = parser.parse_args()
    plot_curves(args.save_dir)
