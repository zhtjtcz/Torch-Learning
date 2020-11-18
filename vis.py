import matplotlib.pyplot as plt


def plot_curves(train_acc=None, test_acc=None, train_loss=None, test_loss=None, lr=None):
    plt.figure(num='result', figsize=(8.5, 6))
    
    if train_acc is not None:
        plt.subplot(2, 2, 1)
        plt.tight_layout(pad=2.5)
        plt.plot([it[0] for it in train_acc], [it[1] for it in train_acc],
                 label='train acc', c='steelblue')
        if test_acc is not None:
            plt.plot([it[0] for it in test_acc], [it[1] for it in test_acc],
                     label='test acc', c='skyblue')
        plt.xlabel('batch')
        plt.ylabel('acc')
        plt.legend(loc='lower right')
    
    if train_loss is not None:
        plt.subplot(2, 2, 2)
        plt.tight_layout(pad=2.5)
        plt.plot([it[0] for it in train_loss], [it[1] for it in train_loss],
                 label='train loss', c='tomato')
        if test_loss is not None:
            plt.plot([it[0] for it in test_loss], [it[1] for it in test_loss],
                     label='test loss', c='orange')
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
    
    if lr is not None:
        plt.subplot(2, 2, 3)
        plt.tight_layout(pad=2.5)
        plt.plot([it[0] for it in lr], [it[1] for it in lr],
                 label='lr', c='darkviolet')
        plt.xlabel('batch')
        plt.ylabel('learning rate')
        plt.legend(loc='upper right')
    
    plt.show()
