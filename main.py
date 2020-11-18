import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloaders, MNIST_img_ch, MNIST_img_size, MNIST_num_classes
from misc import time_str, set_seed, init_params
from models import FCNet
import resnetb
import wideresnet
from vis import plot_curves

USING_GPU = torch.cuda.is_available()

def test(test_ld, net: nn.Module):
    global USING_GPU

    net.eval()
    with torch.no_grad():
        tot_correct, tot_pred, tot_loss, tot_iters = 0, 0, 0., 0
        for (inputs, targets) in test_ld:
            if USING_GPU:
                inputs, targets = inputs.cuda(), targets.cuda()
            logits = net(inputs)
            batch_size = targets.shape[0]
            tot_correct += logits.argmax(dim=1).eq(targets).sum().item()
            tot_pred += batch_size

            tot_loss += F.cross_entropy(logits, targets).item()
            tot_iters += 1
    net.train()

    test_acc = 100 * tot_correct / tot_pred
    test_loss = tot_loss / tot_iters
    return test_acc, test_loss


def main():
    global USING_GPU

    print(f'\n=== cuda is {"" if USING_GPU else "NOT"} available ===\n')
    
    data_root_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', 'mnist'))
    set_seed(0)
    train_loader, test_loader = get_dataloaders(data_root_path=data_root_path, batch_size=128)

    # hyper-parameters:
    BASIC_LR = 5e-4
    MIN_LR = 0.
    WEIGHT_DECAY = 1e-5
    OP_MOMENTUM = 0.9
    EPOCHS = 8
    BATCH_SIZE = 8
    DROP_OUT_RATE = 0.1
    ITERS = len(train_loader)
    print(
        f'=== hyper-params ===\n'
        f'  epochs={EPOCHS}\n'
        f'  train iters={ITERS}\n'
        f'  batch size={BATCH_SIZE}\n'
        f'  cosine lr:{BASIC_LR} -> {MIN_LR}\n'
        f'  weight decay={WEIGHT_DECAY}\n'
        f'  momentum={OP_MOMENTUM}\n'
        f'  drop out={DROP_OUT_RATE}\n'
    )

    set_seed(0)

    '''
    net = FCNet(
        input_dim=MNIST_img_ch * MNIST_img_size ** 2,
        output_dim=MNIST_num_classes,
        dropout_p=DROP_OUT_RATE
    )
    '''

    # net = resnetb.ResNet18(10)

    net = wideresnet.WideResNet(depth=4, widen_factor=2, num_classes=10, dropout_rate=0.2)

    init_params(net, verbose=True)
    if USING_GPU:
        net = net.cuda()
    
    print('=== start training from scratch ===\n')
    
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    lrs = []
    
    set_seed(0)
    optimizer = SGD(net.parameters(), lr=BASIC_LR, weight_decay=WEIGHT_DECAY, momentum=OP_MOMENTUM)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * ITERS, eta_min=MIN_LR)
    
    test_freq = 64
    set_seed(0)
    for epoch in range(EPOCHS):
        for local_iter, (inputs, targets) in enumerate(train_loader):
            global_iter = epoch * ITERS + local_iter
            if USING_GPU:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs.shape: (B, C, H, W)
            # targets.shape: (B,)
            # logits.shape: (B, num_classes)
            
            logits = net(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if global_iter % test_freq == 0 or epoch == EPOCHS - 1 and local_iter == ITERS - 1:
                test_acc, test_loss = test(test_loader, net)
                train_acc = 100 * logits.detach().argmax(dim=1).eq(targets).sum().item() / targets.shape[0]
                train_loss = loss.item()
                lr = scheduler.get_lr()[0]
                
                test_accs.append((global_iter, test_acc))
                test_losses.append((global_iter, test_loss))
                train_accs.append((global_iter, train_acc))
                train_losses.append((global_iter, train_loss))
                lrs.append((global_iter, lr))

                print(
                    f'{time_str()} ep[{epoch+1}/{EPOCHS}], it[{local_iter+1:-3d}/{ITERS}]:'
                    f' tr_acc: {train_acc:5.2f}%, tr_loss: {train_loss:.4f},'
                    f' te_acc: {test_acc:5.2f}%, te_loss: {test_loss:.4f},'
                    f' lr: {lr:6f}'
                )
    
    final_test_acc, _ = test(test_loader, net)
    print(f'\n=== final test acc: {final_test_acc:.2f} ===\n')
    
    plot_curves(train_accs, test_accs, train_losses, test_losses, lrs)


if __name__ == '__main__':
    main()
