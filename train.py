import math
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import argparse
import cv2
#from torchvision import models
#from torchinfo import summary
#torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--continue-training', dest='continue_training', 
                        required=True, choices=['yes', 'no'],
                        help='whether to continue training or not')
args = vars(parser.parse_args())
# Data parameters
data_folder = '../input_json_exp'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.__version__)
# Learning parameters
if args['continue_training'] == 'yes': # continue training or not
    checkpoint = 'checkpoint_ssd300.pth.tar'
else:
    print('Training from beginning')
    checkpoint = None
batch_size = 8  # batch size
# iterations = 40000  # number of iterations to train
iterations = 50000
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 500 # print training status every __ batches
#lr = 1e-3  # learning rate Experiment1
lr = 1e-3 # learning rate Experiment2
decay_lr_at = [5000,15000,25000,35000]  # decay learning rate after these many iterations Experiment1
#decay_lr_at = [4000, 5000,6000,9000,12000,16000]  # decay learning rate after these many iterations Experiment1
#decay_lr_at = [3000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]  # decay learning rate after these many iterations Experiment2
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay Experiment1
#weight_decay = 5e-1  # weight decay  Experiment2
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
min_loss=1000000000000000000.0
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint,map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        min_loss = checkpoint['min_loss']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_dataset = PascalVOCDataset(data_folder,split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)  # using `collate_fn()` here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              val_loader=val_loader, min_loss=min_loss)

        # Save checkpoint
        # save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, val_loader, min_loss):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    # global min_loss
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    val_losses = AverageMeter() #val loss
    start = time.time()
    start_data = time.time()
    # Batches
    print("Started training epoch"+str(epoch))
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
		

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        # print(images.size())
        boxes = [b.to(device) for b in boxes]
        # print(boxes)
        labels = [l.to(device) for l in labels]
        # print(labels)
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
        # print(predicted_locs.size())
        # print(predicted_scores.size())

        #print("predicted_locs:",predicted_locs)
        #print("predicted_scores:", torch.isnan(predicted_scores).any())
        
        
        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar


        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients, if necessary
        #if grad_clip is not None:
            # clip_gradient(optimizer, grad_clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

    print("Started validating for epoch"+str(epoch))   
    # free some memory since their histories may be stored
    # model.eval()
    # with torch.no_grad():
    for i, (images, boxes, labels, _) in enumerate(val_loader):
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # loss
        val_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        if not math.isinf(val_loss.item()):
            val_losses.update(val_loss.item(), images.size(0))
        else:
            print("----------------Boxes (inf)--------------------")
            print(boxes)
            print("----------------Labels (inf)--------------------")
            print(labels)
            print("Val_Loss "+str(val_loss.item()))
            print("Val_Losses.val "+str(val_losses.val))
            print("Val_Losses.sum "+str(val_losses.sum))
            batch_time.update(time.time() - start)
            # c=1
            # for i in images:
            #     cv2.imwrite(f"debug_images/test_image_{epoch}_{c}.jpg", i.numpy()[0])
            #     c+=1

        start = time.time()
    data_time.update(time.time() - start_data)
    # print status
    print("================================================================================================================================================")
    print(val_losses)
    print("================================================================================================================================================")
    print('Epoch: [{0}][{1}]\t'
            'Val Loss {val_loss.val:.2f} ({val_loss.avg:.2f})\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch,len(train_loader),val_loss=val_losses,
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses))
    
    # logging into train.txt
    with open(file='logs/train_logs.txt', mode='a+') as f:
        f.writelines('\nEpoch: [{0}][{1}]\t'
            'Val Loss {val_loss.val:.2f} ({val_loss.avg:.2f})\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, len(train_loader), val_loss=val_losses,
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses))

    # save checkpoint after each epoch
    # if val_losses.avg<min_loss:
    #     min_loss=val_losses.avg
    # save_checkpoint(epoch, model, optimizer, min_loss)
    # print('CheckPoint saved for minimum loss:: {val_loss.avg:.2f}\t'.format(val_loss=val_losses))
    # evaluate(val_loader,model)
    print ("val_losses avg = "+str(val_losses.avg))
    print("min_loss = "+str(min_loss))
    if val_losses.avg<min_loss:
        min_loss=val_losses.avg
        save_checkpoint(epoch, model, optimizer, min_loss)
        print('CheckPoint saved for minimum loss:: {val_loss.avg:.2f}\t'.format(val_loss=val_losses))
        evaluate(val_loader,model)
    print(val_losses.avg)########################################################################################################################################################################################
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

if __name__ == '__main__':
    main()
