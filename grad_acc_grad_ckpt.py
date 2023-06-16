# @title train test function
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = torch.cuda.amp.GradScaler()
# https://github.com/prigoyal/pytorch_memonger/blob/master/models/optimized/resnet_new.py
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

trs=TrainTransform() # for image augmentation during train time
# train function with automatic mixed precision
def strain(dataloader, model, loss_fn, optimizer, scheduler=None, verbose=True):
    size = len(dataloader.dataset)
    model.train()
    loss_list = []
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(): # automatic mixed percision
            x = trs(x) # image augmentation during train time to use gpu
            pred = model(x) # default
            print("train",pred[0])
            # modules = [module for k, module in model._modules.items()]
            # pred = checkpoint_sequential(functions=modules, segments=1, input=x) # gradient checkpointing for resnet and inception only
            # # # pred = checkpoint_sequential(functions=model.mods, segments=1, input=x)
            print((pred.argmax(1)==y).sum().item(), pred.argmax(1).tolist(), y.tolist())
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        if ((batch + 1) % 4 == 0) or (batch + 1 == len(dataloader)): # gradient accumulation
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                print("lr: ", optimizer.param_groups[0]["lr"])

        loss_list.append(loss.item())
        # if (batch) % (size//(10* len(x))) == 0:
        # loss, current = loss.item(), batch * len(x)
        loss, current = loss.item()/len(y), batch * len(x)
        if verbose: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_list
