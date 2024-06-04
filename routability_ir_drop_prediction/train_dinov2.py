import os
import torch
import torch.optim as optim
from tqdm import tqdm
from models.dinov2 import Dinov2VisionTransformer

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from utils.metrics import build_metric, build_roc_prc_metric

def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def train():
    arg_dict = {
        'dataroot': './routability_features_training_data/congestion',
        'dataset_type': 'CongestionDataset',
        'aug_pipeline': ['Flip'],
        'test_mode': False,
        'batch_size': 64,
        'ann_file_train': "./files/train_N28.csv",
        'ann_file_test': "./files/test_N28.csv",
        'loss_type': 'MSELoss',
        'lr': 3e-4,
        'weight_decay': 5e-6,
        'max_iters': 3000, 
        'cpu': False,
        'save_path': "test_dinov2",
    }

    arg_dict['ann_file'] = arg_dict['ann_file_train']

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    model = Dinov2VisionTransformer.from_pretrained("xyzhang626/dinov2-base-patch16-256")

    if not arg_dict['cpu']:
        model = model.cuda()

    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False

    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    epoch_loss = 0
    print_freq = 100
    iter_num = 0
    save_freq = 1000

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                if arg_dict['cpu']:
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                # regular_lr = cosine_lr.get_regular_lr(iter_num)
                # cosine_lr._set_lr(optimizer, regular_lr)

                # print("target shape: ", target.shape)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
        epoch_loss = 0

    val_loss = validate(model, arg_dict)
    print("===> Validate Loss: {:.4f}".format(val_loss))

def validate(model, arg_dict):
    model.eval()

    arg_dict['dataset_type'] = 'CongestionDataset'
    arg_dict['ann_file'] = './files/test_N28.csv'
    arg_dict['test_mode'] = True
    arg_dict['eval_metric'] = ['NRMS', 'SSIM']

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    if not arg_dict['cpu']:
        model = model.cuda()

    # Build metrics
    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0 for k in arg_dict['eval_metric']}

    for feature, label, _ in dataset:

        if arg_dict['cpu']:
            input, target = feature, label
        else:
            input, target = feature.cuda(), label.cuda()

        prediction = model(input)
        for metric, metric_func in metrics.items():
            if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    return avg_metrics['SSIM'] / len(dataset) 

if __name__ == "__main__":
    train()