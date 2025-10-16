from loadData import *
import  model  as  model_module

parser = argparse.ArgumentParser('Interface for  Training')
##### Optimizer - Scheduler
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--warmup', type=float, default=1.0, help='the number of epoch for warmup')
parser.add_argument('--lr_decay_epoch', type=str, default="4-8-16-24-26", help='the index of epoch where the lr decays to lr*0.5')
parser.add_argument('--num_prediction', type=int,default=6, help='the number of modality')
parser.add_argument('--cls_weight', type=float,default=0.1, help='the weight of classification loss')
parser.add_argument('--reg_weight', type=float,default=50.0, help='the weight of regression loss')

#### Speed Up
parser.add_argument('--num_of_gnn_layer', type=int, default=6, help='the number of  layer')
parser.add_argument('--hidden_dim', type=int, default=256, help='init hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='the dimension of attention head')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--num_worker', type=int, default=8, help='number of worker per dataloader')

#### Setting
parser.add_argument('--agent_drop', type=float, default='0.0', help='the ratio of randomly dropping agent')
parser.add_argument('--data_folder', type=str,default="hdgt_waymo_dev_tmp", help='training set')

parser.add_argument('--refine_num', type=int, default=5, help='temporally refine the trajectory')
parser.add_argument('--output_vel', type=str, default="True", help='output in form of velocity') 
parser.add_argument('--cumsum_vel', type=str, default="True", help='cumulate velocity for reg loss')


#### Initialize
parser.add_argument('--checkpoint', type=str, default="none", help='load checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='the index of start epoch (for resume training)')
parser.add_argument('--dev_mode', type=str, default="False", help='develop_mode')

parser.add_argument('--ddp_mode', type=str, default="False", help='False, True, multi_node')
parser.add_argument('--port', type=str, default="31243", help='DDP')

parser.add_argument('--amp', type=str, default="none", help='type of fp16')

#### Log
parser.add_argument('--val_every_train_step', type=int, default=-1, help='every number of training step to conduct one evaluation')
parser.add_argument('--name', type=str, default="hdgt_waymo_dev", help='the name of this setting')
args = parser.parse_args()
os.environ["DGLBACKEND"] = "pytorch"
####

def run_model(dataloader, num_sample, model, optimizer, scheduler, epoch, gpu, global_rank, gpu_count, is_train, args, val_dataloader=None, val_sample_num=None, snapshot_dir=None, scaler=None, amp_data_type=None, device=None):
    print("ZEVEL")
    with torch.set_grad_enabled(is_train):
       for batch_index, data in enumerate(dataloader, 0):
           data["is_train"] = is_train
           data["gpu"] = gpu
           data["device"] = device
           for tensor_name in data["cuda_tensor_lis"]:
               data[tensor_name] = data[tensor_name].to(device, non_blocking=True)
           optimizer.zero_grad()
           num_of_sample = len(data["pred_num_lis"])
           agent_reg_res, agent_cls_res, pred_indice_bool_type_lis = model(data)
           print("ZEVEL_1")
           
def main():
    global_rank=0
    gpu_count=1
    gpu=0
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp=args.amp != "none"
    amp_data_type=torch.float16 if use_amp else torch.float32
    snapshot_dir=None
    scaler=None
    train_dataloader, val_dataloader, train_sample_num, val_sample_num = obtain_dataset(0,1, 2, args)
    model = model_module.HDGT_model(input_dim=11, args=args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    step_per_epoch = train_sample_num // args.batch_size // gpu_count + 1
    epoch = args.n_epoch

    warmup = args.warmup
    if args.start_epoch > 1:
        warmup = 0.0

    scheduler = model_module.WarmupLinearSchedule(optimizer, step_per_epoch*warmup, step_per_epoch*epoch)
    for epoch in range(args.start_epoch, args.n_epoch+1):
        run_model(dataloader=train_dataloader, num_sample=train_sample_num, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, gpu=gpu, global_rank=global_rank, gpu_count=gpu_count, is_train=True, args=args, val_dataloader=val_dataloader, val_sample_num=val_sample_num, snapshot_dir=snapshot_dir, scaler=scaler, amp_data_type=amp_data_type, device=device)

    # agent_reg_res, agent_cls_res, pred_indice_bool_type_lis = model(data)
    print("ZEVEL")
if __name__ == '__main__':
    main()
