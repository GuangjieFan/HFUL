#进行完整版训练恢复
import sys

sys.path.append('../../../')
sys.path.append('')
from Network.utils_network import init_nets
from Dataset.utils_dataset import init_logs, get_dataloader, generate_public_data_idxs
from Idea.utils_idea import update_model_via_private_data_with_two_model, evaluate_network, mkdirs, update_model_via_private_data_with_two_model_un,update_model_via_private_data_with_two_model_nomal,update_model_via_private_data_with_two_model_recover
from Idea.params import args_parser
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from numpy import *
import numpy as np
import torch
import copy
import os
import pandas as pd
from Idea.Ours.lora_fine import fine_tune_with_lora
from Idea.Ours.mask_matrix import ComplexMatrixMasker
args = args_parser()

'''
Global Parameters
'''
Method_Name = 'Ours'
Ablation_Name = 'FCCL'

Temperature = 1
Scenario = args.Scenario
Seed = args.Seed
N_Participants = args.N_Participants
CommunicationEpoch = args.CommunicationEpoch + 20
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize
Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Idea_Ours_Dir = args.Project_Dir + 'Idea/Ours/'
Private_Net_Name_List = args.Private_Net_Name_List
Pariticpant_Params = {
    'loss_funnction': 'KLDivLoss',
    'optimizer_name': 'Adam',
    'learning_rate': 0.001
}
'''
Scenario for large domain gap
'''
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Data_Total_Len_List = args.Private_Data_Total_Len_List
Private_Data_Len_List = args.Private_Data_Len_List
Private_Training_Epoch = args.Private_Training_Epoch
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)
'''
Public data parameters
'''
Public_Dataset_Name = args.Public_Dataset_Name
Public_Dataset_Length = args.Public_Dataset_Length
Public_Dataset_Dir = Dataset_Dir + Public_Dataset_Name
Public_Training_Epoch = args.Public_Training_Epoch


def off_diagonal(x):  # 非对角线矩阵处理
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    logger = init_logs(sub_name=Ablation_Name)
    logger.info('Method Name : ' + Method_Name + ' Ablation Name : ' + Ablation_Name)
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = args.device_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(device)
    logger.info("Initialize Participants' Data idxs and Model")
    # For Digits scenario
    private_dataset_idxs_dict = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(Private_Data_Total_Len_List[index])
        idxes = idxes[0:Private_Data_Len_List[index]]
        private_dataset_idxs_dict[Private_Dataset_Name_List[index]] = idxes  # 处理数据集 分割
    logger.info(private_dataset_idxs_dict)

    # net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)#初始化网络
    net_list = init_nets(n_parties=N_Participants, nets_name_list=Private_Net_Name_List)
    logger.info("Load Participants' Models")
    for i in range(N_Participants):
        network = net_list[i]  # 逐一选取网络
        # network = nn.DataParallel(network, device_ids=device_ids).to(device)
        network = network.to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(
            i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))  # 预训练模型自己的私人数据

    # frozen_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)#加载初始化网络，可以理解为本地数据集训练好的数据
    frozen_net_list = init_nets(n_parties=N_Participants, nets_name_list=Private_Net_Name_List)  # frozen冻结网络
    logger.info("Load Frozen Participants' Models")
    for i in range(N_Participants):
        network = frozen_net_list[i]
        # network = nn.DataParallel(network, device_ids=device_ids).to(device)
        network = network.to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + '/Network/Model_Storage/' + netname + '_' + str(
            i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))  # 可以理解为获取预训练模型当做forzeon模型

    progressive_net_list = init_nets(n_parties=N_Participants, nets_name_list=Private_Net_Name_List)
    # progressive_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)#全局逐层更新的模型
    logger.info("Load Progressive Participants' Models")
    for i in range(N_Participants):
        network = progressive_net_list[i]
        # network = nn.DataParallel(network, device_ids=device_ids).to(device)
        network = network.to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(
            i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))  # 这个叫过程模型加载初始化模型

    logger.info("Initialize Public Data Parameters")
    print(Scenario + Public_Dataset_Name)  # 利用cifra10当做unlable data
    public_data_indexs = generate_public_data_idxs(dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
                                                   size=Public_Dataset_Length)

    public_train_dl, _, _, _ = get_dataloader(dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
                                              train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                              dataidxs=public_data_indexs)  # 将数据tensor张量化
    logger.info('Initialize Private Data Loader')
    private_train_data_loader_list = []
    private_test_data_loader_list = []
    for participant_index in range(N_Participants):
        private_dataset_name = Private_Dataset_Name_List[participant_index]
        private_dataidx = private_dataset_idxs_dict[private_dataset_name]
        private_dataset_dir = Dataset_Dir + private_dataset_name
        train_dl_local, test_dl_local, _, _ = get_dataloader(dataset=private_dataset_name,
                                                             datadir=private_dataset_dir,
                                                             train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                             dataidxs=private_dataidx)
        private_train_data_loader_list.append(train_dl_local)
        private_test_data_loader_list.append(test_dl_local)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    for epoch_index in range(CommunicationEpoch):

        logger.info("The " + str(epoch_index) + " th Communication Epoch")
        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname = Private_Net_Name_List[participant_index]  # 获得预训练模型
            print('netname:',netname)
            private_dataset_name = Private_Dataset_Name_List[participant_index]
            print('private_dataset_name',private_dataset_name)
            private_dataset_dir = Dataset_Dir + private_dataset_name
            print(netname + '_' + private_dataset_name + '_' + private_dataset_dir)  # 打印预训练模型的
            _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                              train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None)
            network = net_list[participant_index]
            # network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network = network.to(device)
            acc_epoch_list.append(
                evaluate_network(network=network, dataloader=test_dl, logger=logger))  # 每个模型的成功率 本地数据集上
        acc_list.append(acc_epoch_list)

        a = datetime.now()
        for _ in range(Public_Training_Epoch):  # 公开训练轮测
            for batch_idx, (images, _) in enumerate(public_train_dl):  # 获得公开数据集
                linear_output_list = []
                linear_output_target_list = []  # Save other participants' linear output#先开始处理logit输出了
                linear_output_progressive_list = []  # Save itself progressive model's linear output
                col_loss_batch_list = []
                '''
                Calculate Linear Output
                '''
                for participant_index in range(N_Participants):
                    network = net_list[participant_index]  # 私人的logit 可以理解本地模型的学习
                    # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network = network.to(device)
                    network.train()
                    images = images.to(device)
                    linear_output = network(x=images)
                    linear_output_target_list.append(linear_output.clone().detach())
                    linear_output_list.append(linear_output)
                '''
                Calculate Progressive Linear Output
                '''
                for progressive_participant_index in range(N_Participants):
                    progressive_network = progressive_net_list[progressive_participant_index]
                    # progressive_network = nn.DataParallel(progressive_network,device_ids=device_ids).to(device)
                    progressive_network = progressive_network.to(device)
                    progressive_network.eval()
                    with torch.no_grad():
                        images = images.to(device)
                        progressive_linear_output = progressive_network(images)
                        linear_output_progressive_list.append(progressive_linear_output)
                '''
                Update Participants' Models via Col Loss#开始更新 这是在服务器端更新的
                '''
                for participant_index in range(N_Participants):
                    '''
                    Calculate the Loss with others
                    '''
                    network = net_list[participant_index]
                    # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network = network.to(device)
                    network.train()
                    optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                    optimizer.zero_grad()
                    linear_output_target_avg_list = []
                    for i in range(N_Participants):
                        if i != participant_index:
                            linear_output_target_avg_list.append(linear_output_target_list[i])
                        if i == participant_index:
                            linear_output_target_avg_list.append(linear_output_progressive_list[i])

                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)  # 计算平均损失
                    linear_output = linear_output_list[participant_index]
                    z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)  # 归一化处理
                    z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(
                        0)) / linear_output_target_avg.std(0)  # 归一化处理
                    # empirical cross-correlation matrix
                    # print("第 participant_index 个客户端的c矩阵:",participant_index)
                    c = z_1_bn.T @ z_2_bn  # 矩阵乘法
                    # print("1*c:", c)

                    # z_1_bn_df = pd.DataFrame(z_1_bn.detach().numpy())
                    # z_2_bn_df = pd.DataFrame(z_2_bn.detach().numpy())
                    #c_df = pd.DataFrame(c.detach().numpy())
                    #file_name = f'dataxiaorongwangque3client_{participant_index}.xlsx'
                    #writer = pd.ExcelWriter(file_name, engine='openpyxl')
                    # z_1_bn_df.to_excel(writer, sheet_name='z_1', index=False)
                    # z_2_bn_df.to_excel(writer, sheet_name='z_2', index=False)
                    #c_df.to_excel(writer, sheet_name='c', index=False)
                    # 保存 Excel 文件
                    # writer.save()
                    # 关闭 writer 对
                    #writer.close()
                    # sum the cross-correlation matrix between all gpus
                    c.div_(len(images))
                    if 39 < epoch_index < 60:
                        if participant_index == 0:
                            transformer = ComplexMatrixMasker(c, 0.55, device)
                            c = transformer.add_matrix()

                    # if participant_index == N_Participants - 1:
                    #     c = -1 * c
                    # z_1_bn_df = pd.DataFrame(z_1_bn.detach().cpu().numpy())
                    # z_2_bn_df = pd.DataFrame(z_2_bn.detach().cpu().numpy())
                    c_df = pd.DataFrame(c.detach().cpu().numpy())
                    file_name = f'data_{participant_index}_unohclient2_newest.xlsx'
                    writer = pd.ExcelWriter(file_name, engine='openpyxl')
                    # z_1_bn_df.to_excel(writer, sheet_name='z_1', index=False)
                    # z_2_bn_df.to_excel(writer, sheet_name='z_2', index=False)
                    c_df.to_excel(writer, sheet_name='c', index=False)
                    writer.close()
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                    col_loss = on_diag + 0.0051 * off_diag
                    col_loss_batch_list.append(col_loss.item())
                    col_loss.backward()
                    optimizer.step()
                col_loss_list.append(col_loss_batch_list)  # 求出的损失值

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]  # 初始化网络
            # network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network = network.to(device)
            network.train()

            frozen_network = frozen_net_list[participant_index]  # 本地训练模型
            # frozen_network = nn.DataParallel(frozen_network,device_ids=device_ids).to(device)
            frozen_network = frozen_network.to(device)
            frozen_network.eval()

            progressive_network = progressive_net_list[participant_index]  # 更新过程的模型
            # progressive_network = nn.DataParallel(progressive_network,device_ids=device_ids).to(device)
            progressive_network = progressive_network.to(device)
            progressive_network.eval()

            private_dataset_name = Private_Dataset_Name_List[participant_index]  # 隐私数据
            private_dataidx = private_dataset_idxs_dict[private_dataset_name]
            private_dataset_dir = Dataset_Dir + private_dataset_name
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=private_dataset_name,
                                                                  datadir=private_dataset_dir,
                                                                  train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                                  dataidxs=private_dataidx)

            private_epoch = max(int(Public_Dataset_Length / len(train_ds_local)), 1)
            private_epoch = Private_Training_Epoch[participant_index]
            if epoch_index <40:

                network, private_loss_batch_list = update_model_via_private_data_with_two_model_nomal(network=network,
                                                                                                frozen_network=frozen_network,
                                                                                                progressive_network=progressive_network,
                                                                                                temperature=Temperature,
                                                                                                private_epoch=private_epoch,
                                                                                                private_dataloader=train_dl_local,
                                                                                                loss_function=
                                                                                                Pariticpant_Params[
                                                                                                    'loss_funnction'],
                                                                                                optimizer_method=
                                                                                                Pariticpant_Params[
                                                                                                    'optimizer_name'],
                                                                                                learing_rate=
                                                                                                Pariticpant_Params[
                                                                                                    'learning_rate'],
                                                                                                logger=logger)


            if epoch_index>=40 :
                if participant_index != 1:
                    network, private_loss_batch_list = update_model_via_private_data_with_two_model(network=network,
                                                                                            frozen_network=frozen_network,
                                                                                            progressive_network=progressive_network,
                                                                                            temperature=Temperature,
                                                                                            private_epoch=private_epoch,
                                                                                            private_dataloader=train_dl_local,
                                                                                            loss_function=
                                                                                            Pariticpant_Params[
                                                                                                'loss_funnction'],
                                                                                            optimizer_method=
                                                                                            Pariticpant_Params[
                                                                                                'optimizer_name'],
                                                                                            learing_rate=
                                                                                            Pariticpant_Params[
                                                                                                'learning_rate'],
                                                                                            logger=logger)

                else:
                    #network = fine_tune_with_lora(model=network, train_dataloader=train_dl_local, epochs=5)#digit模式先微调
                    network, private_loss_batch_list = update_model_via_private_data_with_two_model_un(network=network,
                                                                                                frozen_network=frozen_network,
                                                                                                progressive_network=progressive_network,
                                                                                                temperature=Temperature,
                                                                                                private_epoch=private_epoch,
                                                                                                private_dataloader=train_dl_local,
                                                                                                loss_function=
                                                                                                Pariticpant_Params[
                                                                                                    'loss_funnction'],
                                                                                                optimizer_method=
                                                                                                Pariticpant_Params[
                                                                                                    'optimizer_name'],
                                                                                                learing_rate=
                                                                                                Pariticpant_Params[
                                                                                                    'learning_rate'],
                                                                                                logger=logger)
                    network = fine_tune_with_lora(model=network, train_dataloader=train_dl_local, epochs=5,rank=4)#oh模式先蒸馏



            # if epoch_index >49:
            #     print("   ")
            #     print("   ")
            #     N_Participants = 3
            #     print('N_Participants:',N_Participants)
            #     network, private_loss_batch_list = update_model_via_private_data_with_two_model_recover(network=network,
            #                                                                                 frozen_network=frozen_network,
            #                                                                                 progressive_network=progressive_network,
            #                                                                                 temperature=Temperature,
            #                                                                                 private_epoch=private_epoch,
            #                                                                                 private_dataloader=train_dl_local,
            #                                                                                 loss_function=
            #                                                                                 Pariticpant_Params[
            #                                                                                     'loss_funnction'],
            #                                                                                 optimizer_method=
            #                                                                                 Pariticpant_Params[
            #                                                                                     'optimizer_name'],
            #                                                                                 learing_rate=
            #                                                                                 Pariticpant_Params[
            #                                                                                     'learning_rate'],
            #                                                                                 logger=logger)
            mean_private_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss_batch)
        local_loss_list.append(local_loss_batch_list)
        b = datetime.now()
        temp = b - a
        # with open("time_difference_oh4.txt", "a") as file:  # 使用 "a" 模式可以追加写入
        #     file.write(f"Epoch: {b}, Time difference: {b}\n")
        print(temp)

        '''
        用于迭代 Progressive 模型
        '''
        for j in range(N_Participants):
            progressive_net_list[j] = copy.deepcopy(net_list[j])

        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                private_dataset_dir = Dataset_Dir + private_dataset_name
                print(netname + '_' + private_dataset_name + '_' + private_dataset_dir)
                _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None)

                network = net_list[participant_index]
                # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network = network.to(device)

                acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
            acc_list.append(acc_epoch_list)
            print(acc_list)

        # if epoch_index % 5 == 3 or epoch_index == CommunicationEpoch - 1:
        #     mkdirs(Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario)
        #     mkdirs(Idea_Ours_Dir + '/Model_Storage_recover/' + Scenario)
        #     mkdirs(Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario + '/' + Ablation_Name)
        #     mkdirs(Idea_Ours_Dir + '/Model_Storage_recover/' + Scenario + '/' + Ablation_Name)
        #     mkdirs(
        #         Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)
        #     mkdirs(Idea_Ours_Dir + '/Model_Storage_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)
        #
        #     logger.info('Save Loss')
        #     col_loss_array = np.array(col_loss_list)
        #     np.save(
        #         Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
        #         + '/collaborative_loss.npy', col_loss_array)
        #     local_loss_array = np.array(local_loss_list)
        #     np.save(
        #         Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
        #         + '/local_loss.npy', local_loss_array)
        #     logger.info('Save Acc')
        #     acc_array = np.array(acc_list)
        #     np.save(
        #         Idea_Ours_Dir + '/Performance_Analysis_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
        #         + '/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                network = net_list[participant_index]
                # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network = network.to(device)
                # torch.save(network.state_dict(),
                #            Idea_Ours_Dir + '/Model_Storage_recover/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                #            + '/' + netname + '_' + str(participant_index) + '_' + private_dataset_name + '.ckpt')
                full_path = os.path.join(Idea_Ours_Dir, 'Model_Storage_client_222', Scenario, Ablation_Name, Public_Dataset_Name)

                # 创建目录（如果目录不存在则创建）
                os.makedirs(full_path, exist_ok=True)

                # 保存模型
                torch.save(network.state_dict(),
                           os.path.join(full_path, f'{netname}_{participant_index}_{private_dataset_name}.ckpt'))
