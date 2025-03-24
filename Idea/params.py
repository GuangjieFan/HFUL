import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    '''
    Default Setting
    '''
    parser.add_argument('--Dataset_Dir',type=str,default=' ')
    parser.add_argument('--Project_Dir',type=str,default=' ')
    parser.add_argument('--Original_Path',type=str,default=' ')
    parser.add_argument('--CommunicationEpoch', type=int, default=40)
    parser.add_argument('--CommunicationEpoch2un', type=int, default=15)
    parser.add_argument('--Seed',type=int,default=42)
    parser.add_argument('--device_ids',type=list,default=[0])

    '''
    General Setting
    '''
    parser.add_argument('--N_Participants',type=int,default=4)
    '''
    Scenario for domain gap and corresponding public data parameteres
    '''
    Scenario='Digits'
    parser.add_argument('--Scenario',type=str,default=Scenario)
    parser.add_argument('--Public_Dataset_Name', type=str, default='cifar_10')

    if Scenario =='Digits':

        parser.add_argument('--Private_Net_Name_List', type=list,
                            default=['ResNet10', 'ResNet12', 'Efficientnet', 'Mobilenetv2'])
        parser.add_argument('--Private_Dataset_Name_List', type=list, default=['mnist', 'usps', 'svhn','FashionMNIST'])
        parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[60000, 7291, 73257,60000])
        parser.add_argument('--Private_Data_Len_List', type=list, default=[150, 80, 5000,150])
        parser.add_argument('--Private_Training_Epoch',type=list,default=[40,35,3,40])
        parser.add_argument('--Private_Dataset_Classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parser.add_argument("--Private_Net_Feature_Dim_List", type=list, default=[512, 512, 320, 1280])
        #现在进行 剔除第一个客户端的重训练
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet12', 'Efficientnet', 'Mobilenetv2'])
        # parser.add_argument('--Private_Dataset_Name_List', type=list, default=['usps', 'svhn', 'FashionMNIST'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[7291, 73257, 60000])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[80, 5000, 150])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[35, 3, 40])
        #剔除第二个
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10',  'Efficientnet', 'Mobilenetv2'])
        # parser.add_argument('--Private_Dataset_Name_List', type=list, default=['mnist',  'svhn', 'FashionMNIST'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[60000,  73257, 60000])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[150,  5000, 150])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[40,  3, 40])
        # parser.add_argument('--Private_Dataset_Classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # parser.add_argument("--Private_Net_Feature_Dim_List", type=list, default=[512,  320, 1280])
        #剔除第三个
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10', 'ResNet12', 'Mobilenetv2'])
        # parser.add_argument('--Private_Dataset_Name_List', type=list, default=['mnist', 'usps', 'FashionMNIST'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[60000, 7291,  60000])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[150, 80, 150])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[40, 35,  40])
        # parser.add_argument('--Private_Dataset_Classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # parser.add_argument("--Private_Net_Feature_Dim_List", type=list, default=[512, 512,  1280])
        #剔除第四个
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10', 'ResNet12', 'Efficientnet', ])
        # parser.add_argument('--Private_Dataset_Name_List', type=list, default=['mnist', 'usps', 'svhn'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[60000, 7291, 73257])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[150, 80, 5000])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[40, 35, 3])
        parser.add_argument('--TrainBatchSize',type=int,default=64)
        parser.add_argument('--Local_TrainBatchSize',type=int,default=64)
        parser.add_argument('--TestBatchSize',type=int,default=64)
        parser.add_argument('--Public_Training_Epoch',type=int,default=1)
        parser.add_argument('--Public_Dataset_Length',type=int,default=5000)

    if Scenario =='OfficeHome':

        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet18', 'ResNet18', 'ResNet18', 'ResNet18'])
        parser.add_argument('--Private_Net_Name_List', type=list,
                            default=['ResNet10', 'ResNet12', 'Efficientnet', 'Mobilenetv2'])
        parser.add_argument('--Private_Dataset_Name_List',type=list,default=['Art', 'Clipart','Product','Real World'])
        parser.add_argument('--Private_Data_Total_Len_List',type=list,default=[1700,3050,3100,3050])
        parser.add_argument('--Private_Data_Len_List',type=list,default=[1400, 2000,2500,2000])
        parser.add_argument('--Private_Training_Epoch',type=list,default=[10,6,6,6])
        #客户端组1的剔除重训练配置
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=[ 'ResNet12', 'Efficientnet', 'Mobilenetv2'])
        # parser.add_argument('--Private_Dataset_Name_List', type=list,
        #                     default=['Clipart', 'Product', 'Real World'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[ 3050, 3100, 3050])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[ 2000, 2500, 2000])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[ 6, 6, 6])
        #客户端组的2的剔除重训练配置
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10',  'Efficientnet', 'Mobilenetv2'])
        # #parser.add_argument('--Private_Dataset_Name_List',type=list,default=['Art', 'Product','Real World'])
        # parser.add_argument('--Private_Data_Total_Len_List',type=list,default=[1700,3100,3050])
        # parser.add_argument('--Private_Data_Len_List',type=list,default=[1400, 2500,2000])
        # parser.add_argument('--Private_Training_Epoch',type=list,default=[10,6,6])
        #剔除3客户端的重训练
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10', 'ResNet12', 'Mobilenetv2'])
        # parser.add_argument('--Private_Dataset_Name_List', type=list,
        #                     default=['Art', 'Clipart',  'Real World'])
        # parser.add_argument('--Private_Data_Total_Len_List', type=list, default=[1700, 3050,  3050])
        # parser.add_argument('--Private_Data_Len_List', type=list, default=[1400, 2000,  2000])
        # parser.add_argument('--Private_Training_Epoch', type=list, default=[10, 6,  6])
        #剔除4号客户端
        # parser.add_argument('--Private_Net_Name_List', type=list,
        #                     default=['ResNet10', 'ResNet12', 'Efficientnet'])
        # parser.add_argument('--Private_Dataset_Name_List',type=list,default=['Art', 'Clipart','Product'])
        # parser.add_argument('--Private_Data_Total_Len_List',type=list,default=[1700,3050,3100])
        # parser.add_argument('--Private_Data_Len_List',type=list,default=[1400, 2000,2500])
        # parser.add_argument('--Private_Training_Epoch',type=list,default=[10,6,6])
        parser.add_argument('--Private_Dataset_Classes',type=list,default=['Alarm', 'Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
        'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles','Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk Lamp',
        'Drill', 'Eraser', 'Exit Sign', 'Fan','File Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet',
        'Kettle', 'Keyboard','Knives', 'Lamp Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
        'Paper Clip', 'Pen', 'Pencil', 'Postit Notes', 'Printer', 'Push Pin', 'Radio', 'Refrigerator', 'ruler','Scissors', 'Screwdriver',
        'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone','Toothbrush', 'Toys', 'Trash Can', 'TV', 'Webcam'])
        parser.add_argument('--TrainBatchSize',type=int,default=64)
        parser.add_argument('--TestBatchSize',type=int,default=64)
        parser.add_argument('--Public_Training_Epoch',type=int,default=1)
        parser.add_argument('--Public_Dataset_Length',type=int,default=5000)


    args = parser.parse_args()
    return args