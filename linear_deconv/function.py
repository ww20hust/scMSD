import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import classification_report 
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import random
def plot_list(data):

    # 创建一个空画布
    fig, ax = plt.subplots()

    # 绘制火柴图
    for i, value in enumerate(data):
    # 确定火柴的位置和长度
        x_position = i
        y_position = 0
        matchstick_length = value
        # 绘制火柴
        ax.plot([x_position, x_position], [y_position, matchstick_length], color='black')

    # 隐藏火柴头
    ax.set_yticklabels([])

    # 设置图的标题和横轴标签
    plt.title('figure')
    plt.xlabel('x')

    # 显示火柴图
    plt.show()

def train_model(data_dict):
    model = xgboost.XGBClassifier(n_estimators=600,subsample=0.8)
    model.fit(data_dict['X_train'],data_dict['y_train'])

    y_pred = model.predict(data_dict['X_test'])
    prob = model.predict_proba(data_dict['X_test'])
    report_dict = classification_report(data_dict['y_test'], y_pred, output_dict=True)
    
    return model, y_pred, prob,report_dict

def split_data(data_value,label_value,test_size,random_state):
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(data_value, label_value, 
                                                        test_size=test_size,
                                                        random_state=random_state)
    print(y_train)
    y_train = le.fit_transform(y_train)
    print(y_train)
    y_test = le.fit_transform(y_test)

    data_dict = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}
    return data_dict

def shap_index_out(top_num_shap,shap_values):
    shap_index = []
    for i in range(shap_values.shape[0]):
        contrib_xgb_best = shap_values[i]
        shap_ranked_index = np.argsort(abs(contrib_xgb_best).mean(0))[::-1]
        shap_index.append(shap_ranked_index)
    shap_index = np.array(shap_index)
    shap_index = shap_index[:,:top_num_shap]
    shap_index = shap_index.reshape(-1)
    shap_index = list(set(list(shap_index)))
    return shap_index


def cell_type_ref_train(data,types,num_for_mean,eachtype_num,shap_index):
    data_matrix_list_train = []
    mean = []
    for i in range(len(types)):
        array_temp = np.array(data[data['type']==types[i]].drop(['type'],axis  = 1))[:,shap_index]
        print(array_temp.shape,types[i])
        # 计算每一行的和
        row_sums = np.sum(array_temp, axis=1)
        # 获取按照行和排序的索引
        sorted_indices = np.argsort(row_sums)
        # 根据排序索引对原始数组进行排序
        array_temp = array_temp[sorted_indices]
        array_temp = array_temp[:eachtype_num]
        np.random.shuffle(array_temp)
        np.random.shuffle(array_temp)
        data_matrix_list_train.append(array_temp[num_for_mean:eachtype_num])
        mean.append(np.mean(array_temp[:num_for_mean], axis = 0))
        print(data_matrix_list_train[i].shape,types[i])
    cell_type_ref  = np.array(mean)
    return cell_type_ref,data_matrix_list_train

def MixDataset(data_matrix_list,noise_percentage,ratio):
    cell_type_num = len(data_matrix_list)
    data_matrix_list_noise = [None]*cell_type_num 
    for i in range(cell_type_num):
        noise = np.random.uniform(-0.1, 0.1, data_matrix_list[i].shape)
        data_matrix_list_noise[i] = data_matrix_list[i] + noise_percentage*noise*data_matrix_list[i]
    mix_sample_spec = np.empty((cell_type_num,data_matrix_list_noise[0].shape[-1]))
    for i in range(cell_type_num):
        if ratio[i] == 0:
            mix_sample_spec[i] = np.zeros(data_matrix_list_noise[0].shape[-1])
        else:
            random_indices = np.random.choice(data_matrix_list_noise[i].shape[0], ratio[i], replace=False)
            random_samples = data_matrix_list_noise[i][random_indices]
            mix_sample_spec[i] = np.sum(random_samples, axis=0)
    mix_sample_spec = np.sum(mix_sample_spec, axis=0)
    mix_sample_spec = mix_sample_spec + np.random.uniform(-0.1, 0.1, mix_sample_spec.shape)*noise_percentage*mix_sample_spec
    return mix_sample_spec


def norm_method(method,pretensor):
    if method == 'Min_Max_Scaling':
        min_vals, _ = torch.min(pretensor, dim=1, keepdim=True)
        max_vals, _ = torch.max(pretensor, dim=1, keepdim=True)
        posttensor = (pretensor - min_vals) / (max_vals - min_vals)
        # return posttensor
    elif method == 'z_score':
        mean = pretensor.mean()
        std = pretensor.std()
        # 对Tensor进行Z-Score归一化
        posttensor = (pretensor - mean) / std
    
    elif method == 'l2_norm':
        l2_norm = torch.norm(pretensor, p=2,dim = 1)
        # 进行L2范数归一化
        posttensor = pretensor / l2_norm[:,None]
    
    elif method == 'Max_Scaling':
        # min_vals, _ = torch.min(pretensor, dim=1, keepdim=True)
        max_vals, _ = torch.max(pretensor, dim=1, keepdim=True)
        posttensor = (pretensor) / (max_vals)
    return posttensor

def norm_method(method,pretensor):
    if method == 'Min_Max_Scaling':
        min_vals, _ = torch.min(pretensor, dim=1, keepdim=True)
        max_vals, _ = torch.max(pretensor, dim=1, keepdim=True)
        posttensor = (pretensor - min_vals) / (max_vals - min_vals)
        # return posttensor
    elif method == 'z_score':
        mean = pretensor.mean()
        std = pretensor.std()
        # 对Tensor进行Z-Score归一化
        posttensor = (pretensor - mean) / std
    
    elif method == 'l2_norm':
        l2_norm = torch.norm(pretensor, p=2,dim = 1)
        # 进行L2范数归一化
        posttensor = pretensor / l2_norm[:,None]
    
    elif method == 'Max_Scaling':
        # min_vals, _ = torch.min(pretensor, dim=1, keepdim=True)
        max_vals, _ = torch.max(pretensor, dim=1, keepdim=True)
        posttensor = (pretensor) / (max_vals)
    return posttensor

class MyModel(nn.Module):
    def __init__(self,cell_type_num,init_choice):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.rand(1, cell_type_num))  
        if init_choice == True:
            init.constant_(self.weight, 1/cell_type_num)
    def forward(self, x,norm_method_name):
        ratio_tensor = self.weight/torch.sum(self.weight)
        output = torch.mm(ratio_tensor,x)  # 使用可学习的矩阵进行矩阵乘法运算
        output = norm_method(norm_method_name,output)
        return output,ratio_tensor    
# class MyModel(nn.Module):
#     def __init__(self,cell_type_num,init_choice):
#         super(MyModel, self).__init__()
#         self.weight = nn.Parameter(torch.rand(1, cell_type_num))  
#         if init_choice == True:
#             init.constant_(self.weight, 1/cell_type_num)
#     def forward(self, x,norm_method_name):
#         temptensor = F.softmax(self.weight, dim=1) 
#         output = torch.mm(temptensor,x)  # 使用可学习的矩阵进行矩阵乘法运算
#         output = norm_method(norm_method_name,output)
#         return output    
    
# 定义一个函数来裁剪权重
def clip_weights(model, min_clip=0.0, max_clip=1.0):
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.clamp(param.data, min_clip, max_clip)




def linear_deconvolution(cell_type_num,target,cell_ref,init_choice,clip_choice,lossfunc_name,norm_method_name):
    model = []
    model = MyModel(cell_type_num,init_choice)
    if lossfunc_name == 'mae':
        lossfunc = nn.L1Loss()
    elif lossfunc_name == 'mse':
        lossfunc = nn.MSELoss()
    elif lossfunc_name == 'smooth':
        lossfunc = nn.SmoothL1Loss()
    # model.to(device)
    LEARNING_RATE = 0.001
    optimizer = []
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_step = 1000
    # cell_ref = cell_ref.to(device)
    # target = target.to(device)
    for i in range(train_step):
        output,ratio_tensor= model(cell_ref,norm_method_name)
        bias = torch.ones_like(ratio_tensor) * 0.5
        loss1 = lossfunc(output,target)
        loss2 = lossfunc(ratio_tensor,bias)
        # loss = lossmse(output,target)
        loss = loss1
        # loss = losssmooth(output,target)
        # print("loss",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if clip_choice == True:
            clip_weights(model, min_clip=0, max_clip=10.0)
    return ratio_tensor
    # return F.softmax(model.weight, dim=1)

# import random
def ratio_num_generate(totalcellnum,celltypechoice,totalcelltype):
    ratio = [0]*totalcelltype
    norm_ratio = [0]*len(celltypechoice)
    random_integers = []
    # 生成三个随机整数
    for _ in range(len(celltypechoice)-1):
        random_integer = random.randint(0, totalcellnum - sum(random_integers))
        random_integers.append(random_integer)

    # 计算第四个整数以确保和为N
    last_integer = totalcellnum - sum(random_integers)
    random_integers.append(last_integer)

    np.random.shuffle(random_integers)

    for i in range(len(random_integers)):
        ratio[celltypechoice[i]] = random_integers[i]
        norm_ratio[i] = random_integers[i]/sum(random_integers)

    
    return ratio,norm_ratio

def caculate_metrics(list1,list2):
    x = np.array(list1)
    y = np.array(list2)
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    # 计算标准差
    std_x = np.std(x)
    std_y = np.std(y)

    # 计算皮尔逊相关系数
    pearson_corr = np.corrcoef(x, y)[0, 1]

    # 计算 Lin's CCC
    ccc = (2 * pearson_corr * std_x * std_y) / (std_x**2 + std_y**2 + (mean_x - mean_y)**2)

    coefficients = np.polyfit(x, y, 1)
    m = coefficients[0]
    b = coefficients[1]

    predicted_y = [m * xx+b for xx in x]
    # 计算残差
    residuals = [a - b for a, b in zip(y,predicted_y)]
    # 计算残差的标准差
    residual_std = np.std(residuals)

    return pearson_corr,ccc,residual_std,m,b,coefficients,rmse

def withoutnan(output,target):
    i = 0
    output_without_nan = []
    target_without_nan = []
    for x in output:
        # if not np.isnan(x) and x>=0 and x<=1:
        if not np.isnan(x):
            output_without_nan.append(output[i])
            target_without_nan.append(target[i])
        i = i+1
    return output_without_nan,target_without_nan

def ListDimConvert(two_dim_list):
    one_dim_list = [item for sublist in two_dim_list for item in sublist]
    return one_dim_list

def plot_point_line(target_list,output_list,savename,loc,save_choice):
    plt.figure(figsize=(9,6))
    x_stick = target_list
    y_stick = output_list
    plt.scatter(x_stick, y_stick, c="purple", cmap='viridis',alpha=0.7)
    pearson_corr,ccc,residual_std,m,b,coefficients,rmse = caculate_metrics(x_stick,y_stick)
    all_text = f'y = {m:.3f}x + ({b:.3f})\nres_std = {residual_std:.4f}\nPearson\'s r = {pearson_corr:.4f}\nLin\'s ccc = {ccc:.4f}\nRMSE = {rmse:.4f}'
    equation_text = f'y = {m:.3f}x + ({b:.3f})'
    plt.text(0,1,equation_text,fontsize = 20,color = 'black')

    line = np.poly1d(coefficients)
    x_line = np.linspace(min(x_stick), max(x_stick), 100)
    y_line = line(x_line)
    plt.plot(x_line, y_line, linestyle = '--', color='y')   
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)   
    plt.rcParams['font.family'] = 'sans-serif'
    plt.xticks(fontsize=15)
    plt.xlabel('True Fraction',size = 20)
    plt.yticks(fontsize=15)
    plt.ylabel('Predicted Fraction',size = 20)
    plt.legend(loc=loc)
    if save_choice == True:    
        plt.savefig(savename,format = 'svg',dpi = 300)
    plt.show()
    print(all_text)
    return [m,b,residual_std,pearson_corr,ccc,rmse]

def process_plot(sample_list,savename,save_choice,celltypenum):
    output1 = []
    target1 = []
    matrics_return = []
    for k in range(celltypenum):
        for j in  range(len(sample_list)):
            predict,target_ratio,ratio_num = sample_list[j]
            
            predict1 = predict[:,k].tolist()
            output1.append(predict1)
            target_ratio1 = target_ratio[k]
            target1.append(target_ratio1)
        output1 = ListDimConvert(output1)
        output1,target1 = withoutnan(output1,target1)
        x = plot_point_line(target1,output1,'C:/Users/ww/Desktop/scp/Leduc2022/fig4/'+f'{k}'+savename+'.svg','upper left',save_choice)
        matrics_return.append(x)
        x = []
        output1 = []
        target1 = []
    return matrics_return