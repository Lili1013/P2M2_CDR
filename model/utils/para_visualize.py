import matplotlib.pyplot as plt
import numpy as np
import random

def draw_emb_size():
    # fig,ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # plt.figure(figsize=(12,12))
    fig,axs = plt.subplots(2,3,figsize=(10,10))
    x = np.arange(4)
    bar_width=0.5
    name_list = ['8','16','32','64']
    value_rmse_ciao = [0.8905,0.8736,0.8538,0.8687]
    value_mae_ciao = [0.7068,0.6810,0.6406,0.6509]
    value_rmse_epinions = [0.9792,0.9568,0.9453,0.9534]
    value_mae_epinions = [0.7886,0.7467,0.7253,0.7324]
    value_rmse_dianping = [0.7193,0.7067,0.6893,0.6918]
    value_mae_dianping = [0.5560,0.5514,0.5311,0.5424]
    # plt.subplot(2,3,1)
    for a,b in zip(x,value_rmse_ciao):
        axs[0,0].text(a,b,b,ha='center',va='bottom',)
    axs[0,0].bar(x,value_rmse_ciao,bar_width,label='RMSE',color='red')
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(name_list)
    axs[0, 0].set_ylabel('RMSE')
    axs[0,0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)

    # plt.subplot(2,3,4)
    for a,b in zip(x,value_mae_ciao):
        axs[1,0].text(a,b,b,ha='center',va='bottom',)
    axs[1,0].bar(x,value_mae_ciao,bar_width,label='MAE',color='red')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(name_list)

    # axs[0,0].xticks(x+bar_width/2,name_list)
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].set_xlabel('Ciao')
    # axs[1,0].xticks(x+bar_width/2,name_list)

    # plt.subplot(2,3,2)
    for a,b in zip(x,value_rmse_epinions):
        axs[0,1].text(a,b,b,ha='center',va='bottom',)
    axs[0,1].bar(x,value_rmse_epinions,bar_width,label='RMSE')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(name_list)
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)


    # plt.subplot(2,3,5)
    for a,b in zip(x,value_mae_epinions):
        axs[1,1].text(a,b,b,ha='center',va='bottom',)
    axs[1,1].bar(x,value_mae_epinions,bar_width,label='MAE')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(name_list)
    axs[1, 1].set_ylabel('MAE')
    axs[1, 1].set_xlabel('Epinions')
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)


    # plt.subplot(2,3,3)
    for a,b in zip(x,value_rmse_dianping):
        axs[0,2].text(a,b,b,ha='center',va='bottom',)
    axs[0,2].bar(x,value_rmse_dianping,bar_width,label='RMSE',color='green')
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(name_list)
    axs[0, 2].set_ylabel('RMSE')
    axs[0, 2].spines['top'].set_visible(False)
    axs[0, 2].spines['right'].set_visible(False)
    # axs[0,2].xticks(x+bar_width/2,name_list)

    # plt.subplot(2,3,6)
    for a,b in zip(x,value_mae_dianping):
        axs[1,2].text(a,b,b,ha='center',va='bottom',)
    axs[1,2].bar(x,value_mae_dianping,bar_width,label='MAE',color='green')
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels(name_list)
    axs[1, 2].set_ylabel('MAE')
    axs[1, 2].set_xlabel('Dianping')
    # axs[0,0].xticks(x+bar_width/2,name_list)
    axs[1, 2].spines['top'].set_visible(False)
    axs[1, 2].spines['right'].set_visible(False)

    # axs[1,2].xticks(x+bar_width/2,name_list)
    plt.subplots_adjust(hspace=0.3,wspace=0.5)
    plt.show()

def draw_beta():
    x = np.linspace(0,12,12)
    y_ciao_rmse = [0.8603,0.8578,0.8539,0.8564,0.8586,0.8587,0.8538,0.8522,0.8591,0.8586,0.8588,0.8537]
    y_ciao_mae = [0.6569,0.6528,0.6480,0.6501,0.6550,0.6537,0.6406,0.6482,0.6466,0.6515,0.6502,0.6476]
    y_epinions_rmse = []
    y_epinions_mae = random.sample(range(12),12)
    y_dianping_rmse = random.sample(range(12),12)
    y_dianping_mae = random.sample(range(12),12)
    fig,ax = plt.subplots(2,3,figsize=(15,5))
    ax[0,0].plot(x,y_ciao_rmse,label='line1',color='green')
    ax[0,0].set_xticks(np.linspace(0,12,12))
    ax[0,0].set_xticklabels(['0.001','0.01','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],rotation=45)
    ax[0, 0].set_ylabel('RMSE')


    ax[0,1].plot(x,y_epinions_rmse,label='line2',color = 'purple')
    ax[0,1].set_xticks(np.linspace(0, 12, 12))
    ax[0,1].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[0, 1].set_ylabel('RMSE')

    ax[0,2].plot(x, y_dianping_rmse, label='line2', color='blue')
    ax[0,2].set_xticks(np.linspace(0, 12, 12))
    ax[0,2].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[0, 2].set_ylabel('RMSE')

    ax[1,0].plot(x, y_ciao_mae, label='line2', color='green')
    ax[1,0].set_xticks(np.linspace(0, 12, 12))
    ax[1,0].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 0].set_ylabel('MAE')
    ax[1, 0].set_xlabel('Ciao')

    ax[1,1].plot(x, y_epinions_mae, label='line2', color='purple')
    ax[1,1].set_xticks(np.linspace(0, 12, 12))
    ax[1,1].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 1].set_ylabel('MAE')
    ax[1, 1].set_xlabel('Epinions')

    ax[1,2].plot(x, y_dianping_mae, label='line2', color='blue')
    ax[1,2].set_xticks(np.linspace(0, 12, 12))
    ax[1,2].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 2].set_ylabel('MAE')
    ax[1, 2].set_xlabel('Dianping')

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()
    pass

def draw_emb_dim():
    plt.figure(figsize=(8,5))
    x_labels = ['16','32','64','128']
    # #phone_sport
    y_source_hr = [0.5303,0.7133,0.9281,0.9022]
    y_source_ndcg = [0.3199,0.5023,0.5827,0.5633]
    y_target_hr = [0.5966,0.7645,0.9151,0.9039]
    y_target_ndcg = [0.3313,0.5329,0.5998,0.5823]
    #sport_cloth
    # y_source_hr = [0.5679, 0.7239, 0.8964, 0.8804]
    # y_source_ndcg = [0.3243, 0.4156, 0.5491, 0.5287]
    # y_target_hr = [0.6718, 0.6965, 0.8642, 0.8568]
    # y_target_ndcg = [0.3727, 0.3917, 0.5236, 0.5153]
    bar_width = 0.1
    index = np.arange(len(x_labels))
    # plt.bar(index-0.15,y_source_hr,bar_width,color='lightblue',hatch='//',edgecolor='black',label='Phone-HR@10')
    # plt.bar(index-0.05,y_target_hr,bar_width,color='lightblue',hatch='**',edgecolor='black',label='Sport-HR@10')
    # plt.bar(index+0.05,y_source_ndcg,bar_width,color='lightgreen',hatch='//',edgecolor='black',label='Phone-NDCG@10')
    # plt.bar(index+0.15,y_target_ndcg,bar_width,color='lightgreen',hatch='**',edgecolor='black',label='Sport-NDCG@10')
    plt.bar(index - 0.15, y_source_hr, bar_width, color='lightblue', hatch='//', edgecolor='black', label='Sport-HR@10')
    plt.bar(index - 0.05, y_target_hr, bar_width, color='lightblue', hatch='**', edgecolor='black', label='Cloth-HR@10')
    plt.bar(index + 0.05, y_source_ndcg, bar_width, color='lightgreen', hatch='//', edgecolor='black',
            label='Sport-NDCG@10')
    plt.bar(index + 0.15, y_target_ndcg, bar_width, color='lightgreen', hatch='**', edgecolor='black',
            label='Cloth-NDCG@10')
    plt.xticks(index,x_labels)
    plt.legend()
    plt.show()

    pass

def draw_alpha():
    plt.figure(figsize=(8,5))
    x_labels = ['0.0001', '0.0005', '0.001', '0.005','0.01']
    #phone_sport
    y_source_hr = [0.9073, 0.8772, 0.9281, 0.7833, 0.7831]
    y_source_ndcg = [0.5669, 0.5435, 0.5827, 0.4603, 0.4598]
    y_target_hr = [0.9014, 0.9112, 0.9151, 0.9024, 0.8897]
    y_target_ndcg = [0.6033, 0.5774, 0.5998, 0.5706, 0.5656]
    #sport_cloth
    # y_source_hr = [0.8602, 0.8518, 0.8964, 0.8464, 0.8479]
    # y_source_ndcg = [0.5174, 0.5094, 0.5491, 0.5005, 0.5009]
    # y_target_hr = [0.8092, 0.7897, 0.8642, 0.7670, 0.7918]
    # y_target_ndcg = [0.4752, 0.4633, 0.5236, 0.4433, 0.4619]
    index = np.arange(len(x_labels))
    # plt.plot(index,y_source_hr,color = 'red', marker ='o',linestyle='-',label='Phone-HR@10')
    # plt.plot(index,y_target_hr,color='red',marker='*',linestyle='--',label='Sport-HR@10')
    # plt.plot(index,y_source_ndcg,color='blue',marker='o',linestyle='-',label='Phone-NDCG@10')
    # plt.plot(index,y_target_ndcg,color='blue',marker='*',linestyle='--',label='Sport-NDCG@10')

    plt.plot(index, y_source_hr, color='red', marker='o', linestyle='-', label='Sport-HR@10')
    plt.plot(index, y_target_hr, color='red', marker='*', linestyle='--', label='Cloth-HR@10')
    plt.plot(index, y_source_ndcg, color='blue', marker='o', linestyle='-', label='Sport-NDCG@10')
    plt.plot(index, y_target_ndcg, color='blue', marker='*', linestyle='--', label='Cloth-NDCG@10')
    plt.xticks(index, x_labels)
    plt.legend(loc='center left')
    plt.show()
    pass
def draw_lambda():
    plt.figure(figsize=(8, 5))
    x_labels = ['0.005', '0.01', '0.02', '0.03','0.04']
    # #phone_sport
    y_source_hr = [0.9358, 0.9281, 0.9104, 0.9023, 0.8868]
    y_source_ndcg = [0.5883, 0.5827, 0.5703, 0.5627, 0.5377]
    y_target_hr = [0.9236, 0.9151, 0.8921, 0.8842, 0.8745]
    y_target_ndcg = [0.6063, 0.5998, 0.5581, 0.5448, 0.5301]
    #sport_cloth
    # y_source_hr = [0.8996, 0.8964, 0.8630, 0.8564, 0.8291]
    # y_source_ndcg = [0.5529, 0.5491, 0.5197, 0.5106, 0.4890]
    # y_target_hr = [0.8765, 0.8642, 0.7948, 0.7878, 0.7782]
    # y_target_ndcg = [0.5312, 0.5236, 0.4661, 0.4582, 0.4522]
    index = np.arange(len(x_labels))
    # plt.plot(index, y_source_hr, color='purple', marker='o', linestyle='-', label='Phone_HR@10')
    # plt.plot(index, y_target_hr, color='purple', marker='*', linestyle='--', label='Sport-HR@10')
    # plt.plot(index, y_source_ndcg, color='gray', marker='o', linestyle='-', label='Phone-NDCG@10')
    # plt.plot(index, y_target_ndcg, color='gray', marker='*', linestyle='--', label='Sport-NDCG@10')

    plt.plot(index, y_source_hr, color='purple', marker='o', linestyle='-', label='Sport-HR@10')
    plt.plot(index, y_target_hr, color='purple', marker='*', linestyle='--', label='Cloth-HR@10')
    plt.plot(index, y_source_ndcg, color='gray', marker='o', linestyle='-', label='Sport-NDCG@10')
    plt.plot(index, y_target_ndcg, color='gray', marker='*', linestyle='--', label='Cloth-NDCG@10')
    plt.xticks(index, x_labels)
    plt.legend(loc='center left')
    plt.show()


    # bar_width = 0.1
    # index = np.arange(len(x_labels))
    # plt.bar(index - 0.15, y_source_hr, bar_width, color='lightpurple', hatch='//', edgecolor='black', label='Phone-HR@10')
    # # plt.xticks(index)
    # # plt.xlabel(x_labels)
    # plt.bar(index - 0.05, y_target_hr, bar_width, color='lightpurple', hatch='**', edgecolor='black', label='Sport-HR@10')
    # plt.bar(index + 0.05, y_source_ndcg, bar_width, color='lightgray', hatch='//', edgecolor='black',
    #         label='Phone-NDCG@10')
    # plt.bar(index + 0.15, y_target_ndcg, bar_width, color='lightgray', hatch='**', edgecolor='black',
    #         label='Sport-NDCG@10')
    # plt.xticks(index, x_labels)
    # plt.legend()
    # plt.show()

def draw_agg_way_phone_sport():
    plt.figure(figsize=(8, 5))
    x_labels = ['Element-wise Sum', 'Concatenation', 'Attention']
    # #phone_sport
    y_source_hr = [0.9281,0.8275,0.8970]
    y_source_ndcg = [0.5827,0.4876,0.5554]
    y_target_hr = [0.9151,0.8104,0.8885]
    y_target_ndcg = [0.5998,0.4782,0.5577]
    # sport_cloth
    # y_source_hr = [0.5679, 0.7239, 0.8964, 0.8804]
    # y_source_ndcg = [0.3243, 0.4156, 0.5491, 0.5287]
    # y_target_hr = [0.6718, 0.6965, 0.8642, 0.8568]
    # y_target_ndcg = [0.3727, 0.3917, 0.5236, 0.5153]
    bar_width = 0.1
    index = np.arange(len(x_labels))
    # plt.bar(index-0.15,y_source_hr,bar_width,color='lightblue',hatch='//',edgecolor='black',label='Phone-HR@10')
    # plt.bar(index-0.05,y_target_hr,bar_width,color='lightblue',hatch='**',edgecolor='black',label='Sport-HR@10')
    # plt.bar(index+0.05,y_source_ndcg,bar_width,color='lightgreen',hatch='//',edgecolor='black',label='Phone-NDCG@10')
    # plt.bar(index+0.15,y_target_ndcg,bar_width,color='lightgreen',hatch='**',edgecolor='black',label='Sport-NDCG@10')
    plt.bar(index - 0.15, y_source_hr, bar_width, color='yellow', hatch='//', edgecolor='black', label='Phone-HR@10')
    plt.bar(index - 0.05, y_target_hr, bar_width, color='yellow', hatch='**', edgecolor='black', label='Sport-HR@10')
    plt.bar(index + 0.05, y_source_ndcg, bar_width, color='pink', hatch='//', edgecolor='black',
            label='Phone-NDCG@10')
    plt.bar(index + 0.15, y_target_ndcg, bar_width, color='pink', hatch='**', edgecolor='black',
            label='Sport-NDCG@10')
    plt.xticks(index, x_labels)
    plt.legend(loc=(0.52,0.75))
    plt.show()
def draw_agg_way_sport_cloth():
    plt.figure(figsize=(8, 5))
    x_labels = ['Element-wise Sum', 'Concatenation', 'Attention']
    # #phone_sport
    y_source_hr = [0.8964,0.8587,0.8555]
    y_source_ndcg = [0.5491,0.5143,0.5148]
    y_target_hr = [0.8642,0.7924,0.7751]
    y_target_ndcg = [0.5236,0.4570,0.4505]
    # sport_cloth
    # y_source_hr = [0.5679, 0.7239, 0.8964, 0.8804]
    # y_source_ndcg = [0.3243, 0.4156, 0.5491, 0.5287]
    # y_target_hr = [0.6718, 0.6965, 0.8642, 0.8568]
    # y_target_ndcg = [0.3727, 0.3917, 0.5236, 0.5153]
    bar_width = 0.1
    index = np.arange(len(x_labels))
    # plt.bar(index-0.15,y_source_hr,bar_width,color='lightblue',hatch='//',edgecolor='black',label='Phone-HR@10')
    # plt.bar(index-0.05,y_target_hr,bar_width,color='lightblue',hatch='**',edgecolor='black',label='Sport-HR@10')
    # plt.bar(index+0.05,y_source_ndcg,bar_width,color='lightgreen',hatch='//',edgecolor='black',label='Phone-NDCG@10')
    # plt.bar(index+0.15,y_target_ndcg,bar_width,color='lightgreen',hatch='**',edgecolor='black',label='Sport-NDCG@10')
    plt.bar(index - 0.15, y_source_hr, bar_width, color='yellow', hatch='//', edgecolor='black', label='Sport-HR@10')
    plt.bar(index - 0.05, y_target_hr, bar_width, color='yellow', hatch='**', edgecolor='black', label='Cloth-HR@10')
    plt.bar(index + 0.05, y_source_ndcg, bar_width, color='pink', hatch='//', edgecolor='black',
            label='Sport-NDCG@10')
    plt.bar(index + 0.15, y_target_ndcg, bar_width, color='pink', hatch='**', edgecolor='black',
            label='Cloth-NDCG@10')
    plt.xticks(index, x_labels)
    plt.legend(loc=(0.52,0.75))
    plt.show()



if __name__ == '__main__':
    # draw_emb_size()
    # draw_emb_dim()
    # draw_alpha()
    # draw_lambda()
    # draw_alpha()
    # draw_beta()
    draw_agg_way_phone_sport()
