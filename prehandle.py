#coding:utf-8

#KDD99数据集预处理
#共使用39个特征，去除了原数据集中20、21号特征

import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn import preprocessing # 数据标准化处理



#定义KDD99字符型特征转数值型特征函数
def char2num(sourceFile, handledFile):
    print('START: 字符型特征转数值型特征函数中')
    data_file=open(handledFile,'w',newline='')     #python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    global dataCnt
    with open(sourceFile, 'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        dataCnt=0   #记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line=np.array(row)   #将每行数据存入temp_line数组里
            temp_line[1]=handleProtocol(row)   #将源文件行中3种协议类型转换成数字标识
            temp_line[2]=handleService(row)    #将源文件行中70种网络服务类型转换成数字标识
            temp_line[3]=handleFlag(row)       #将源文件行中11种网络连接状态转换成数字标识
            temp_line[41]=handleLabel(row)   #将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)
            dataCnt+=1
            #输出每行数据中所修改后的状态
    data_file.close()
    print('FINISH: 字符型特征转数值型特征函数完成\n')

#将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]

#定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol(input):
    protocol_list=['tcp','udp','icmp']
    if input[1] in protocol_list:
        return find_index(input[1],protocol_list)[0]

#定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService(input):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                  'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                  'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                  'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                  'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                  'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                  'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    if input[2] in service_list:
        return find_index(input[2],service_list)[0]

#定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)[0]

#定义将源文件行中攻击类型转换成数字标识的函数(共出现了22个攻击类型+1个未受到攻击)
def handleLabel(input):
    global label_list
    label_list = ['normal.',  # normal
                  'back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.',  # DOS
                  'ipsweep.', 'nmap.', 'portsweep.', 'satan.',  # PROBE
                  'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.',  # R2L
                  'buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']  # U2R

    if input[41] in label_list:
        return find_index(input[41], label_list)[0]
    else:
        label_list.append(input[41])
        return find_index(input[41], label_list)[0]

def standardize(inputFile):
    import warnings
    # 忽略UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
    # warnings.warn("Numerical issues were encountered "
    warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
    print('START: 数据标准化中')
    dataMatrix = np.loadtxt(open(inputFile,"rb"),delimiter=",",skiprows=0) # 读入数据
    labelColumn = dataMatrix[:,-1]
    result = preprocessing.scale(dataMatrix[:,:-1]) # 标签列不参与训练
    print('FINISH: 数据标准化完成\n')
    return result, labelColumn

def normalize(inMatrix):
    print('START: 数据归一化中')
    np.seterr(divide='ignore',invalid='ignore') # 忽略0/0的报错
    minVals = inMatrix.min(0)
    maxVals = inMatrix.max(0)
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(inMatrix))
    m = inMatrix.shape[0]
    normData = inMatrix - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    # 去掉数据中的空列
    print('FINISH: 数据归一化完成\n')
    return normData, ranges, minVals




def exportData(npData, outputFile):
    

    
    pd_data = pd.DataFrame(npData, columns=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                                            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                                            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                                            'is_host_login', 'is_guest_login', 'count',  'srv_count',  'serror_rate', 'srv_serror_rate',
                                            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                                            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                                            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'])
    pd_data.drop('num_outbound_cmds', axis=1, inplace=True)  # 删除存在空值的列
    pd_data.drop('is_host_login', axis=1, inplace=True)  # 删除存在空值的列
    pd_data.to_csv(outputFile, header=None, index=None)
    



def run(source,temp):
    char2num(source, temp)  # 字符型特征转数字型特征
    stdData, labelColumn = standardize(temp)
    normData, _, _ = normalize(stdData)

    #数据集乱序
    np.random.seed(116)
    np.random.shuffle(normData)
    np.random.seed(116)
    np.random.shuffle(labelColumn)
    
    #按6：2：2分出训练集，验证集和测试集
    n_data=len(labelColumn)
    split_ind1 = int(n_data * 0.6)
    split_ind2 = int(n_data * 0.8)

    train_data=normData[:split_ind1,:]
    train_label = labelColumn[:split_ind1]
    val_data=normData[split_ind1:split_ind2,:]
    val_label = labelColumn[split_ind1:split_ind2]
    test_data=normData[split_ind2:,:]
    test_label = labelColumn[split_ind2:]
    
    

    label = pd.DataFrame(train_label,columns=["attack_type"])
    label.to_csv(".//dataset//"+"train_label.csv", header=None, index=None)
    label = pd.DataFrame(val_label, columns=["attack_type"])
    label.to_csv(".//dataset//"+"val_label.csv", header=None, index=None)
    label = pd.DataFrame(test_label, columns=["attack_type"])
    label.to_csv(".//dataset//"+"test_label.csv", header=None, index=None)
    
    print('START: 数据导出中')
    exportData(train_data, ".//dataset//"+"train_data.csv")
    exportData(val_data, ".//dataset//"+"val_data.csv")
    exportData(test_data, ".//dataset//"+"test_data.csv")
    
    print(f'FINISH: 数据导出成功\n共导出 {dataCnt} 条数据')


if __name__=='__main__': 
    start_time=datetime.now()

    sourceFile= './/dataset//kddcup.data_10_percent_corrected'
    deCharFile = './/dataset//decharedData.csv'
    
    run(sourceFile,deCharFile)
    
    end_time=datetime.now()
    print("运行时间 ",(end_time-start_time),'s')  #输出程序运行时间
