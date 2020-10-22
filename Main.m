%% Compare our algorithms with Matlab's ANFIS on two small datasets
clc; clearvars; close all; rng('default'); warning off all;

nMFs=2;% number of MFs in each input domain
alpha=.1;% learning rate
rr=.05;%l2 coefficient
gama=0.8;%typical rate
lam=.1;%UR coefficient
nAlgs=4;%使用的算法个数
batchSize=128;
KFold=5;%k折
P=0.5;
PCA_dims=5; % PCA降维后的特征维度
nIt=150;%训练次数
nRepeats=1;%重复训练次数
nLabels=6; %分类数
% datasets={'xie1',...
%     'xie2',...
%     'xie3',...
%     'xie4',...
%     'xie5'};%.mat数据矩阵 
 datasets={'dis1',...
    'dis2',...
    'dis3',...
   	'dis4',...
   	'dis5'};%.mat数据矩阵 
% datasets={'liu1',...
%     'liu2',...
%     'liu3',...
%     'liu4',...
%     'liu5'};
% datasets={'zhou1',...
%     'zhou2',...
%     'zhou3',...
%     'zhou4',...
% %     'zhou5'};
% datasets={'cheng1',...
%     'cheng2',...
%     'cheng3',...
%     'cheng4',...
%     'cheng5'};

lineStyles={'k-','k--','k:','r-','r--','r:','g-','g--','g:','b-','b--','b:','c-','c--','c:','m-','m--','m:'};
% lineStyles={'k-','k--','r-','r--','g-','g--','b-','b--','c-','c--','m-','m--'};

CEtrain=cell(length(datasets),nRepeats); CEvalidation=cell(length(datasets),nRepeats); CEtest=cell(length(datasets),nRepeats); times=cell(length(datasets),nRepeats); usedDim=zeros(1,length(datasets));%初始化训练集、测试集均方差矩阵、训练时间、实际使用维度
BT=cell(length(datasets),nAlgs);
CT=BT;SigmaT=BT;VT=BT;GamaT=BT;BetaT=BT;

for s=1:length(datasets)%读取每一个.mat矩阵
    temp=load(['.\data\\' datasets{s} '.mat']);
    data=temp.data;
    %% 数据预处理
    %去掉零手势数据
    i=1;
    while i<=length(data)
        if data(i,end)==0
            data(i,:)=[];
        else
            i=i+1;
        end
    end
    
    %输入z标准化处理
%     X0=data(:,1:end-1); y0=data(:,end);% y_m=mean(y0); y0=y0-y_m;%X0提取除去6pMMG+3IMU数据，y0提取最后一列数据：数据对应的结果
%     X0=data(:,1:end-4); y0=data(:,end);% y_m=mean(y0); y0=y0-y_m;%X0提取除去6pMMG+2IMU数据，y0提取最后一列数据：数据对应的结果
%     X0=data(:,1:end-7); y0=data(:,end);% y_m=mean(y0); y0=y0-y_m;%X0提取除去6pMMG+第一组IMU数据，y0提取最后一列数据：数据对应的结果
    X0=[data(:,1:end-10),data(:,end-6:end-4)]; y0=data(:,end);% y_m=mean(y0); y0=y0-y_m;%X0提取除去6pMMG+1IMU数据，y0提取最后一列数据：数据对应的结果
%     X0=data(:,1:end-10); y0=data(:,end);% y_m=mean(y0); y0=y0-y_m;%X0提取除去6pMMG数据，y0提取最后一列数据：数据对应的结果
    X0 = zscore(X0); [N0,M]=size(X0);

    %% 训练&测试
    CEtrain{s}=nan(nAlgs,nIt,KFold); CEvalidation{s}=nan(nAlgs,nIt,KFold); CEtest{s}=nan(nAlgs,KFold); times{s}=nan(nAlgs,KFold);
    for r=1:nRepeats
        indices=crossvalind('Kfold',data(1:N0,M),KFold);%进行随机分包
        for k=1:KFold
            %划分测试集、验证集和训练集
            idsTest=(indices==k);
            if k<KFold
                idsValidation=(indices==k+1); 
            else
                idsValidation=(indices==k-1);
            end
            idsTrain=~(idsTest+idsValidation);
            XTrain=X0(idsTrain,:); yTrain=y0(idsTrain,:);
            XValidation=X0(idsValidation,:); yValidation=y0(idsValidation,:);
            XTest=X0(idsTest,:); yTest=y0(idsTest,:);
            %% 1 our MC  %核心 alpha 指数调 nMFs不用调
            tic;     
            [CEtrain{s,r}(1,:,k),CEvalidation{s,r}(1,:,k),CEtest{s,r}(1,k),BT{s,1},CT{s,1},SigmaT{s,1},VT{s,1}]=...
                MC(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,P,nMFs,nIt,nLabels,batchSize,PCA_dims);%MC
            times{s}(1,k)=toc;%计时
            
            %% 2 our MC_TS  %核心 alpha 指数调 nMFs不用调
            tic;     
            [CEtrain{s,r}(2,:,k),CEvalidation{s,r}(2,:,k),CEtest{s,r}(2,k),BT{s,2},CT{s,2},SigmaT{s,2},VT{s,2}]=...
                MC_TS(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,P,nMFs,nIt,nLabels,batchSize,gama,PCA_dims);%MC
            times{s}(2,k)=toc;%计时

            %% 3 our MC_UR  %核心 alpha 指数调 nMFs不用调
            tic;     
            [CEtrain{s,r}(3,:,k),CEvalidation{s,r}(3,:,k),CEtest{s,r}(3,k),BT{s,3},CT{s,3},SigmaT{s,3},VT{s,3}]=...
                MC_UR(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,lam,P,nMFs,nIt,nLabels,batchSize,PCA_dims);%MC
            times{s}(3,k)=toc;%计时

            %% 4 our MC_TS_UR  %核心 alpha 指数调 nMFs不用调
            tic;     
            [CEtrain{s,r}(4,:,k),CEvalidation{s,r}(4,:,k),CEtest{s,r}(4,k),BT{s,4},CT{s,4},SigmaT{s,4},VT{s,4}]=...
                MC_TS_UR(XTrain,yTrain,XValidation,yValidation,XTest,yTest,alpha,rr,lam,P,nMFs,nIt,nLabels,batchSize,gama,PCA_dims);%MC
            times{s}(4,k)=toc;%计时
 
%              %% 5 our OAA_LSE_GD  %核心 alpha 指数调 nMFs不用调
%             tic;     
%             [CEtrain{s}(5,:,r),CEtest{s}(5,:,r),BT{s,5},CT{s,5},SigmaT{s,5}]=...
%                 OAA_LSE_GD(XTrain,yTrain,XTest,yTest,alpha,nMFs,nIt,nLabels,batchSize,PCA_dims);%OAA
%             times{s}(5,r)=toc;%计时
        end
        
    end
end
% 绘制每个数据集上六种手势单独训练的
for i=1:nAlgs
    MCEtr=0;
    MCEva=0;
    MCEte=0;
    time_total=0;
    figure;
    set(gcf, 'Position', 1/2*get(0, 'Screensize'));hold on;
    for s=1:length(datasets)    
        MCEtr=MCEtr+mean(CEtrain{s}(i,end,:));
        MCEva=MCEva+mean(CEvalidation{s}(i,end,:));
        MCEte=MCEte+mean(CEtest{s}(i,:));
        CEtr=smooth(nanmean(CEtrain{s}(i,:,:),3),10);
        CEva=smooth(nanmean(CEvalidation{s}(i,:,:),3),10);
        CEte=nanmean(CEtest{s}(i,:),2).*ones(1,nIt);
        plot(CEtr,lineStyles{3*s-2},'linewidth',2);
        plot(CEva,lineStyles{3*s-1},'linewidth',2);
        plot(CEte,lineStyles{3*s},'linewidth',2);
        time_total=time_total+sum(times{s}(i,:));
    end
    set(gca,'yscale','log'); xlabel('Iteration'); ylabel('MC TS UR Classification Error');
    legend('Train1 ','Test1 ','Train2 ','Test2 ','Train3 ','Test3 ','Train4 ','Test4 ','Train5 ','Test5 ','location','northeast');  
    disp(['Mean_Train_Error:' num2str(MCEtr/length(datasets))]);
    disp(['Mean_Validation_Error:' num2str(MCEva/length(datasets))]);
    disp(['Mean_Test_Error:' num2str(MCEte/length(datasets))]);
    disp(['Mean_Time:' num2str(time_total/(nRepeats*length(datasets)))]);
end
save(['MC1' num2str(alpha) num2str(gama) num2str(lam) ' .mat'],'CEtrain','CEvalidation','CEtest','times');
% save('parameters.mat','BT','CT','SigmaT','VT');