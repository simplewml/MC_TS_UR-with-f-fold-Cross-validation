function [CEtrain,CEvalidation,CEtest,BT,CT,SigmaT,VT]=MC1(XTr,yTr,XVa,yVa,XTe,yTe,alpha,rr,P,numMFs,numIt,plabel,batchSize,PCA_dims)
%XTrain、yTrain训练数据集；XTest、yTest测试数据集；alpha学习率；numMFs隶属函数个数；numIt训练次数；batchSize训练数据量
beta1=0.9; beta2=0.999; 
MinCE=1; count=0; Maxcount=numIt;

%% 手势数据按类打标签 softmax
yTrain=zeros(length(yTr),plabel);
for i=1:length(yTr)
    for j=1:plabel
        if yTr(i)==j
            yTrain(i,j)=1;
        end
    end
end
yValidation=zeros(length(yVa),plabel);
for i=1:length(yVa)
    for j=1:plabel
        if yVa(i)==j
            yValidation(i,j)=1;
        end
    end
end
yTest=zeros(length(yTe),plabel);
for i=1:length(yTe)
    for j=1:plabel
        if yTe(i)==j
            yTest(i,j)=1;
        end
    end
end

[N,M]=size(XTr);[Nv,~]=size(XVa); NValidation=size(XVa,1); NTest=size(XTe,1);
if M>PCA_dims%输入的维度大于最大输入特征维度，用PCA降维
    [~,XPCA,latent]=pca([XTr;XVa;XTe]);%XPCA：原始数据在新生成的主成分空间里的坐标值，latent：X0协方差矩阵特征值（XPCA每一列的方差）从大到小排列
    realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');%实际使用特征最大数据维度
    usedDim=min(PCA_dims,realDim98);
    XTrain=XPCA(1:N,1:usedDim);XValidation=XPCA(N+1:N+Nv,1:usedDim); XTest=XPCA(N+Nv+1:end,1:usedDim); [N,M]=size(XTrain);%实际使用的训练、测试数据集
end

numMFsVec=numMFs*ones(M,1);
R=numMFs^M; % number of rules
C=zeros(M,numMFs); Sigma0=C; B=ones(R,M+1);%C、Sigma每个隶属函数的mu、sigma，前向参数；B：x的系数和常数项，后向参数
V=ones(R,plabel);
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),numMFs);%C初值
    Sigma0(m,:)=std(XTrain(:,m));%每一列的标准差
end
Sigma=Sigma0;

%% Iterative update
mu=zeros(M,numMFs);
output=zeros(batchSize,1);
yorg=output;
mC=0; vC=0; mSigma=0; vSigma=0; mB=0; vB=0; mV=0; vV=0; yPred=nan(batchSize,plabel); ycPred=yPred;

for it=1:numIt %numIt次训练
    deltaC=zeros(M,numMFs); deltaSigma=deltaC;    deltaB=rr*B; deltaB(:,1)=0; deltaV=zeros(R,plabel);% consequent
    f=ones(batchSize,R); % firing level of rules
    idsTrain=datasample(1:N,batchSize,'replace',false);%训练集的训练数据index
    for n=1:batchSize%一次训练使用的数据量
        for m=1:M % membership grades of MFs 层1
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));%高斯隶属函数模糊化结果
        end
        % droprule
        for r=1:R
            if rand<P
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            else
                f(n,r)=0;
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            f(n,:)=1;
            for r=1:R
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        fBar=f(n,:)/sum(f(n,:));%得到归一化触发强度w 层3
        yR=[1 XTrain(idsTrain(n),:)]*B';%计算每个规则的结果矩阵y 层4     
        yPred(n,:)=fBar.*yR*V; %层5
        ycPred(n,:)=exp( yPred(n,:) ) ./ sum(exp(yPred(n,:)) ); %softmax prediction 层6
        
         % Compute delta softmax
        for r=1:R%对于每个规则
            temp=-sum((yTrain(idsTrain(n),:)./ycPred(n,:)-(1.-yTrain(idsTrain(n),:))./(1.-ycPred(n,:))).*(ycPred(n,:).*(1.-ycPred(n,:))).*V(r,:))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);%第n组样本训练的误差*（规则r的结果-yPred)*规则r占的触发强度比重
            if ~isnan(temp) && abs(temp)<inf%没有数据缺失&&触发强度之和不为0
                vec=idx2vec(r,numMFsVec);
                % delta of c, sigma, and b
                for m=1:M%输入特征维度
                    deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaB(r,m+1)=deltaB(r,m+1)-sum((yTrain(idsTrain(n),:)./ycPred(n,:)-(1.-yTrain(idsTrain(n),:))./(1.-ycPred(n,:))).*(ycPred(n,:).*(1.-ycPred(n,:))).*V(r,:))*fBar(r)*XTrain(idsTrain(n),m);%deltaB：2:m+1为第n组样本训练的误差*规则r的触发强度比重*训练数据
                end
                % delta of br0
                deltaB(r,1)=deltaB(r,1)-sum((yTrain(idsTrain(n),:)./ycPred(n,:)-(1.-yTrain(idsTrain(n),:))./(1.-ycPred(n,:))).*(ycPred(n,:).*(1.-ycPred(n,:))).*V(r,:))*fBar(r);%deltaB：1为第n组样本训练的误差*规则r的触发强度比重
                for p=1:plabel
                    deltaV(r,p)=deltaV(r,p)-((yTrain(idsTrain(n),p)./ycPred(n,p)-(1.-yTrain(idsTrain(n),p))/(1.-ycPred(n,p))).*(ycPred(n,p).*(1.-ycPred(n,p))))*fBar(r)*yR(r);
                end
            end
        end
    end
    
     % Training error softmax
    for i=1:length(ycPred)
    	[~,output(i)]=max(ycPred(i,:));
        [~,yorg(i)]=max(yTrain(idsTrain(i),:));
    end
    CEtrain(it)=1-sum(output-yorg==0)/length(ycPred);
    
    % Adam
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    mB=beta1*mB+(1-beta1)*deltaB;
    vB=beta2*vB+(1-beta2)*deltaB.^2;
    mBHat=mB/(1-beta1^it);
    vBHat=vB/(1-beta2^it);
    mV=beta1*mV+(1-beta1)*deltaV;
    vV=beta2*vV+(1-beta2)*deltaV.^2;
    mVHat=mV/(1-beta1^it);
    vVHat=vV/(1-beta2^it);
    % update C, Sigma and B, using AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat;
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(.1*Sigma0,Sigma-lrSigma.*mSigmaHat);
    lrB=min(ub,max(lb,alpha./(sqrt(vBHat)+10^(-8))));
    B=B-lrB.*mBHat;
    lrV=min(ub,max(lb,alpha./(sqrt(vVHat)+10^(-8))));
    V=V-lrV.*mVHat;
    
    % Validation error
    f=ones(NValidation,R); % firing level of rules
    for n=1:NValidation
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XValidation(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        
        for r=1:R % firing levels of rules
            idsMFs=idx2vec(r,numMFsVec);
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));
            end
        end
    end
    yR=[ones(NValidation,1) XValidation]*B';%计算每个规则的结果矩阵y 层4     
    yPredValidation=f.*yR./sum(f,2)*V; 
    ycPredValidation=exp(yPredValidation)./sum(exp(yPredValidation),2); % prediction
    for i=NValidation
        [~,yVaPred]=max((ycPredValidation'));
        [~,yv]=max((yValidation'));
    end
    CEvalidation(it)=1-sum(yVaPred-yv==0)/NValidation;
    
    %early stopping
    if CEvalidation(it)<MinCE
        MinCE=CEvalidation(it);
        count=0;
    else
        count=count+1;
    end
    %% compute and visualize the confusion matrix
%     if (it==numIt)||(count>Maxcount)
    if it==numIt
        f=ones(NTest,R); % firing level of rules
        for n=1:NTest
            for m=1:M % membership grades of MFs
                mu(m,:)=exp(-(XTest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
            end

            for r=1:R % firing levels of rules
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        yR=[ones(NTest,1) XTest]*B';%计算每个规则的结果矩阵y 层4     
        yPredTest=f.*yR./sum(f,2)*V; 
        ycPredTest=exp(yPredTest)./sum(exp(yPredTest),2); % prediction
        for i=1:NTest
            [~,yTePred]=max((ycPredTest'));
            [~,yt]=max((yTest'));
        end
        CEtest=1-sum(yTePred-yt==0)/NTest;
        
        actual_label=yt';
        NumInClass=zeros(plabel,1);
        for i=1:plabel
            NumInClass(i)=sum(yt==i);
        end
        NameClass={'1';'2';'3';'4';'5';'6'};
        [ConfusionMatrix]=compute_confusion_matrix(yTePred',NumInClass,NameClass);
        break;
    end
end

BT=B;
CT=C;
SigmaT=Sigma;
VT=V;
