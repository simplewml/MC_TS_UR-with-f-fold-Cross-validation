function [CEtrain,CEvalidation,CEtest,BT,CT,SigmaT,VT]=MC_TS(XTr,yTr,XVa,yVa,XTe,yTe,alpha,rr,P,numMFs,numIt,plabel,batchSize,gama,PCA_dims)
beta1=0.9; beta2=0.999;
MinCE=1; count=0; Maxcount=numIt;

%% ????????? softmax
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

%get xH,xL
XTrainH=[];
yTrainH=[];
XTrainL=[];
yTrainL=[];
X = tsne(XTrain);
p=kde(X', [.15;.1;], [], 'Gaussian');
dX = evaluate(p,X');
sdX=sort(dX,'descend');
Srsd=sdX(find(cumsum(sdX)>=gama*sum(dX),1,'first'));
for i=1:N
    if dX(i)>=Srsd
        XTrainH=[XTrainH;XTrain(i,:)];
        yTrainH=[yTrainH;yTrain(i,:)];
    else
        XTrainL=[XTrainL;XTrain(i,:)];
        yTrainL=[yTrainL;yTrain(i,:)];
    end
end
Nh=length(XTrainH);
Nl=length(XTrainL);
nh=round(0.8*batchSize);
nl=batchSize-nh;

%initialize
numMFsVec=numMFs*ones(M,1);
R=numMFs^M; % number of rules
C=zeros(M,numMFs); Sigma0=C; B=ones(R,M+1); V=ones(R,plabel);
for m=1:M % Initialization
    C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),numMFs);%C??
    Sigma0(m,:)=std(XTrain(:,m));%???????
end
Sigma=Sigma0;
% Gama=0.5*ones(1,M);Beta=0.5*ones(1,M);ep=10^(-8);

%% Iterative update
mu=zeros(M,numMFs);
output=zeros(batchSize,1);
yorg=output;
mC=0; vC=0; mSigma=0; vSigma=0; mB=0; vB=0; mV=0; vV=0;
yPred=nan(batchSize,plabel); ycPred=yPred;

for it=1:numIt 
    deltaC=zeros(M,numMFs); deltaSigma=deltaC; deltaB=rr*B; deltaB(:,1)=0; 
    deltaV=zeros(R,plabel);% consequent
%     deltaV=rr*V;
%     deltaGama=zeros(1,M);deltaBeta=zeros(1,M);
    f=ones(batchSize,R); % firing level of rules
    idsTrainH=datasample(1:Nh,nh,'replace',false);%????????index
    idsTrainL=datasample(1:Nl,nl,'replace',false);%????????index
    
    %TB
    mnXTrain=[XTrainH(idsTrainH,:);XTrainL(idsTrainL,:)];
    mnyTrain=[yTrainH(idsTrainH,:);yTrainL(idsTrainL,:)];
    
    for n=1:batchSize%??????????
        for m=1:M % membership grades of MFs ?1
            mu(m,:)=exp(-(mnXTrain(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));%???????????
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
        
        fBar=f(n,:)/sum(f(n,:));%?????????w ?3
        yR=[1 mnXTrain(n,:)]*B';%???????????y ?4
        yPred(n,:)=fBar.*yR*V; %?5
        ycPred(n,:)=exp( yPred(n,:) ) ./ sum(exp(yPred(n,:)) ); %softmax prediction ?6
        
          % Compute delta softmax
        for r=1:R%??????
            temp=-sum((mnyTrain(n,:)./ycPred(n,:)-(1.-mnyTrain(n,:))./(1.-ycPred(n,:))).*ycPred(n,:).*(1.-ycPred(n,:)).*V(r,:).*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r));%?n????????*???r???-yPred)*??r????????
            if ~isnan(temp) && abs(temp)<inf%??????&&????????0
                vec=idx2vec(r,numMFsVec);
                % delta of c, sigma, and b
                for m=1:M%??????
                    deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(mnXTrain(n,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(mnXTrain(n,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaB(r,m+1)=deltaB(r,m+1)-sum((mnyTrain(n,:)./ycPred(n,:)-(1.-mnyTrain(n,:))./(1.-ycPred(n,:))).*(ycPred(n,:).*(1-ycPred(n,:)).*V(r,:))*fBar(r)*mnXTrain(n,m));%deltaB?2:m+1??n????????*??r???????*????
                end
                % delta of br0
                deltaB(r,1)=deltaB(r,1)-sum((mnyTrain(n,:)./ycPred(n,:)-(1.-mnyTrain(n,:))./(1.-ycPred(n,:))).*(ycPred(n,:).*(1.-ycPred(n,:)).*V(r,:))*fBar(r));%deltaB?1??n????????*??r???????
                for p=1:plabel
                    deltaV(r,p)=deltaV(r,p)-((mnyTrain(n,p)./ycPred(n,p)-(1-mnyTrain(n,p))/(1-ycPred(n,p)))*(ycPred(n,p)*(1-ycPred(n,p))))*fBar(r)*yR(r);
                end
            end
        end
    end
     
     % Training error softmax
    for i=1:length(ycPred)
        [~,output(i)]=max(ycPred(i,:));
        [~,yorg(i)]=max(mnyTrain(i,:));
    end
    CEtrain(it)=1-sum(output-yorg==0)/length(ycPred);
    
    % Adabound
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
    % update C, Sigma, B and V using AdaBound
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
    yR=[ones(NValidation,1) XValidation]*B';%???????????y ?4 
    yPredValidation=f.*yR./sum(f,2)*V; 
    ycPredValidation=exp(yPredValidation)./sum(exp(yPredValidation),2); % prediction
    for i=1:NValidation
        [~,yVaPred]=max((ycPredValidation'));
        [~,yv]=max((yValidation'));
    end
    CEvalidation(it)=1-sum(yVaPred-yv==0)/NValidation;

    %early stopping
    if CEvalidation(it)<MinCE
        MinCE=CEvalidation(it);
        count=0;
%         MB=B;
%         MC=C;
%         MSigma=Sigma;
%         MV=V;
    else
        count=count+1;
    end
    %% compute and visualize the confusion matrix
%     if (it==numIt)||(count>Maxcount)
    if it==numIt
         f=ones(NTest,R); % firing level of rules
        for n=1:NTest
            for m=1:M % membership grades of MFs
%                 mu(m,:)=exp(-(XTest(n,m)-MC(m,:)).^2./(2*MSigma(m,:).^2));
                mu(m,:)=exp(-(XTest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
            end

            for r=1:R % firing levels of rules
                idsMFs=idx2vec(r,numMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
%         yR=[ones(NTest,1) XTest]*MB';%计算每个规则的结果矩阵y 层4     
%         yPredTest=f.*yR./sum(f,2)*MV; 
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

% BT=MB;
% CT=MC;
% SigmaT=MSigma;
% VT=MV;
BT=B;
CT=C;
SigmaT=Sigma;
VT=V;
