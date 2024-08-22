clc
clear
cd('G:\Downscaling\MIROC6\Volga\intervolga');
fileinfo=dir('*.nc');
fnames={fileinfo.name};
fn=fnames';

%buliding cs grid
%36-48N
   Ygridcenterstart=36.25;
for j=1:36;
      Ygridcenter(j)=Ygridcenterstart+j*0.5-0.5;
end
 %46-56E
  Xgridcenterstart=46.25;
  for k=1:64;
      Xgridcenter(k)=Xgridcenterstart+k*0.5-0.5;
  end
  
  %---------------------------------------------------
%Defination matrix of Regression
mat_R_train=zeros(36,64)
mat_R_val=zeros(36,64)
mat_R_test=zeros(36,64)
mat_R_total=zeros(36,64)

%Defination matrix of RMSE
mat_RMSE_train=zeros(36,64)
mat_RMSE_val=zeros(36,64)
mat_RMSE_test=zeros(36,64)
mat_RMSE_total=zeros(36,64)

%Defination matrix of downscaled value for diffrent scenarios
ds_historical=zeros(36,64,1380);
ds_119=zeros(36,64,1032);
ds_126=zeros(36,64,1032);
ds_245=zeros(36,64,1032);
ds_370=zeros(36,64,1032);
ds_434=zeros(36,64,1032);
ds_460=zeros(36,64,1032);
ds_585=zeros(36,64,1032);

%reading ncfile items of inputs and target in historical period
% cd('G:\Downscaling\MRI-ESM2-0\caspian\intercs');
cd('G:\Downscaling\MIROC6\Volga\intervolga');
fileinfo=dir('*.nc');
fnames={fileinfo.name};
fn=fnames';
%reading ncfile items of inputs and target in historical period

%hurs
hurs119=ncread(cell2mat(fn(1)),'hurs');
hurs119=permute(hurs119,[2,1,3]);

hurs126=ncread(cell2mat(fn(2)),'hurs');
hurs126=permute(hurs126,[2,1,3]);

hurs245=ncread(cell2mat(fn(3)),'hurs');
hurs245=permute(hurs245,[2,1,3]);

hurs370=ncread(cell2mat(fn(4)),'hurs');
hurs370=permute(hurs370,[2,1,3]);

hurs434=ncread(cell2mat(fn(5)),'hurs');
hurs434=permute(hurs434,[2,1,3]);

hurs460=ncread(cell2mat(fn(6)),'hurs');
hurs460=permute(hurs460,[2,1,3]);

hurs585=ncread(cell2mat(fn(7)),'hurs');
hurs585=permute(hurs585,[2,1,3]);

hurs=ncread(cell2mat(fn(8)),'hurs');
hurs=hurs(:,:,601:end);
hurs=permute(hurs,[2,1,3]);

%pr

pr119=ncread(cell2mat(fn(9)),'pr');
pr119=permute(pr119,[2,1,3]);
pr119=pr119.*(2592000);
 
pr126=ncread(cell2mat(fn(10)),'pr');
pr126=permute(pr126,[2,1,3]);
pr126=pr126.*(2592000);

pr245=ncread(cell2mat(fn(11)),'pr');
pr245=permute(pr245,[2,1,3]);
pr245=pr245.*(2592000);

pr370=ncread(cell2mat(fn(12)),'pr');
pr370=permute(pr370,[2,1,3]);
pr370=pr370.*(2592000);

pr434=ncread(cell2mat(fn(13)),'pr');
pr434=permute(pr434,[2,1,3]);
pr434=pr434.*(2592000);

pr460=ncread(cell2mat(fn(14)),'pr');
pr460=permute(pr460,[2,1,3]);
pr460=pr460.*(2592000);

pr585=ncread(cell2mat(fn(15)),'pr');
pr585=permute(pr585,[2,1,3]);
pr585=pr585.*(2592000);

pr=ncread(cell2mat(fn(16)),'pr');
pr=pr(:,:,601:end);
pr=permute(pr,[2,1,3]);
pr=pr.*(2592000);

%psl
psl119=ncread(cell2mat(fn(17)),'psl');
psl119=permute(psl119,[2,1,3]);

psl126=ncread(cell2mat(fn(18)),'psl');
psl126=permute(psl126,[2,1,3]);

psl245=ncread(cell2mat(fn(19)),'psl');
psl245=permute(psl245,[2,1,3]);

psl370=ncread(cell2mat(fn(20)),'psl');
psl370=permute(psl370,[2,1,3]);

psl434=ncread(cell2mat(fn(21)),'psl');
psl434=permute(psl434,[2,1,3]);

psl460=ncread(cell2mat(fn(22)),'psl');
psl460=permute(psl460,[2,1,3]);

psl585=ncread(cell2mat(fn(23)),'psl');
psl585=permute(psl585,[2,1,3]);

psl=ncread(cell2mat(fn(24)),'psl');
psl=psl(:,:,601:end);
psl=permute(psl,[2,1,3]);

%tas
tas119=ncread(cell2mat(fn(25)),'tas');
tas119=permute(tas119,[2,1,3]);
tas119=bsxfun(@plus,tas119, -273.15);

tas126=ncread(cell2mat(fn(26)),'tas');
tas126=permute(tas126,[2,1,3]);
tas126=bsxfun(@plus,tas126, -273.15);
 
tas245=ncread(cell2mat(fn(27)),'tas');
tas245=permute(tas245,[2,1,3]);
tas245=bsxfun(@plus,tas245, -273.15);

tas370=ncread(cell2mat(fn(28)),'tas');
tas370=permute(tas370,[2,1,3]);
tas370=bsxfun(@plus,tas370, -273.15);

tas434=ncread(cell2mat(fn(29)),'tas');
tas434=permute(tas434,[2,1,3]);
tas434=bsxfun(@plus,tas434, -273.15);

tas460=ncread(cell2mat(fn(30)),'tas');
tas460=permute(tas460,[2,1,3]);
tas460=bsxfun(@plus,tas460, -273.15);

tas585=ncread(cell2mat(fn(31)),'tas');
tas585=permute(tas585,[2,1,3]);
tas585=bsxfun(@plus,tas585, -273.15);

tas=ncread(cell2mat(fn(32)),'tas');
tas=tas(:,:,601:end);
tas=permute(tas,[2,1,3]);
tas=bsxfun(@plus,tas, -273.15);

%tasmax
tasmax119=ncread(cell2mat(fn(33)),'tasmax');
tasmax119=permute(tasmax119,[2,1,3]);
tasmax119=bsxfun(@plus,tasmax119, -273.15);

tasmax126=ncread(cell2mat(fn(34)),'tasmax');
tasmax126=permute(tasmax126,[2,1,3]);
tasmax126=bsxfun(@plus,tasmax126, -273.15);
 
tasmax245=ncread(cell2mat(fn(35)),'tasmax');
tasmax245=permute(tasmax245,[2,1,3]);
tasmax245=bsxfun(@plus,tasmax245, -273.15);
 
tasmax370=ncread(cell2mat(fn(36)),'tasmax');
tasmax370=permute(tasmax370,[2,1,3]);
tasmax370=bsxfun(@plus,tasmax370, -273.15);

tasmax434=ncread(cell2mat(fn(37)),'tasmax');
tasmax434=permute(tasmax434,[2,1,3]);
tasmax434=bsxfun(@plus,tasmax434, -273.15);

tasmax460=ncread(cell2mat(fn(38)),'tasmax');
tasmax460=permute(tasmax460,[2,1,3]);
tasmax460=bsxfun(@plus,tasmax460, -273.15);

tasmax585=ncread(cell2mat(fn(39)),'tasmax');
tasmax585=permute(tasmax585,[2,1,3]);
tasmax585=bsxfun(@plus,tasmax585, -273.15);

tasmax=ncread(cell2mat(fn(40)),'tasmax');
tasmax=tasmax(:,:,601:end);
tasmax=permute(tasmax,[2,1,3]);
tasmax=bsxfun(@plus,tasmax, -273.15);

%tasmin
tasmin119=ncread(cell2mat(fn(41)),'tasmin');
tasmin119=permute(tasmin119,[2,1,3]);
tasmin119=bsxfun(@plus,tasmin119, -273.15);

tasmin126=ncread(cell2mat(fn(42)),'tasmin');
tasmin126=permute(tasmin126,[2,1,3]);
tasmin126=bsxfun(@plus,tasmin126, -273.15);
 
tasmin245=ncread(cell2mat(fn(43)),'tasmin');
tasmin245=permute(tasmin245,[2,1,3]);
tasmin245=bsxfun(@plus,tasmin245, -273.15);
 
tasmin370=ncread(cell2mat(fn(44)),'tasmin');
tasmin370=permute(tasmin370,[2,1,3]);
tasmin370=bsxfun(@plus,tasmin370, -273.15);

tasmin434=ncread(cell2mat(fn(45)),'tasmin');
tasmin434=permute(tasmin434,[2,1,3]);
tasmin434=bsxfun(@plus,tasmin434, -273.15);

tasmin460=ncread(cell2mat(fn(46)),'tasmin');
tasmin460=permute(tasmin460,[2,1,3]);
tasmin460=bsxfun(@plus,tasmin460, -273.15);

tasmin585=ncread(cell2mat(fn(47)),'tasmin');
tasmin585=permute(tasmin585,[2,1,3]);
tasmin585=bsxfun(@plus,tasmin585, -273.15);

tasmin=ncread(cell2mat(fn(48)),'tasmin');
tasmin=tasmin(:,:,601:end);
tasmin=permute(tasmin,[2,1,3]);
tasmin=bsxfun(@plus,tasmin, -273.15);

%sfcwind
sfcwind119=ncread(cell2mat(fn(49)),'sfcWind');
sfcwind119=permute(sfcwind119,[2,1,3]);

sfcwind126=ncread(cell2mat(fn(50)),'sfcWind');
sfcwind126=permute(sfcwind126,[2,1,3]);

sfcwind245=ncread(cell2mat(fn(51)),'sfcWind');
sfcwind245=permute(sfcwind245,[2,1,3]);
 
sfcwind370=ncread(cell2mat(fn(52)),'sfcWind');
sfcwind370=permute(sfcwind370,[2,1,3]);

sfcwind434=ncread(cell2mat(fn(53)),'sfcWind');
sfcwind434=permute(sfcwind434,[2,1,3]);

sfcwind460=ncread(cell2mat(fn(54)),'sfcWind');
sfcwind460=permute(sfcwind460,[2,1,3]);

sfcwind585=ncread(cell2mat(fn(55)),'sfcWind');
sfcwind585=permute(sfcwind585,[2,1,3]);

sfcwind=ncread(cell2mat(fn(56)),'sfcWind');
sfcwind=sfcwind(:,:,601:end);
sfcwind=permute(sfcwind,[2,1,3]);

%zg
% zg119=ncread(cell2mat(fn(57)),'zg');
% zg1000_119=squeeze(zg119(:,:,1,:));
% zg1000_119=permute(zg1000_119,[2,1,3]);
% zg850_119=squeeze(zg119(:,:,3,:));
% zg850_119=permute(zg850_119,[2,1,3]);
% zg500_119=squeeze(zg119(:,:,6,:));
% zg500_119=permute(zg500_119,[2,1,3]);
% zg200_119=squeeze(zg119(:,:,10,:));
% zg200_119=permute(zg200_119,[2,1,3]);
% zg50_119=squeeze(zg119(:,:,14,:));
% zg50_119=permute(zg50_119,[2,1,3]);

zg126=ncread(cell2mat(fn(57)),'zg');
zg1000_126=squeeze(zg126(:,:,1,:));
zg1000_126=permute(zg1000_126,[2,1,3]);
zg850_126=squeeze(zg126(:,:,3,:));
zg850_126=permute(zg850_126,[2,1,3]);
zg500_126=squeeze(zg126(:,:,6,:));
zg500_126=permute(zg500_126,[2,1,3]);
zg200_126=squeeze(zg126(:,:,10,:));
zg200_126=permute(zg200_126,[2,1,3]);
zg50_126=squeeze(zg126(:,:,14,:));
zg50_126=permute(zg50_126,[2,1,3]);
 
% zg245=ncread(cell2mat(fn(59)),'zg');
% zg1000_245=squeeze(zg245(:,:,1,:));
% zg1000_245=permute(zg1000_245,[2,1,3]);
% zg850_245=squeeze(zg245(:,:,3,:));
% zg850_245=permute(zg850_245,[2,1,3]);
% zg500_245=squeeze(zg245(:,:,6,:));
% zg500_245=permute(zg500_245,[2,1,3]);
% zg200_245=squeeze(zg245(:,:,10,:));
% zg200_245=permute(zg200_245,[2,1,3]);
% zg50_245=squeeze(zg245(:,:,14,:));
% zg50_245=permute(zg50_245,[2,1,3]);
 
% zg370=ncread(cell2mat(fn(60)),'zg');
% zg1000_370=squeeze(zg370(:,:,1,:));
% zg1000_370=permute(zg1000_370,[2,1,3]);
% zg850_370=squeeze(zg370(:,:,3,:));
% zg850_370=permute(zg850_370,[2,1,3]);
% zg500_370=squeeze(zg370(:,:,6,:));
% zg500_370=permute(zg500_370,[2,1,3]);
% zg200_370=squeeze(zg370(:,:,10,:));
% zg200_370=permute(zg200_370,[2,1,3]);
% zg50_370=squeeze(zg370(:,:,14,:));
% zg50_370=permute(zg50_370,[2,1,3]);

% zg434=ncread(cell2mat(fn(61)),'zg');
% zg1000_434=squeeze(zg434(:,:,1,:));
% zg1000_434=permute(zg1000_434,[2,1,3]);
% zg850_434=squeeze(zg434(:,:,3,:));
% zg850_434=permute(zg850_434,[2,1,3]);
% zg500_434=squeeze(zg434(:,:,6,:));
% zg500_434=permute(zg500_434,[2,1,3]);
% zg200_434=squeeze(zg434(:,:,10,:));
% zg200_434=permute(zg200_434,[2,1,3]);
% zg50_434=squeeze(zg434(:,:,14,:));
% zg50_434=permute(zg50_434,[2,1,3]);

% zg460=ncread(cell2mat(fn(62)),'zg');
% zg1000_460=squeeze(zg460(:,:,1,:));
% zg1000_460=permute(zg1000_460,[2,1,3]);
% zg850_460=squeeze(zg460(:,:,3,:));
% zg850_460=permute(zg850_460,[2,1,3]);
% zg500_460=squeeze(zg460(:,:,6,:));
% zg500_460=permute(zg500_460,[2,1,3]);
% zg200_460=squeeze(zg460(:,:,10,:));
% zg200_460=permute(zg200_460,[2,1,3]);
% zg50_460=squeeze(zg460(:,:,14,:));
% zg50_460=permute(zg50_460,[2,1,3]);

zg585=ncread(cell2mat(fn(58)),'zg');
zg1000_585=squeeze(zg585(:,:,1,:));
zg1000_585=permute(zg1000_585,[2,1,3]);
zg850_585=squeeze(zg585(:,:,3,:));
zg850_585=permute(zg850_585,[2,1,3]);
zg500_585=squeeze(zg585(:,:,6,:));
zg500_585=permute(zg500_585,[2,1,3]);
zg200_585=squeeze(zg585(:,:,10,:));
zg200_585=permute(zg200_585,[2,1,3]);
zg50_585=squeeze(zg585(:,:,14,:));
zg50_585=permute(zg50_585,[2,1,3]);

zg=ncread(cell2mat(fn(59)),'zg');
zg=zg(:,:,:,601:end);
zg1000=squeeze(zg(:,:,1,:));
zg1000=permute(zg1000,[2,1,3]);
zg850=squeeze(zg(:,:,3,:));
zg850=permute(zg850,[2,1,3]);
zg500=squeeze(zg(:,:,6,:));
zg500=permute(zg500,[2,1,3]);
zg200=squeeze(zg(:,:,10,:));
zg200=permute(zg200,[2,1,3]);
zg50=squeeze(zg(:,:,14,:));
zg50=permute(zg50,[2,1,3]);

cd('G:\PhD_2\PhD\GHCN');
filename='air.mon.mean.v501-croped-1900-2014-volga.nc';
air_ghcn_CS=ncread(filename,'air');
air_ghcn_CS=permute(air_ghcn_CS,[2,1,3]);

% cd('E:\PhD\GHCN');
cd('G:\PhD_2\PhD\GHCN');
filename='precip.mon.total.v501-volga2-1900-2014.nc';
precip_ghcn_cs=ncread(filename,'precip');
precip_ghcn_cs=permute(precip_ghcn_cs,[2,1,3]);
precip_ghcn_cs=precip_ghcn_cs*10;

%test correlatio for gounding predictors
for j=1:36
    for k=1:64
R=corrcoef(tas(j,k,:),precip_ghcn_cs(j,k,:));
R_tas(j,k)=R(1,2);
R=corrcoef(tasmin(j,k,:),precip_ghcn_cs(j,k,:));
R_tasmin(j,k)=R(1,2);
R=corrcoef(tasmax(j,k,:),precip_ghcn_cs(j,k,:));
R_tasmax(j,k)=R(1,2);
R=corrcoef(hurs(j,k,:),precip_ghcn_cs(j,k,:));
R_hurs(j,k)=R(1,2);
R=corrcoef(pr(j,k,:),precip_ghcn_cs(j,k,:));
R_pr(j,k)=R(1,2);
R=corrcoef(psl(j,k,:),precip_ghcn_cs(j,k,:));
R_psl(j,k)=R(1,2);
R=corrcoef(sfcwind(j,k,:),precip_ghcn_cs(j,k,:));
R_sfcwind(j,k)=R(1,2);
R=corrcoef(zg1000(j,k,:),precip_ghcn_cs(j,k,:));
R_zg1000(j,k)=R(1,2);
R=corrcoef(zg850(j,k,:),precip_ghcn_cs(j,k,:));
R_zg850(j,k)=R(1,2);
R=corrcoef(zg500(j,k,:),precip_ghcn_cs(j,k,:));
R_zg500(j,k)=R(1,2);
R=corrcoef(zg200(j,k,:),precip_ghcn_cs(j,k,:));
R_zg200(j,k)=R(1,2);
R=corrcoef(zg50(j,k,:),precip_ghcn_cs(j,k,:));
R_zg50(j,k)=R(1,2);
    end
end

R_mean_tas=nanmean(mean(R_tas));
R_mean_tasmin=nanmean(mean(R_tasmin));
R_mean_tasmax=nanmean(mean(R_tasmax));
R_mean_hurs=nanmean(mean(R_hurs));
R_mean_pr=nanmean(mean(R_pr));
R_mean_psl=nanmean(mean(R_psl));
R_mean_sfcwind=nanmean(mean(R_sfcwind));
R_mean_zg1000=nanmean(mean(R_zg1000));
R_mean_zg850=nanmean(mean(R_zg850));
R_mean_zg500=nanmean(mean(R_zg500));
R_mean_zg200=nanmean(mean(R_zg200));
R_mean_zg50=nanmean(mean(R_zg50));



% parfor j=1:36;
parfor j=1:36;
     for k=1:64;
    input_precip_cs=squeeze([tas(j,k,:),pr(j,k,:)]);   
    target_precip_cs=squeeze(precip_ghcn_cs(j,k,:));
       
    %%%%%%%%%%
    %reading scenarios for prediction
% 
    input_precip_cs_119=squeeze([tas119(j,k,:),pr119(j,k,:)]);
    input_precip_cs_119=input_precip_cs_119';
    
    input_precip_cs_126=squeeze([tas126(j,k,:),pr126(j,k,:)]);
    input_precip_cs_126=input_precip_cs_126';

    input_precip_cs_245=squeeze([tas245(j,k,:),pr245(j,k,:)]);
    input_precip_cs_245=input_precip_cs_245';
%     
    input_precip_cs_370=squeeze([tas370(j,k,:),pr370(j,k,:)]);
    input_precip_cs_370=input_precip_cs_370';
    
    input_precip_cs_434=squeeze([tas434(j,k,:),pr434(j,k,:)]);
    input_precip_cs_434=input_precip_cs_434';

    input_precip_cs_460=squeeze([tas460(j,k,:),pr460(j,k,:)]);
    input_precip_cs_460=input_precip_cs_460';


    input_precip_cs_585=squeeze([tas585(j,k,:),pr585(j,k,:)]);
    input_precip_cs_585=input_precip_cs_585';
    
    %%%%%%%%%%

inputs =input_precip_cs;
targets = target_precip_cs';

% Create a Fitting Network
hiddenLayerSize = 3;
net = fitnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)
Cyt=corrcoef(outputs,targets)
R=Cyt(2,1)
ds_historical(j,k,:)=outputs;
% Recalculate Training, Validation and Test Performance
trainTargets = targets .* tr.trainMask{1};
valTargets = targets  .* tr.valMask{1};
testTargets = targets  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs)
valPerformance = perform(net,valTargets,outputs)
testPerformance = perform(net,testTargets,outputs)

%RMSE
RMSE_train=sqrt(trainPerformance)
RMSE_val=sqrt(valPerformance)
RMSE_test=sqrt(testPerformance)
RMSE_total=sqrt(performance)

%Regression
R_train=regression(trainTargets,outputs)
R_val=regression(valTargets,outputs)
R_test=regression(valTargets,outputs)
R_total=regression(targets,outputs)

%filling the matrix
mat_R_train(j,k)=R_train
mat_R_val(j,k)=R_val
mat_R_test(j,k)=R_test
mat_R_total(j,k)=R_total

mat_RMSE_train(j,k)=RMSE_train
mat_RMSE_val(j,k)=RMSE_val
mat_RMSE_test(j,k)=RMSE_test
mat_RMSE_total(j,k)=RMSE_total

% DS=downscaled
ds_historical(j,k,:)=outputs;
ds_119(j,k,:)=net(input_precip_cs_119');
ds_126(j,k,:)=net(input_precip_cs_126');
ds_245(j,k,:)=net(input_precip_cs_245');
ds_370(j,k,:)=net(input_precip_cs_370');
ds_434(j,k,:)=net(input_precip_cs_434');
ds_460(j,k,:)=net(input_precip_cs_460');
ds_585(j,k,:)=net(input_precip_cs_585');
  end
end

%%drawing performance
%buliding cs grid
%36-48N
   Ygridcenterstart=45.25;
for j=1:36;
      Ygridcenter(j)=Ygridcenterstart+j*0.5-0.5;
end
 %46-56E
  Xgridcenterstart=30.25;
  for k=1:64;
      Xgridcenter(k)=Xgridcenterstart+k*0.5-0.5;
  end
  
 [X,Y] = meshgrid(Xgridcenter,Ygridcenter);

  S = shaperead('G:\PhD_2\PhD\shapefile-volga\volgashapefile\volga.shp');
  S2=shaperead('G:\PhD_2\PhD\shapefile-volga\volgashapefile\volgapolyline.shp');
  in = inpolygon(X,Y,S.X,S.Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Correlation figure % Regression matrix
%Raw data correlation (pr)
 for j=1:36
    for k=1:64
R=corrcoef(pr(j,k,:),precip_ghcn_cs(j,k,:));
R_pr(j,k)=R(1,2);
    end
end

BB5=zeros(36,64);
BB_new=R_pr
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k)
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

 figure(1)
 subplot(3,2,1);
 contourf(XX,YY,BB_new,'LineColor','none');
 axis equal;
 colorbar;
 caxis([0 0.8]);
 xlabel('Raw pr Correltion');
 a1=max(max(BB5));
 b1=min(min(BB5));
 mapshow(S2, 'color', 'black');
 hold on
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 %R_total
  BB4=mat_R_total;
  BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(1)
subplot(3,2,2);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
a2=max(max(BB5));
b2=min(min(BB5));
% caxis([b4 a4]);
caxis([0 0.8]);
xlabel('Total Correltion');
mapshow(S2, 'color', 'black');
hold on  
% maxaxis=[a1,a2,a3,a4];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%R_train
BB4=mat_R_train;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(1)
subplot(3,2,3);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
a3=max(max(BB5));
b3=min(min(BB5));
% caxis([0.1 0.4]);
% caxis([0 0.35]);
% caxis([b1 a1]);
caxis([0 0.8]);
xlabel('Train Correltion');
mapshow(S2, 'color', 'black');
hold on  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%R_val
BB4=mat_R_val;
  BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(1)
subplot(3,2,4);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
a4=max(max(BB5));
b4=min(min(BB5));
% caxis([0.95 1]);
% caxis([0 0.35]);
% caxis([b2 a2]);
caxis([0 0.8]);
xlabel('Validation Correltion');
mapshow(S2, 'color', 'black');
hold on  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%R_test
BB4=mat_R_test;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(1)
subplot(3,2,5);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
% caxis([0.95 1]);
% caxis([0 0.35]);
a5=max(max(BB5));
b5=min(min(BB5));
% caxis([b3 a3]);
caxis([0 0.8]);
xlabel('Test Correltion');
mapshow(S2, 'color', 'black');
hold on  
set(gcf,'color','w')
range_R=[b1,a1;b2,a2;b3,a3;b4,a4;b5,a5];
% a=max(maxaxis);
% minaxis=[b1,b2,b3,b4];
% b=min(minaxis);
% caxis([b a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RMSE figure
%Raw data RMSE
rmse=zeros(36,64);
for j=1:36
    for k=1:64
obs=squeeze(precip_ghcn_cs(j,k,:));
mo=squeeze(pr(j,k,:));
difsquare=(obs-mo).^2;
sum=0
for i=1:1380
  sum=sum+difsquare(i);
end 
sum=sum./1380;
rmse(j,k)=sqrt(sum);
    end
end

BB4=rmse;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(2)
subplot(3,2,1);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a6=max(max(BB5));
b6=min(min(BB5));
xlabel('Raw pr RMSE');
mapshow(S2, 'color', 'black');
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Total RMSE
BB4=mat_RMSE_total;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(2)
subplot(3,2,2);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
a7=max(max(BB5));
b7=min(min(BB5));
% caxis([b8 a8])
caxis([15 45]);
xlabel('Total RMSE');
mapshow(S2, 'color', 'black');
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RMSE_Train
figure(2)
BB4=mat_RMSE_train;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

subplot(3,2,3);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
a8=max(max(BB5));
b8=min(min(BB5));
% caxis([b5 a5]);
caxis([15 45]);
xlabel('Train RMSE');
mapshow(S2, 'color', 'black');
hold on  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RMSE_val
BB4=mat_RMSE_val;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(2)
subplot(3,2,4);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
% caxis([0 25]);
a9=max(max(BB5));
b9=min(min(BB5));
% caxis([b6 a6]);
caxis([15 45]);
xlabel('Validation RMSE');
mapshow(S2, 'color', 'black');
hold on  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RMSE_test
BB4=mat_RMSE_test;
BB_new=zeros(36,64);
  for i=1:2:35 %(24-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(20-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end

figure(2)
subplot(3,2,5);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
% caxis([0 25]);
a10=max(max(BB5));
b10=min(min(BB5));
% caxis([19 26]);
caxis([15 45]);
xlabel('Test RMSE');
mapshow(S2, 'color', 'black');
hold on 
set(gcf,'color','w')
range_RMSE=[b6,a6;b7,a7;b8,a8;b9,a9;b10,a10];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RAw data versus donscasled 
%Raw data correlation (pr)
  for j=1:36
    for k=1:64
R=corrcoef(pr(j,k,:),precip_ghcn_cs(j,k,:));
R_pr(j,k)=R(1,2);
    end
end
 
BB5=zeros(36,64);
BB_new=R_pr
parfor i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k)
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_R_raw=BB5;
 figure(3)
 subplot(2,4,1);
 contourf(XX,YY,BB_new,'LineColor','none');
 axis equal;
 colorbar;
 caxis([0 0.8]);
 xlabel('Raw Precipitation CC');
 a11=max(max(BB5));
 b11=min(min(BB5)); 
 mapshow(S2, 'color', 'black');
 
%%
rmse=zeros(36,64);
for j=1:36
    for k=1:64
obs=squeeze(precip_ghcn_cs(j,k,:));
mo=squeeze(pr(j,k,:));
difsquare=(obs-mo).^2;
sum=0
for i=1:1380
  sum=sum+difsquare(i);
end 
sum=sum./1380;
rmse(j,k)=sqrt(sum);
    end
end
BB4=rmse;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_RMSE_raw=BB5;
figure(3)
subplot(2,4,2);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a12=max(max(BB5));
b12=min(min(BB5));
xlabel('Raw Precipitation RMSE');
mapshow(S2, 'color', 'black');
hold on
%%
bias=zeros(36,64);
MAE=zeros(36,64);
for j=1:36
    for k=1:64
obs=squeeze(precip_ghcn_cs(j,k,:));
mo=squeeze(pr(j,k,:));
erorr=(obs-mo);
erorr2=abs(erorr);
sum=0
sum2=0
for i=1:1380
  sum=sum+erorr(i);
  sum2=sum2+erorr2(i);
end 
sum=sum./1380;
sum2=sum2./1380;
bias(j,k)=sum2;
MAE(j,k)=sum2;
    end
end
BB4=bias;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_bias_raw=BB5;
figure(3)
subplot(2,4,3);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a13=max(max(BB5));
b13=min(min(BB5));
xlabel('Raw Precipitation Bias');
mapshow(S2, 'color', 'black');
hold on
 
 
BB4=MAE;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_MAE_raw=BB5
figure(3)
subplot(2,4,4);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a14=max(max(BB5));
b14=min(min(BB5));
xlabel('Raw Precipitation MAE');
mapshow(S2, 'color', 'black');
hold on
set(gcf,'color','w');
range_Raw=[b11,a11;b12,a12;b13,a13;b14,a14];
 
%%%%%%%%%%%%%%%%%%%
 
%downslaed data correlation (pr)
  for j=1:36
    for k=1:64
R=corrcoef(ds_historical(j,k,:),precip_ghcn_cs(j,k,:));
R_pr(j,k)=R(1,2);
    end
end
 
BB5=zeros(36,64);
BB_new=R_pr
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k)
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_R_downscaled=BB5;
 figure(3)
 subplot(2,4,5);
 contourf(XX,YY,BB_new,'LineColor','none');
 axis equal;
 colorbar;
 caxis([0 0.8]);
 xlabel('Downscaled Precipitation CC');
 a15=max(max(BB5));
 b15=min(min(BB5)); 
 mapshow(S2, 'color', 'black');
 
%%
rmse=zeros(36,64);
for j=1:36
    for k=1:64
obs=squeeze(precip_ghcn_cs(j,k,:));
mo=squeeze(ds_historical(j,k,:));
difsquare=(obs-mo).^2;
sum=0
for i=1:1380
  sum=sum+difsquare(i);
end 
sum=sum./1380;
rmse(j,k)=sqrt(sum);
    end
end
BB4=rmse;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_RMSE_downscaled=BB5;
figure(3)
subplot(2,4,6);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a16=max(max(BB5));
b16=min(min(BB5));
xlabel('Downscaled Precipitation RMSE');
mapshow(S2, 'color', 'black');
hold on
%%
bias=zeros(36,64);
MAE=zeros(36,64);
for j=1:36
    for k=1:64
obs=squeeze(precip_ghcn_cs(j,k,:));
mo=squeeze(ds_historical(j,k,:));
erorr=(obs-mo);
erorr2=abs(erorr);
sum=0
sum2=0
for i=1:1380
  sum=sum+erorr(i);
  sum2=sum2+erorr2(i);
end 
sum=sum./1380;
sum2=sum2./1380;
bias(j,k)=sum2;
MAE(j,k)=sum2;
    end
end
BB4=bias;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_Bias_downscaled=BB5;
figure(3)
subplot(2,4,7);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a17=max(max(BB5));
b17=min(min(BB5));
xlabel('Downscaled Precipitation Bias');
mapshow(S2, 'color', 'black');
hold on
 
BB4=MAE;
BB_new=zeros(36,64);
  for i=1:2:35 %(36-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
      for k=1:2:63  %(64-1) -1 chon dar har statementyeki jolotaresh ham hesab mishe
          ii=floor(i/4)+1;
          kk=floor(k/4)+1;
          BB_new(i,k)= BB4(ii,kk);
          BB_new(i,k+1)= BB4(ii,kk);
          BB_new(i+1,k)= BB4(ii,kk);
          BB_new(i+1,k+1)= BB4(ii,kk);
      end
  end
  
XX=zeros(36,64);
YY=zeros(36,64);
BB5=zeros(36,64);
for i=1:36
    for k=1:64
        if in(i,k)==0
            XX(i,k)=nan
            YY(i,k)=nan
            BB5(i,k)=nan
        else
            XX(i,k)=X(i,k) 
            YY(i,k)=Y(i,k)
            BB5(i,k)=BB_new(i,k)
        end
    end
end
mat_MAE_downscaled=BB5;
figure(3)
subplot(2,4,8);
contourf(XX,YY,BB_new,'LineColor','none');
axis equal;
colorbar;
caxis([15 45]);
a18=max(max(BB5));
b18=min(min(BB5));
xlabel('Downscaled Precipitation MAE');
mapshow(S2, 'color', 'black');
hold on
set(gcf,'color','w');
range_dowmscaled=[b15,a15;b16,a16;b17,a17;b18,a18];

clearvars -except ds_119 ds_126 ds_245 ds_370 ds_434 ds_460 ds_585 ssp119_p mat_R_train mat_R_val mat_R_test mat_R_total mat_RMSE_train mat_RMSE_val mat_RMSE_test mat_RMSE_total ds_historical range_R range_RMSE range_dowmscaled range_Raw mat_Bias_downscaled mat_MAE_downscaled mat_R_downscaled mat_RMSE_downscaled mat_bias_raw mat_MAE_raw mat_R_raw mat_RMSE_raw   
cd('G:\Downscaling\MIROC6\Volga\Precipitaion');
save('MIROC6-volga-precip');

