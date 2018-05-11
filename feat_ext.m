function feat_ext(arg)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% DIR.dataset = 'K:/ISPRS_semantic_labeling_Vaihingen/';
DIR.dataset = '../ISPRS_semantic_labeling_Vaihingen/';
DIR.Ftop = [DIR.dataset 'top_inverted/'];
DIR.Fndsm = [DIR.dataset 'ndsm/'];
DIR.Fdsm = [DIR.dataset 'dsm/'];
DIR.Fgt = [DIR.dataset 'gts_for_participants/'];
DIR.Fout = [DIR.dataset '9_feature_mat/'];
test=false;
if strcmp(arg,'train')
    DIR.file_train = [DIR.dataset 'train.txt'];
    file = dlmread(DIR.file_train);
elseif strcmp(arg,'valid')
    DIR.file_valid = [DIR.dataset 'valid.txt'];
    file = dlmread(DIR.file_valid);
elseif strcmp(arg,'test')
    test = true;
    DIR.file_test = [DIR.dataset 'test.txt'];
    file = dlmread(DIR.file_test);
else
    file = arg;
end
    
%%

%{
 LABELS = [... 
    [255, 255, 255]; 	%  Impervious surfaces
    [0, 0, 255];     	%  Building
	[0, 255, 255];		%  Low vegetation
    [0, 255, 0];     	%  Tree
    [255, 255, 0];   	%  Car
    [255, 0, 0]   		%  Clutter/background  
    ...
    ]/255; 
%}


LABELS = [... 
    [255, 255, 255]; 	%  Impervious surfaces
    [255, 0, 0];     	%  Building
	[255, 255, 0];		%  Low vegetation
    [0, 255, 0];     	%  Tree
    [0, 255, 255];   	%  Car
    [0, 0, 255]   		%  Clutter/background  
    ...
    ]/255;
%%
for n = file(:)'
    
    top = imread(sprintf('%stop_mosaic_09cm_area%d.png',DIR.Ftop,n));
    dsm = imread(sprintf('%sdsm_09cm_matching_area%d.tif',DIR.Fdsm,n));
    % if n == 11, dsm = dsm(12:end,:); end
    ndsm = imread(sprintf('%sdsm_09cm_matching_area%d_normalized.jpg',DIR.Fndsm,n));
    if(~test)
        label = imread(sprintf('%stop_mosaic_09cm_area%d.tif',DIR.Fgt,n));
        label = rgb2ind(label, LABELS);
    end
%%    
    ndvi = uint8(255*double(top(:,:,1)-top(:,:,2))./double(top(:,:,1)+top(:,:,2)));
    [~,sat,~] = rgb2hsv(top);   
    sat = uint8(255*sat);    
    %avg = uint8(mean(top,3));
    %max = max(top,[],3);        
    ma = max(dsm(:));
    mi = min(dsm(:));
    udsm = uint8(255*(dsm-mi)/(ma-mi));
    clear ma mi
    entpy = entropyfilt(udsm);
    entpy2 = entropyfilt(ndsm);
    
    [dx,dy] = imgradientxy(dsm);
    dz = ones(size(dx));
    [azi,ele,~] = cart2sph(dx,dy,dz);
    clear dx dy dz
    
    imlab = applycform(top, makecform('srgb2lab'));
    l = imlab(:,:,1);
    a = imlab(:,:,2);
    b = imlab(:,:,3);
    clear imlab 

    if(~test)
        savecsv(sprintf('%sfeat%d.csv',DIR.Fout,n),...
                         ndvi,sat,ndsm,...
                         l, a, b,...
                         azi, ele,...
                         entpy,entpy2,...
                         label);
    else
        % savecsv(sprintf('%sfeat%d.csv',DIR.Fout,n),...
        %              ndvi,sat,ndsm,...
        %              l, a, b,...
        %              azi, ele,...
        %              entpy,entpy2);
        save(sprintf('%sfeat%d.mat',DIR.Fout,n), 'ndvi', 'sat', 'l', 'a', 'b', 'entpy', 'entpy2', 'azi', 'ele');
        fprintf('saved %s\n', sprintf('%sfeat%d.mat',DIR.Fout,n));
    end
end  
    

end

