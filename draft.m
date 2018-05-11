rng(2015);
DIR.dataset = '../ISPRS_semantic_labeling_Vaihingen/';
DIR.Ftop = [DIR.dataset 'top_inverted/'];
DIR.file_train = [DIR.dataset 'test.txt'];
file = dlmread(DIR.file_train);
fea = {};
f = [];
%%

for n = file(:)'
    top = imread(sprintf('%stop_mosaic_09cm_area%d.png',DIR.Ftop,n));
    fea{n} = filterbank17d(top);
    i = randi(size(fea{n},1),150000,1);
    j = unique(i);
    f = cat(1, f, fea{n}(j,:));
end

%%
tic
%     rng(7777777);
    vl_threads(0);
    [centers, ~, ~] = vl_kmeans(f', ...
                            50, 'Verbose', ...
                            'Distance', 'l2', ...
                            'MaxNumIterations', 200, ...
                            'Algorithm', 'Lloyd') ;
save kmeans_centers centers
toc
% %%
% tic
% [~,centers] = kmeans(f,50,'MaxIter',100);
% save kmean_center centers
% toc
%%
DIR.Fout = [DIR.dataset 'texton_mat/'];
Mdl = KDTreeSearcher(centers');
%%
file = dlmread(DIR.file_train);
for n = file(:)'
    texton = uint8(knnsearch(Mdl,fea{n},'K',1));
    % savecsv(sprintf('%stexton%d.csv', DIR.Fout, n), texton);
    save(sprintf('%stexton%d.mat', DIR.Fout, n), 'texton');
    fprintf('saved %s\n', sprintf('%stexton%d.mat', DIR.Fout, n));
end
%%

%{
 DIR.file_valid = [DIR.dataset 'valid.txt'];
file = dlmread(DIR.file_valid);
for n = file(:)'
    top = imread(sprintf('%stop_mosaic_09cm_area%d.tif',DIR.Ftop,n));
    fea{n} = filterbank17d(top);
    texton = uint8(knnsearch(Mdl,fea{n},'K',1));
    savecsv(sprintf('%stexton%d.csv',DIR.Fout,n),...
                     texton);
end
%}
