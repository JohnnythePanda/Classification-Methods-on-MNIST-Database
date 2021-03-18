function [U,S,V,threshold,w,sortData1,sortData2] = dc_trainer(data1,data2,feature)
    
    nd = size(data1,2);
    nc = size(data2,2);
    [U,S,V] = svd([data1 data2],'econ'); 
    
    
    total = S*V';
    U = U(:,1:feature); % Add this in
    data1 = total(1:feature,1:nd);
    data2 = total(1:feature,nd+1:nd+nc);
    md = mean(data1,2);
    mc = mean(data2,2);

    Sw = 0;
    for k=1:nd
        Sw = Sw + (data1(:,k)-md)*(data1(:,k)-md)';
    end
    for k=1:nc
        Sw = Sw + (data2(:,k)-mc)*(data2(:,k)-mc)';
    end
    Sb = (md-mc)*(md-mc)';
    
    [V2,D] = eig(Sb,Sw);
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    vData1 = w'*data1;
    vData2 = w'*data2;
    
    if mean(vData1)>mean(vData2)
        w = -w;
        vData1 = -vData1;
        vData2 = -vData2;
    end
    
    % Don't need plotting here
    sortData1 = sort(vData1);
    sortData2 = sort(vData2);
    t1 = length(sortData1);
    t2 = 1;
    while sortData1(t1)>sortData2(t2)
    t1 = t1-1;
    t2 = t2+1;
    end
    threshold = (sortData1(t1)+sortData2(t2))/2;

    % We don't need to plot results
end

