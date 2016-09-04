function [amse] =  AMSE(data, model, type, option)
    if(strcmpi(option, 'distances')),
        DX = pdist2(data.x, model.refX);
        if(model.bias == 1),
            DX = [ones(size(data.x, 1), 1) DX];
        end
        DYh = DX*model.B;
        DY = pdist2(data.y, model.refY);
        errors = (DY - DYh);
    else
        [~, errors] = test_MLM(model, data, type);
    end
    amse = mean(mean(errors.^2));