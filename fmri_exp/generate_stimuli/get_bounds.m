function bounds = get_bounds(start, stop, method, sr, len_source)

start_smp = round(start * (sr/1000))+1; % add 1 because MATLAB?
stop_smp = round(stop * (sr/1000));


switch method
    case 1 % Use start 
        bounds = [start_smp len_source];
    case 2 % Use midbound
        halfway = (start_smp(2:end)-stop_smp(1:end-1))/2;
        bounds = [1 (stop_smp(1:end-1) + halfway) len_source];
    case 3 % Use stop
        bounds = [1 stop_smp];
end
bounds = round(bounds);