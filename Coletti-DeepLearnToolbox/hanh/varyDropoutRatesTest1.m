% get a cluster handle and a job
cluster = parcluster('anthill');
job = createJob(cluster)

% set your resource requests (these two are required)
qsubargs = '-l h_rt=48:00:00 -l virtual_free=1G'  %sets max runtime to 48 hr and memory used to 1G
set(cluster, 'IndependentSubmitFcn', {@independentSubmitFcn, qsubargs});

% test dropout values
dropoutRates = 0.15:0.05:0.75;
for i = 1:length(dropoutRates)
    createTask(job,@test_dropout_rates,5,{.2,dropoutRates(i),1});
end


%submit it and check status
job.submit()
% job.wait() %will block till job is done
% job.get()  %will display job stats

% %get results
% results = job.fetchOutputs()
% results{1:4}

%clean up  (removes Job* directory and files)
% delete(job);
% clear job;