% get a cluster handle and a job
cluster = parcluster('anthill');
job = createJob(cluster);

% set your resource requests (these two are required)
qsubargs = '-l h_rt=96:00:00 -l virtual_free=1G';  %sets max runtime to 48 hr and memory used to 1G
set(cluster, 'IndependentSubmitFcn', {@independentSubmitFcn, qsubargs});

% test dropout values
modelRange = 1:5;
inputCorruptFraction = 0.2;
dropoutRate = 0.5;
noise = 'salt_pepper';
numepochs = 3000;

%testing with tanh
activation = 'tanh_opt';
for modelnum = modelRange;
    createTask(job,@test_dropout,5,{noise, inputCorruptFraction, dropoutRate, activation, numepochs, modelnum});
end

%testing with relu
% activation = 'relu';
% for modelnum = modelRange;
%     createTask(job,@test_connect,5,{noise, inputCorruptFraction, dropoutRate, activation, modelnum});
% end

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