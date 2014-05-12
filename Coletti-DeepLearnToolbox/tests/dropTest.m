addpath('../NN/Autoencoder_Code/');

% get a cluster handle and a job
% cluster = parcluster('anthill');
% job = createJob(cluster);

% set your resource requests (these two are required)
% qsubargs = '-l h_rt=96:00:00 -l virtual_free=1G';  %sets max runtime to 48 hr and memory used to 1G
% set(cluster, 'IndependentSubmitFcn', {@independentSubmitFcn, qsubargs});

% test dropout values
modelRange = 1:5;
inputCorruptFraction = 0.2;
rates = [0, 0.25, 0.5, 0.75];
noises = {'drop','salt_pepper','random','gaussian'};
initializations = {'pretraining'};
% numepochs = 3000;

%testing with tanh
activations = {'tanh_opt', 'sigm', 'relu'};
for activation_ind = 1:size(activations, 2)
    for noise_ind = 1:size(noises, 2)
        for dropoutRate = rates
            for initialization_ind = 1:size(initializations, 2)
                for modelnum = modelRange
                    activation = activations{activation_ind};
                    noise = noises{noise_ind};
                    initialization = initializations{initialization_ind};
%                     createTask(job,@test_connect,5,{noise, inputCorruptFraction, dropoutRate, activation, initialization, modelnum});
                    test_connect(noise, inputCorruptFraction, dropoutRate, activation, initialization, modelnum);
                end
            end
        end
    end
end

%testing with relu
% activation = 'relu';
% for modelnum = modelRange;
%     createTask(job,@test_connect,5,{noise, inputCorruptFraction, dropoutRate, activation, modelnum});
% end

%submit it and check status
% job.submit()
% job.wait() %will block till job is done
% job.get()  %will display job stats

% %get results
% results = job.fetchOutputs()
% results{1:4}

%clean up  (removes Job* directory and files)
% delete(job);
% clear job;