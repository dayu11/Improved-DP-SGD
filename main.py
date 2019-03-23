from __future__ import print_function

import sys

import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from absl import app
from absl import flags
import copy
import torch.nn.functional as F


from load_data import MNIST_data, Covertype_data
from model import ConvNet, Linear, Logistic, LinearMnist, ConvNetTest
from rdp_accountant import _compute_rdp, _compute_delta
from get_steps import get_sigma

FLAGS=flags.FLAGS
flags.DEFINE_string('dataset', 'mnist', 'which dataset to run, [mnist, covertype]')
flags.DEFINE_integer('max_steps', 15000, 'maximum training steps')
flags.DEFINE_integer('experiment_id', 0, 'used to distinguish different instance of experiment')
flags.DEFINE_float('q', 0.01, 'samping ratio q')
flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_float('momentum', .6, '0 for not using (0,1):momentum 2:Adam')
flags.DEFINE_float('clip_bound', 12, 'the clipping bound of individual gradient')
flags.DEFINE_bool('convex', True, 'convex objective or not')
flags.DEFINE_float('sigma', -1, 'deprecated')
flags.DEFINE_float('init_sigma', -1, 'initial sigma')
flags.DEFINE_float('epsilon', 2, 'desired epsilon')
flags.DEFINE_float('delta', -1, 'desired delta in (0,1) , other values mean no privately training. IF -1, will be set as 1/n**2')
flags.DEFINE_bool('cuda', True, 'use gpu or not')
flags.DEFINE_float('SGN', 2, 'using varying norm or not, 0 for not using, 1 for both decrease clip bound and sigma, 2 for decrease clip bound and fix sigma')
flags.DEFINE_integer('auto_sigma', 0, '0 for fixed sigma to run target epoched, 1 for fixed bigger sigma')
flags.DEFINE_integer('epoches', 20, 'epoches to run')


def clip_grads(clip_bound, model): #clipping individual gradient with FLAGS.clip_bound
    para_norm=0.
    for para in model.parameters():
        para_norm+=torch.sum(torch.mul(para.grad,para.grad))
    para_norm=torch.sqrt(para_norm)

    if(para_norm > clip_bound):
        for para in model.parameters():
            para.grad=torch.div(para.grad, para_norm/clip_bound)
            
    return [param.grad for param in model.parameters()]

def add_noise(sigma, grad_list): #adding noise, notice sigma=FLAGS.sigma*FLAGS.clip_bound
    cuda = FLAGS.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda and not FLAGS.convex else "cpu")
    for i, grad in enumerate(grad_list):
        mean=torch.zeros_like(grad).to(device)
        stddev=torch.add(torch.zeros_like(grad), sigma).to(device) #mean is used as zeros tensor
        grad_list[i]=torch.add(grad, torch.normal(mean,stddev))
    
    return grad_list

def logging(info, mode='a'):
    try:
        os.mkdir('logs')
    except:
        pass    

    f=open('logs/log%d.txt'%FLAGS.experiment_id, mode)
    f.write(info)
    f.close()

def sampler(tuple): #samping each index with samping probability q
    n=tuple[1].shape[0]
    rand=np.random.rand(n)
    index=(rand<=FLAGS.q)
    index=np.arange(0, n)[index]
    return tuple[0][index], tuple[1][index]

def get_norm(grads, num=1): #get L2 norm of given list
    norm=0.
    for grad in grads:
        grad=grad/num
        norm+=torch.sum(torch.mul(grad,grad))
    return np.sqrt(norm)

def vary_noise(diff):
    steps=FLAGS.epoches/FLAGS.q
    FLAGS.sigma=min(FLAGS.init_sigma+diff, FLAGS.sigma+diff/steps)
    
def vary_bound(t):
    if(FLAGS.dataset=='mnist'):
        steps= 500
    else:
        steps= 500
    ratio=min(t/steps, 1)
    return 1+ratio

def check_norm(t, model, norm_list, train_loss):
    conv_list=[]
    fc_list=[]
    for (name, _),para  in zip(model.named_parameters(), model.parameters()):
        if para.requires_grad:
            if('conv' in name):
                conv_list.append(para.grad)
            elif('fc' in name):
                fc_list.append(para.grad)
            #print(para.data)
    #print(len(conv_list))
    norm_list.append([get_norm(conv_list), get_norm(fc_list), get_norm(conv_list+fc_list), train_loss])
    #print('at %d norm of conv is : '%t, get_norm(conv_list))
    #print('at %d norm of fc is : '%t , get_norm(fc_list))   


def train(model, device, train_tuple, optimizer, diff, total_privacy_l, t, norm_list):
    model.train()
    data, target=sampler(train_tuple)
    data, target = data.to(device), target.to(device)
    
    if(FLAGS.delta>0 and FLAGS.delta<1): # we are training model privately
        train_loss=0.

        sigma=FLAGS.sigma
        clip_bound=FLAGS.clip_bound
        if(FLAGS.SGN==1):
            clip_bound=FLAGS.clip_bound/vary_bound(t)
        elif(FLAGS.SGN==2):
            v=vary_bound(t)
            clip_bound=FLAGS.clip_bound/v
            sigma=sigma*v

        for i in range(data.shape[0]): #Here we compute individual gradients sequentially. However parallel computing is achieveable.
            optimizer.zero_grad()      #See Ian Goodfellow's post in https://github.com/tensorflow/tensorflow/issues/4897
            if(FLAGS.dataset=='mnist' and not FLAGS.convex):
                output = model(data[i].reshape([1, 1, 28, 28]))
            elif(FLAGS.dataset=='mnist'):
                output = model(data[i].reshape([1, 784]))
            else:
                output = model(data[i].reshape([1,54]))
            loss = F.nll_loss(output, target[i].reshape([1]))
            train_loss+=loss/data.shape[0]
            loss.backward()

            if(i==0):
                accmulated_grad=clip_grads(clip_bound, model)  #clip grads for each set of parameter and return them
            else:
                #if(t%50==0):
                #    print(get_norm(clip_grads(FLAGS.clip_bound, model), 1))
                accmulated_grad=[x+y for x,y in zip(accmulated_grad, clip_grads(clip_bound, model))]


        #print(clip_bound, ' s: ', sigma)
        #current_norm=get_norm(accmulated_grad, data.shape[0])
        #print('norm before noise: ', current_norm)
        if(FLAGS.delta==-1):
            accmulated_grad=add_noise(2*clip_bound*sigma, accmulated_grad)#due to the different definition of differential privacy
        else:
            accmulated_grad=add_noise(clip_bound*sigma, accmulated_grad)#add noise so we can privately release gradient
        
        #print(data.shape[0])
        #print(get_norm(accmulated_grad, data.shape[0]))

        #We accumulate the rdp of each step. rdp at order t+1 is equivalent to alpha at order t.
        #See Ilya Mironov, https://arxiv.org/pdf/1702.07476.pdf The code is from
        #https://github.com/tensorflow/models/tree/master/research/differential_privacy/privacy_accountant/python
        curr_privacy_l=[_compute_rdp(FLAGS.q, sigma, order) for order in range(2, 2+128)]  
        total_privacy_l=[x+y for x, y in zip(curr_privacy_l, total_privacy_l)]

        for i, (param,grad) in enumerate(zip(model.parameters(), accmulated_grad)):  #make use of noisy gradients
            param.grad=grad/data.shape[0]

    else : # no privately training
        optimizer.zero_grad()
        output = model(data)
        train_loss = F.nll_loss(output, target)
        train_loss.backward()
        if(t%10==0):
            check_norm(t, model, norm_list, train_loss.detach().numpy())
            #    norm_list.append(get_norm([para.grad for para in model.parameters()]))
            #    np.save('norm_list.npy', np.array(norm_list))
    return total_privacy_l

def test(model, device, test_tuple, t):
    model.eval()
    test_loss = 0
    correct = 0
    data, target=test_tuple
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_num=data.shape[0]
    test_loss /= test_num
    accuracy=100. * correct / test_num
    print('At step %d: '%t)
    print('\nTest loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        accuracy))
    return accuracy

def main(argv):
    cuda = FLAGS.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda and not FLAGS.convex else "cpu")
    log_freq=20

    total_steps=max(FLAGS.max_steps, int(FLAGS.epoches/FLAGS.q)) #total steps

    #if(FLAGS.SGN==2):
    #    total_steps*=4

    if(FLAGS.dataset=='mnist'):
        train_tuple=MNIST_data().train()
        test_tuple=MNIST_data().test()
        if(FLAGS.convex):
            train_tuple=(train_tuple[0].reshape(-1, 784), train_tuple[1])
            test_tuple=(test_tuple[0].reshape(-1, 784), test_tuple[1])
            model = Logistic(FLAGS.dataset).to(device)
        else:
            #train_tuple=(train_tuple[0].reshape(-1, 784), train_tuple[1])
            #test_tuple=(test_tuple[0].reshape(-1, 784), test_tuple[1])
            model = ConvNet().to(device)
    elif(FLAGS.dataset=='covertype'):
        train_tuple=Covertype_data.train()
        test_tuple=Covertype_data.test()
        if(FLAGS.convex):
            model = Logistic(FLAGS.dataset).to(device)
        else:
            model = Linear().to(device)

    if(FLAGS.momentum==0):
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=0)
    elif(FLAGS.momentum>0 and FLAGS.momentum<=1):
        if(FLAGS.momentum==1):
            FLAGS.momentum=0.5
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
    else:
        optimizer = optim.Adam(model.parameters())
    if(FLAGS.delta==-1):
        FLAGS.delta=1./(train_tuple[0].shape[0]**2)

    diff=0
    if(FLAGS.delta != 0 and FLAGS.epoches!=-1):
        if(FLAGS.auto_sigma==0): 
            FLAGS.sigma=get_sigma(FLAGS.q, FLAGS.epoches, FLAGS.epsilon, FLAGS.delta)
        elif(FLAGS.auto_sigma==1):  
            FLAGS.SGN=0
            FLAGS.sigma=get_sigma(FLAGS.q, FLAGS.epoches, FLAGS.epsilon, FLAGS.delta)
            FLAGS.sigma*=2

    #FLAGS.sigma=20
    #recording information of this experiment instance
    experiment_info='Dataset: %r \nSampling probability: %r \nDelta: %r \nConvex: %r \nClip_bound: %r \nSigma: %r\nMomentum: %r\nAuto_sigma: %d\nSGN: %d \nEpoches: %d \nEpsilon: %r \n'%(FLAGS.dataset, FLAGS.q, FLAGS.delta,
    FLAGS.convex, FLAGS.clip_bound, FLAGS.sigma, FLAGS.momentum, FLAGS.auto_sigma, FLAGS.SGN, FLAGS.epoches, FLAGS.epsilon)
    logging(experiment_info, 'w')
    
    total_privacy_l=[0.]*128     #tracking alpha at different orders [1,128], can be converted to (epsilon,delta)-differential privacy
    epsilons=[0.5, 1., 2.0] 
    deltas=[0., 0., 2.0]     #one delta for one epsilon
    log_array=[]
    norm_list=[]
 
    for t in range(1, total_steps+1):
        #print(FLAGS.sigma, 'here')
        #get the gradients, notice the optimizer.step() is ran outside the train function.
        total_privacy_l=train(model, device, train_tuple, optimizer, diff, total_privacy_l, t, norm_list)
        if(FLAGS.delta>0 and FLAGS.delta<1):   #training privately
            all_failed=True
            for i, eps in enumerate(epsilons): 
                if(deltas[i]>FLAGS.delta):     #discarding the epsilon we already failed
                    continue
                #use rdp_accountant to get delta for given epsilon
                if_update_delta, order=_compute_delta(range(2,2+128), total_privacy_l, eps)
                #print(if_update_delta, 'hereheee')
                if(if_update_delta>FLAGS.delta):  #record the final model satisfies (eps,deltas[i])-differential privacy
                    accuracy=test(model, device, test_tuple, t)
                    info='For epislon %r, delta %r we get accuracy: %r%% at step %r\n'%(eps, deltas[i], accuracy, t)
                    deltas[i]=1.                  #abort current epsilon
                    logging(info)
                    print(info)
                else:
                    deltas[i]=if_update_delta  #update delta
                    all_failed=False           #still got at least one epsilon not failed
            if(not all_failed):
                optimizer.step()
            else :
                info='failed at all given epsilon, exiting\n'
                print(info)
                logging(info)
                exit()
        else: #training no privately
            optimizer.step()

        if(t%log_freq==0):
            #aa=1
            accuracy=test(model, device, test_tuple, t)
            log_array.append(copy.deepcopy([t, accuracy, epsilons, deltas]))
            np.save('logs/log%d.npy'%FLAGS.experiment_id, np.array(log_array, dtype=object))


    #np.save('norm_list.npy', np.array(norm_list, dtype=object))

if __name__ == '__main__':
    app.run(main)