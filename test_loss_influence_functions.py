import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-training_steps", type=int)
parser.add_argument("-dampening", type=float)
parser.add_argument("-cg_iters", type=int)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import uuid
import numpy as np
import string
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from influence import Influence
from mnist_models import Model
from utils import corrupt_mnist, Dataset, leave_one_out_dataset
from IPython.display import clear_output

np.random.seed(1)

FLAGS = None

# Import data
#mnist = corrupt_mnist(input_data.read_data_sets("mnist_data", one_hot=True), 0.0)
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

testset = Dataset( (mnist.test.images, mnist.test.labels) )
trainset = Dataset ((mnist.train.images, mnist.train.labels) )

sess = tf.InteractiveSession()
model = Model("convnet", trainset, testset, sess, training_steps=args.training_steps)
saver = tf.train.Saver()
tf.global_variables_initializer().run()
summary_writter = tf.summary.FileWriter("./Hvp_summaries", sess.graph)

#checkpoint_file = "/home/cgel/deep_quad/checkpoints/convnet.ckpt"
checkpoint_file = "/home/cgel/deep_quad/checkpoints/"+str(uuid.uuid4())

print("\nTraining the model")
model.train(vervose=2)
saver.save(sess, checkpoint_file)

print("\nCreating Influence class")
scale = float(len(trainset.labels)) * 10
inf = Influence(model.cross_entropy, testset, model.cross_entropy, trainset, model.input_ph, model.y_, scale, cg_iters = args.cg_iters, dampening=args.dampening, vervose=1)
inf.compute_s()

# compute the influences for every image in the training set
print("\nComputing the influences for every image in the training set")
trainset_influence_map = []
trainset_influence = []
#for i in range(len(mnist.train.labels)):
for i in range(500):
    z = (mnist.train.images[i:i+1], mnist.train.labels[i:i+1])
    influ = inf.of(z)
    trainset_influence.append(influ)
    trainset_influence_map.append( (influ, i))
    if (i+1) % 10000==0:
        print("progress: %.2f%%"%(float(i)/len(mnist.train.labels)) )
print("Done")


# generate a list ordering the examples of the training set by most influential
abs_influence = [ (abs(i_influ), i) for i_influ, i in trainset_influence_map]
abs_influence.sort(reverse=True)
rank = [ (trainset_influence[j], j) for _, j in abs_influence]

subset_infs = []
losses_2k = []
losses_10k = []
losses_30k = []
losses_50k = []
# leave one out retraining 
print("\nLeave one out retraining")
#subset = rank[0:20]
subset = rank[0:1]
saver.restore(sess, checkpoint_file)
base_testset_loss = model.testset_loss()
for j in range(len(subset)):
    _, i = subset[j]
    z = (trainset.images[i:i+1], trainset.labels[i:i+1])
    z_influ = inf.of(z)
    subset_infs.append(z_influ )
    model.trainset = leave_one_out_dataset(trainset, i)
    model.update(2000, learning_rate=1e-7)
    losses_2k.append(model.testset_loss())
    #model.update(8000, learning_rate=1e-7)
    losses_10k.append(model.testset_loss())
    #model.update(20000, learning_rate=1e-7)
    losses_30k.append(model.testset_loss())
    #model.update(20000, learning_rate=1e-7)
    losses_50k.append(model.testset_loss())
    print(j, ":  ", z_influ, model.testset_loss())
    saver.restore(sess, checkpoint_file)
    
report =  "--- Arguments ---"
report += "\nTraining steps: "+str( model.training_step_count)
report += "\nDampening: "+ str(args.dampening)
report += "\nCg_iters: "+ str(args.cg_iters)

report += "\n\n--- Model info ---"
report += "\ntrain_loss: "+ str(model.evaluate_on(trainset))
report += "\nTrain_acc: "+ str(model.evaluate_accuracy_on(trainset))
report += "\nTest_loss: "+ str(model.evaluate_on(testset))
report += "\nTest_acc: "+ str(model.evaluate_accuracy_on(testset))

report += "\n\n--- Influence correlation ---"
report += "\n2k: "+ str(np.corrcoef(losses_2k, subset_infs)[0][1])
report += "\n10k: "+ str(np.corrcoef(losses_10k, subset_infs)[0][1])
report += "\n30k: "+ str(np.corrcoef(losses_30k, subset_infs)[0][1])
report += "\n50k: "+ str(np.corrcoef(losses_50k, subset_infs)[0][1])
print(report)

log_filename = "~/deep_quad/logs/tl_inf_"+ str(args.training_steps) +"_"+ str(args.dampening) +"_" + str(args.cg_iters) + "_" + str(uuid.uuid4())

f = open(log_filename, "w")

def list_to_str(f, l):
    return "["+string.join(map(str, l), ", ") + "]"

f.write(report)
f.write("\n\n\n")
f.write("subset_infs = "+list_to_str(subset_infs))
f.write("losses_2k = "+list_to_str(losses_2k))
f.write("losses_10k = "+list_to_str(losses_10k))
f.write("losses_30k = "+list_to_str(losses_30k))
f.write("losses_50k = "+list_to_str(losses_50k))

f.close()