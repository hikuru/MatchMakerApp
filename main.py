import MatchMakerApp as mm_app
import matchmaker.helper_funcs
import matchmaker.MatchMaker as mm
import tensorflow as tf
import os

args = mm_app.argument_parser()

num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
GPU = True
if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

layers = mm_app.read_architecture() # read architecture file and store layer parameters
modelName = args.saved_model_name # name of the model to save the weights

# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

isPredict = True

while isPredict:
    # get drug and cell line information from user
    cid1, cid2, cl, cl_id, chem1, chem2, cl_exp = mm_app.get_user_input()

    # prepare input format for MatchMaker
    pair = mm_app.prep_pair(chem1, chem2, cl_exp)

    # generate MatchMaker network architecture
    model = mm.generate_network(pair, layers, modelName, inDrop, drop)

    # load the model weights
    model.load_weights('matchmaker/'+modelName+'.h5')

    # predict in Drug1, Drug2 order
    pred1 = mm.predict(model, [pair['drug1'],pair['drug2']])
    # predict in Drug2, Drug1 order
    pred2 = mm.predict(model, [pair['drug2'],pair['drug1']])
    # take the mean for final prediction
    pred = (pred1 + pred2) / 2
    print("The predicted synergy is {}".format(pred[0]))
    yesNo = input("Do you want to continue to predict synergy? (yes/no) ") 
    if yesNo == "yes":
        isPredict = True
    else:
        isPredict = False

