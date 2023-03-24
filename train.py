import os,sys
import torch
import importlib
from utils import options
from utils.util import log
def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training Level-S2fM)".format(sys.argv[0]))
    opt_cmd = options.parse_arguments(sys.argv[1:])      # indicate the parameters after train.py
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)
    # torch.set_default_dtype(getattr(torch,opt.prec))
    # config the opt
    with torch.cuda.device(opt.device):
        model = importlib.import_module("pipelines.{}".format(opt.pipeline))
        m = model.Model(opt)
        m.load_dataset(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)
        m.train(opt)

if __name__=="__main__":
    main()
