class Config:
    # Paths and directories
    root_path = "./sprna/"  # Root directory
    model_file = "./sprna/model.ckpt"  # Path to value network
    train_dir = './sprna/train/'  # Results directory
    eval_dir = './sprna/eval/'  # Evaluation results directory

    # Training parameters
    episodes = 500  # Number of episodes to play
    mx_epochs = 30  # Max epochs to play
    batch_size = 64  # Batch size
    replay_size = 100000  # Replay memory size
    sample_train = 10000  # Replay sample size to train on
    sample_test = 10000  # Replay sample size to test on
    batch_playout_size = 10  # Batch size for the playout
    optimizer = "adam"  # Optimizer
    criterion = "mse"  # Loss
    lr = 0.001  # Learning rate
    # Testing parameters
    test_size = 0.1  # Test size percentage

    # RNA parameters
    W = 8  # The feature parameter W
    maxseq_len = 400  # Max sequence length to be designed
    single_sites = ["G","A","U","C"]
    paired_sites = ["GC","CG","AU","UA","UG","GU"]
    #single_sites = ["Q","W","E","R"]
    #paired_sites = ["bq","bw","be","br","bt","by"]

    # other parameters
    trackprint = False
    num_workers = 1  # Number of workers
    log_train = True  # Log train results
    debug = False
    debug_test = True