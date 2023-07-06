# CUDA_IDX=0;
# time_str=`date +%Y%m%d%H%M%S`
# mkdir nohup_logs/${time_str}
# python main.py --cuda $CUDA_IDX --dataname MUTAG > nohup_logs/${time_str}/MUTAG.log;
# python main.py --cuda $CUDA_IDX --dataname PROTEINS > nohup_logs/${time_str}/PROTEINS.log;
# python main.py --cuda $CUDA_IDX --dataname REDDIT-BINARY > nohup_logs/${time_str}/REDDIT-BINARY.log;

# CUDA_IDX=1;
# time_str=`date +%Y%m%d%H%M%S`
# mkdir nohup_logs/${time_str}
# python main.py --cuda $CUDA_IDX --dataname COLLAB > nohup_logs/${time_str}/COLLAB.log;
# python main.py --cuda $CUDA_IDX --dataname DD > nohup_logs/${time_str}/DD.log;
# python main.py --cuda $CUDA_IDX --dataname NCI1 > nohup_logs/${time_str}/NCI1.log;

# CUDA_IDX=2;
# time_str=`date +%Y%m%d%H%M%S`
# mkdir nohup_logs/${time_str}
# python main.py --cuda $CUDA_IDX --dataname ENZYMES > nohup_logs/${time_str}/ENZYMES.log;
# python main.py --cuda $CUDA_IDX --dataname PTC_MR > nohup_logs/${time_str}/PTC_MR.log; # running
# python main.py --cuda $CUDA_IDX --dataname NCI109 > nohup_logs/${time_str}/NCI109.log;

# CUDA_IDX=3;
# time_str=`date +%Y%m%d%H%M%S`
# mkdir nohup_logs/${time_str}
# python main.py --cuda $CUDA_IDX --dataname REDDIT-MULTI-5K > nohup_logs/${time_str}/REDDIT-MULTI-5K.log;


CUDA_IDX=0;
time_str=`date +%Y%m%d%H%M%S`
mkdir nohup_logs/${time_str}
# for i in {1..10};
# do

# Namespace(alpha=0.8, cuda=0, dataname='ENZYMES', depth=5, epochs=100, hidden=512, layer=2, lr=0.0001, out_hidden=64, rate=0.1, ring_width=1, w=1e-05, warmup=100.0)
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname ENZYMES --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 1 --loss_fn sce >> nohup_logs/${time_str}/ENZYMES.log; # base
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname ENZYMES --depth 5 --rate 0.004 --contrast_with_central_nodes 1 --ring_width 1 --loss_fn sce >> nohup_logs/${time_str}/ENZYMES.log; # contrast_with_central_nodes
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname ENZYMES --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 6 --loss_fn sce >> nohup_logs/${time_str}/ENZYMES.log; # ring_width
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname ENZYMES --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 1 --loss_fn mse >> nohup_logs/${time_str}/ENZYMES.log; # loss_fn

# ptc-mr

# Namespace(dataname='NCI109', cuda=0, depth=2, rate=0.2, ring_width=2, out_hidden=64, hidden=512, epochs=100, lr=1e-05, warmup=100.0, w=1e-05, acc='80.2+-1.9', alpha=0.3, layer=3)
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname NCI109 --depth 2 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # base
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname NCI109 --depth 2 --rate 0.004 --contrast_with_central_nodes 1 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # contrast_with_central_nodes
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname NCI109 --depth 2 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 3 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # ring_width
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname NCI109 --depth 2 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn mse >> nohup_logs/${time_str}/NCI109.log; # loss_fn


# Namespace(acc='80.97+-2.7', alpha=0.5, cuda=0, dataname='NCI1', depth=5, epochs=20, hidden=512, layer=3, lr=1e-05, out_hidden=64, rate=0.1, ring_width=2, w=0.0001, warmup=100.0)
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 20 --dataname NCI1 --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/NCI1.log; # base
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 20 --dataname NCI1 --depth 5 --rate 0.004 --contrast_with_central_nodes 1 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/NCI1.log; # contrast_with_central_nodes
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 20 --dataname NCI1 --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 6 --loss_fn sce >> nohup_logs/${time_str}/NCI1.log; # ring_width
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 20 --dataname NCI1 --depth 5 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn mse >> nohup_logs/${time_str}/NCI1.log; # loss_fn

# valid-Namespace(acc=0.8994152046783626, alpha=0.17717860906319477, cmd_first=1, contrast_with_central_nodes=0, cuda=0, dataname='MUTAG', depth=5, epochs=100, hidden=512, layer=1, lr=0.0001, rate=0.072, ring_width=2, w=1e-06, warmup=61.39677578223307)
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname MUTAG --depth 5 --rate 0.072 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/MUTAG.log; # base
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname MUTAG --depth 5 --rate 0.072 --contrast_with_central_nodes 1 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/MUTAG.log; # contrast_with_central_nodes
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname MUTAG --depth 5 --rate 0.072 --contrast_with_central_nodes 0 --ring_width 6 --loss_fn sce >> nohup_logs/${time_str}/MUTAG.log; # ring_width
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname MUTAG --depth 5 --rate 0.072 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn mse >> nohup_logs/${time_str}/MUTAG.log; # loss_fn

# Namespace(acc='75.83+-2.1', alpha=0.5, cuda=0, dataname='PROTEINS', depth=3, epochs=100, hidden=512, layer=2, lr=1e-05, out_hidden=64, rate=0.1, ring_width=2, w=1e-05, warmup=100.0)
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname PROTEINS --depth 3 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/PROTEINS.log; # base
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname PROTEINS --depth 3 --rate 0.004 --contrast_with_central_nodes 1 --ring_width 2 --loss_fn sce >> nohup_logs/${time_str}/PROTEINS.log; # contrast_with_central_nodes
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname PROTEINS --depth 3 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 4 --loss_fn sce >> nohup_logs/${time_str}/PROTEINS.log; # ring_width
    python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname PROTEINS --depth 3 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 2 --loss_fn mse >> nohup_logs/${time_str}/PROTEINS.log; # loss_fn

# Namespace(dataname='DD', cuda=0, depth=12, rate=0.4, ring_width=12, out_hidden=64, hidden=512, epochs=50, lr=0.0001, warmup=100.0, w=1e-05, acc='78.61+-4.3', alpha=0.5, layer=3)
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 50 --dataname DD --depth 12 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 12 --loss_fn sce >> nohup_logs/${time_str}/DD.log; # base
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 50 --dataname DD --depth 12 --rate 0.004 --contrast_with_central_nodes 1 --ring_width 12 --loss_fn sce >> nohup_logs/${time_str}/DD.log; # contrast_with_central_nodes
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 50 --dataname DD --depth 12 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 13 --loss_fn sce >> nohup_logs/${time_str}/DD.log; # ring_width
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 50 --dataname DD --depth 12 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 12 --loss_fn mse >> nohup_logs/${time_str}/DD.log; # loss_fn




# Namespace(dataname='REDDIT-BINARY', cuda=1, depth=10, rate=0.1, ring_width=7, out_hidden=64, hidden=512, epochs=100, lr=0.001, warmup=100.0, w=1e-05, acc='55.17+-8.4,0.7', alpha=0.5, layer=2)
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 100 --dataname REDDIT-BINARY --depth 10 --rate 0.004 --contrast_with_central_nodes 0 --ring_width 7 --loss_fn sce >> nohup_logs/${time_str}/REDDIT-BINARY.log; # base



# NCI109: acc: 0.8005853452126284 alpha: 0.1815153326691898 contrast_with_central_nodes: 1 depth: 14 epochs: 740.0 hidden: 0 layer: 2 lr: 1 rate: 0.036000000000000004 ring_width: 15.0 w: 2 warmup: 47.43640631891315
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 740 --dataname NCI109 --depth 14 --rate 0.036 --contrast_with_central_nodes 0 --ring_width 15 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # base
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 740 --dataname NCI109 --depth 14 --rate 0.036 --contrast_with_central_nodes 1 --ring_width 15 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # contrast_with_central_nodes
    # # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 740 --dataname NCI109 --depth 14 --rate 0.036 --contrast_with_central_nodes 0 --ring_width 15 --loss_fn sce >> nohup_logs/${time_str}/NCI109.log; # ring_width
    # python main.py --cmd_first 1 --cuda $CUDA_IDX --epochs 740 --dataname NCI109 --depth 14 --rate 0.036 --contrast_with_central_nodes 0 --ring_width 15 --loss_fn mse >> nohup_logs/${time_str}/NCI109.log; # loss_fn


    # python main.py --cuda $CUDA_IDX --dataname PROTEINS > nohup_logs/${time_str}/PROTEINS.log; # done
    # python main.py --cuda $CUDA_IDX --dataname REDDIT-BINARY > nohup_logs/${time_str}/REDDIT-BINARY.log;

    # python main.py --cuda $CUDA_IDX --dataname COLLAB > nohup_logs/${time_str}/COLLAB.log;
    # python main.py --cuda $CUDA_IDX --dataname DD > nohup_logs/${time_str}/DD.log;
    # python main.py --cuda $CUDA_IDX --dataname NCI1 > nohup_logs/${time_str}/NCI1.log;

    # python main.py --cuda $CUDA_IDX --dataname ENZYMES > nohup_logs/${time_str}/ENZYMES.log;
    # python main.py --cuda $CUDA_IDX --dataname PTC_MR > nohup_logs/${time_str}/PTC_MR.log; # done
    # python main.py --cuda $CUDA_IDX --dataname NCI109 > nohup_logs/${time_str}/NCI109.log;

    # python main.py --cuda $CUDA_IDX --dataname REDDIT-MULTI-5K > nohup_logs/${time_str}/REDDIT-MULTI-5K.log;
# done