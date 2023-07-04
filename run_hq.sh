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
python main.py --cuda $CUDA_IDX --dataname MUTAG > nohup_logs/${time_str}/MUTAG.log; # done
python main.py --cuda $CUDA_IDX --dataname PROTEINS > nohup_logs/${time_str}/PROTEINS.log; # done
python main.py --cuda $CUDA_IDX --dataname REDDIT-BINARY > nohup_logs/${time_str}/REDDIT-BINARY.log;

python main.py --cuda $CUDA_IDX --dataname COLLAB > nohup_logs/${time_str}/COLLAB.log;
python main.py --cuda $CUDA_IDX --dataname DD > nohup_logs/${time_str}/DD.log;
python main.py --cuda $CUDA_IDX --dataname NCI1 > nohup_logs/${time_str}/NCI1.log;

python main.py --cuda $CUDA_IDX --dataname ENZYMES > nohup_logs/${time_str}/ENZYMES.log;
python main.py --cuda $CUDA_IDX --dataname PTC_MR > nohup_logs/${time_str}/PTC_MR.log; # done
python main.py --cuda $CUDA_IDX --dataname NCI109 > nohup_logs/${time_str}/NCI109.log;

python main.py --cuda $CUDA_IDX --dataname REDDIT-MULTI-5K > nohup_logs/${time_str}/REDDIT-MULTI-5K.log;