time_str=`date +%Y%m%d%H%M%S`
mkdir nohup_logs/${time_str}
# 循环遍历 1-3 和 1-5
j=1;
CUDA_IDX=0;
for i in {1..15};
do 
    for k in {1..16};
    do
        echo $i, 0.0$j, $k >> temp.log;
        python main.py --dataname MUTAG --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/MUTAG.log;
        # python main.py --dataname IMDB-BINARY --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/IMDB-BINARY.log;
        # python main.py --dataname IMDB-MULTI --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/IMDB-MULTI.log;
        python main.py --dataname PROTEINS --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/PROTEINS.log;
        # python main.py --dataname REDDIT-BINARY --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/REDDIT-BINARY.log;
        python main.py --dataname NCI1 --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX}  >> nohup_logs/${time_str}/NCI1.log;
        # python main.py --dataname REDDIT-MULTI-5K --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/REDDIT-MULTI-5K.log;
        python main.py --dataname COLLAB --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/COLLAB.log;
        # python main.py --dataname DD --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX}  >> nohup_logs/${time_str}/DD.log;
        python main.py --dataname ENZYMES --depth $i --rate 0.$j  --ring_width $k  --cuda ${CUDA_IDX} >> nohup_logs/${time_str}/ENZYMES.log;
    done
    # for k in {2..7};
    # do
    #     echo $i, 0.$j, $k >> temp.log;
    #     python main.py --dataname MUTAG --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/MUTAG.log;
    #     python main.py --dataname IMDB-BINARY --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/IMDB-BINARY.log;
    #     python main.py --dataname IMDB-MULTI --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/IMDB-MULTI.log;
    #     python main.py --dataname PROTEINS --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/PROTEINS.log;
    #     python main.py --dataname REDDIT-BINARY --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/REDDIT-BINARY.log;
    #     python main.py --dataname NCI1 --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/NCI1.log;
    #     python main.py --dataname REDDIT-MULTI5K --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/REDDIT-MULTI5K.log;
    #     python main.py --dataname COLLAB --depth $i --rate 0.$j  --ring_width $k  >> nohup_logs/${time_str}/COLLAB.log;
    # done
done