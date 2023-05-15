time_str=`date +%Y%m%d%H%M%S`
mkdir nohup_logs/${time_str}
# 循环遍历 1-3 和 1-5
for i in {1..3};
do 
    for j in {1..10};
    do
    echo $i, 0.0$j >> temp.log;
    # python main.py --dataname COLLAB --depth $i --rate 0.$j >> nohup_logs/${time_str}/COLLAB.log;
    # python main.py --dataname REDDIT-MULTI5K --depth $i --rate 0.$j >> nohup_logs/${time_str}/REDDIT-MULTI5K.log;
    python main.py --dataname IMDB-BINARY --depth $i --rate 0.$j >> nohup_logs/${time_str}/IMDB-BINARY.log;
    python main.py --dataname IMDB-MULTI --depth $i --rate 0.$j >> nohup_logs/${time_str}/IMDB-MULTI.log;
    done
    for j in {2..7};
    do
    echo $i, 0.$j >> temp.log;
    # python main.py --dataname COLLAB --depth $i --rate 0.$j >> nohup_logs/${time_str}/COLLAB.log;
    # python main.py --dataname REDDIT-MULTI-5K --depth $i --rate 0.$j >> nohup_logs/${time_str}/REDDIT-MULTI-5K.log;
    python main.py --dataname IMDB-BINARY --depth $i --rate 0.$j >> nohup_logs/${time_str}/IMDB-BINARY.log;
    python main.py --dataname IMDB-MULTI --depth $i --rate 0.$j >> nohup_logs/${time_str}/IMDB-MULTI.log;
    done
done