time_str=`date +%Y%m%d%H%M%S`

# 循环遍历 1-3 和 1-5
for i in {1..3};
do 
    for j in {1..5};
    do
    echo $i, 0.$j;
    python main.py --dataname COLLAB --depth $i --rate 0.$j >> nohup_logs/${time_str}_COLLAB.log;
    done
done